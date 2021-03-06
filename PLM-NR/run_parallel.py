import numpy as np
import torch
import pickle
import hashlib
import logging
from tqdm.auto import tqdm
import torch.optim as optim
from pathlib import Path
import utils
import os
import sys
import csv
import logging
import datetime
import time
from dataloader_parallel import DataLoaderTrain, DataLoaderTest
from torch.utils.data import Dataset, DataLoader
from preprocess import read_news, read_news_lm, get_doc_input, get_doc_input_lm
from model_bert import Model
from parameters import parse_args

from transformers import AutoTokenizer, AutoModel, AutoConfig
from run import train


def test(args):
    start_time = time.time()

    if args.enable_hvd:
        import horovod.torch as hvd

    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(
        args.enable_hvd, args.enable_gpu)

    if args.load_ckpt_test is not None:
        #TODO: choose ckpt_path
        ckpt_path = utils.get_checkpoint(os.path.expanduser(args.model_dir), args.load_ckpt_test)
    else:
        ckpt_path = utils.latest_checkpoint(os.path.expanduser(args.model_dir))

    assert ckpt_path is not None, 'No ckpt found'
    if args.enable_gpu:
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
    else:
        checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

    if 'subcategory_dict' in checkpoint:
        subcategory_dict = checkpoint['subcategory_dict']
    else:
        subcategory_dict = {}

    category_dict = checkpoint['category_dict']
    word_dict = checkpoint['word_dict']
    domain_dict = checkpoint['domain_dict']
    user_dict = checkpoint['user_dict']
    pretrain_lm_path = os.path.expanduser(args.pretrain_lm_path)  # or by name e.g. "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(pretrain_lm_path))
    config = AutoConfig.from_pretrained(os.path.expanduser(pretrain_lm_path), output_hidden_states=True)
    language_model = AutoModel.from_pretrained(os.path.expanduser(pretrain_lm_path),config=config)

    # auto adjust hyper parameter by pre-trained model config
    args.num_layers = config.num_hidden_layers if args.num_layers is None else args.num_layers
    args.word_embedding_dim = config.hidden_size if args.word_embedding_dim is None else args.word_embedding_dim


    model = Model(args, language_model, len(user_dict), len(category_dict), len(domain_dict), len(subcategory_dict))

    if args.enable_gpu:
        model.cuda()

    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model loaded from {ckpt_path}")

    if args.enable_hvd:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    model.eval()
    torch.set_grad_enabled(False)

    # save 1~2 minutes time, manually delete the cache file if cache is outdated
    news_cache_path = os.path.join(os.path.expanduser(args.root_data_dir), f'{args.dataset}/{args.test_dir}/news_cache.pkl')
    if os.path.exists(news_cache_path):
        news, news_index, _, _, _ = pickle.load(open(news_cache_path, "rb"))
    else:
        news, news_index, _, _, _ = read_news_lm(
            os.path.join(os.path.expanduser(args.root_data_dir),
                        f'{args.dataset}/{args.test_dir}/news.tsv'), 
            args,
            tokenizer
        )
        pickle.dump((news, news_index, category_dict, domain_dict, subcategory_dict), open(news_cache_path, "wb"))

    news_title, news_title_type, news_title_attmask, \
    news_abstract, news_abstract_type, news_abstract_attmask, \
    news_body, news_body_type, news_body_attmask, \
    news_category, news_domain, news_subcategory = get_doc_input_lm(
        news, news_index, category_dict, domain_dict, subcategory_dict, args)

    news_combined = np.concatenate([
    x for x in
    [news_title, news_title_type, news_title_attmask, \
    news_abstract, news_abstract_type, news_abstract_attmask, \
    news_body, news_body_type, news_body_attmask, \
    news_category, news_domain, news_subcategory]
    if x is not None], axis=1)


    class NewsDataset(Dataset):
        def __init__(self, data):
            self.data = data

        def __getitem__(self, idx):
            return self.data[idx]

        def __len__(self):
            return self.data.shape[0]

    def news_collate_fn(arr):
        arr = torch.LongTensor(arr)
        return arr

    news_dataset = NewsDataset(news_combined)
    news_dataloader = DataLoader(news_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                collate_fn=news_collate_fn)

    news_scoring = []
    with torch.no_grad():
        for input_ids in tqdm(news_dataloader):
            if args.enable_gpu:
                input_ids = input_ids.cuda()
            news_vec = model.news_encoder(input_ids)
            news_vec = news_vec.to(torch.device("cpu")).detach().numpy()
            # news_vec = np.zeros((len(input_ids), 64))
            news_scoring.extend(news_vec)

    news_scoring = np.array(news_scoring)

    logging.info("news scoring num: {}".format(news_scoring.shape[0]))

    dataloader = DataLoaderTest(
        news_index=news_index,
        news_scoring=news_scoring,
        user_dict=user_dict,
        word_dict=word_dict,
        news_bias_scoring= None,
        data_dir=os.path.join(os.path.expanduser(args.root_data_dir),
                            f'{args.dataset}/{args.test_dir}'),
        filename_pat=args.filename_pat,
        args=args,
        worker_size=hvd_size,
        worker_rank=hvd_rank,
        cuda_device_idx=hvd_local_rank,
        enable_prefetch=True,
        enable_shuffle=False,
        enable_gpu=args.enable_gpu,
    )

    from metrics import roc_auc_score, ndcg_score, mrr_score, ctr_score

    AUC = []
    MRR = []
    nDCG5 = []
    nDCG10 = []
    SCORE = []

    outfile_metrics = os.path.join(os.path.expanduser(args.model_dir), "metrics_{}_{}_{}.tsv".format(ckpt_path.rsplit('/',1)[1].rsplit('.',1)[0], datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"), hvd_local_rank))
    def print_metrics(hvd_local_rank, cnt, mean_metrics, save=True):
        metrics = "[{}] Ed: {}: {}".format(hvd_local_rank, cnt, \
            '\t'.join(["{:0.2f}".format(i * 100) for i in mean_metrics]))
        logging.info(metrics)
        # save the metrics result
        def write_tsv(etrics):
            with open(outfile_metrics, 'a') as out_file:
                out_file.write(metrics + '\n')
            logging.info(f"Saved metrics to {outfile_metrics}")
        if save:
            write_tsv(metrics)
        return metrics

    def get_mean(arr):
        return [np.array(i).mean() for i in arr]

    def score_func(model, batch):
        user_ids, log_vecs, log_mask, impression_ids, news_vecs, news_bias, labels = [], [], [], [], [], [], []
        for (impression_id, user_id, log_vec, mask, news_vec, bias, label) in batch:
            if args.ignore_unseen_user and user_id[0] == 0:
                continue
            if args.only_unseen_user and user_id[0] != 0:
                continue
            impression_ids.append(impression_id)
            user_ids.append(user_id)
            log_vecs.append(log_vec)
            log_mask.append(mask)
            news_vecs.append(news_vec)
            news_bias.append(bias)
            labels.append(label)
        if args.enable_gpu:
            user_ids = torch.LongTensor(user_ids).cuda()
            log_vecs = torch.FloatTensor(log_vecs).cuda()
            log_mask = torch.FloatTensor(log_mask).cuda()
        else:
            user_ids = torch.LongTensor(user_ids)
            log_vecs = torch.FloatTensor(log_vecs)
            log_mask = torch.FloatTensor(log_mask)

        if args.enable_gpu:
            user_ids = user_ids.cuda(non_blocking=True)
            log_vecs = log_vecs.cuda(non_blocking=True)
            log_mask = log_mask.cuda(non_blocking=True)

        if (args.ignore_unseen_user or args.only_unseen_user) and len(user_ids) == 0:
            return []

        user_vecs = model.user_encoder(user_ids, log_vecs, log_mask)

        scores = []
        for impression_id, user_vec, news_vec, bias, label in zip(
                impression_ids, user_vecs, news_vecs, news_bias, labels):

            if label.mean() == 0 or label.mean() == 1:
                continue

            if args.enable_gpu:
                news_vec = torch.FloatTensor(news_vec).cuda()
            else:
                news_vec = torch.FloatTensor(news_vec)

            user_vec = user_vec.unsqueeze(0)
            news_vec = news_vec.unsqueeze(0)
            score = model.scoring(news_vec, user_vec).to(torch.device("cpu")).detach().numpy().squeeze(0)

            # label is -1 is for test set and prediction only
            if(np.all(label == -1)):
                scores.append([impression_id, score])
                continue

            auc = roc_auc_score(label, score)
            mrr = mrr_score(label, score)
            ndcg5 = ndcg_score(label, score, k=5)
            ndcg10 = ndcg_score(label, score, k=10)

            AUC.append(auc)
            MRR.append(mrr)
            nDCG5.append(ndcg5)
            nDCG10.append(ndcg10)

        return scores

    for cnt, batch in enumerate(dataloader):
        # enable parallel scoring
        SCORE.extend(score_func(model, batch))
        # SCORE.extend(utils.parallel(score_func, batch,
        #                             int(args.batch_size/8), 8, 'threads', False,  # batch_size, n_jobs, synchronize
        #                             model))

        if cnt % args.log_steps == 0:
            print_metrics(hvd_rank, cnt * args.batch_size, get_mean([AUC, MRR, nDCG5,  nDCG10]), save=False)

    # stop scoring
    dataloader.join()

    # print and save metrics
    if len(AUC) > 0:
        logging.info("Print final metrics")
        print_metrics(hvd_rank, cnt * args.batch_size, get_mean([AUC, MRR, nDCG5,  nDCG10]))

    # format the score: ImpressionID [Rank-of-News1,Rank-of-News2,...,Rank-of-NewsN]
    logging.info("Process scoring")
    for score in tqdm(SCORE):
        argsort = np.argsort(-score[1])
        ranks = np.empty_like(argsort)
        ranks[argsort] = np.arange(len(score[1]))
        score[1] = (ranks + 1).tolist()

    # save the prediction result
    outfile_prediction = os.path.join(os.path.expanduser(args.model_dir), "prediction_{}_{}_{}.tsv".format(ckpt_path.rsplit('/',1)[1].rsplit('.',1)[0], datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"), hvd_local_rank))
    def write_tsv(score):
        with open(outfile_prediction, 'wt') as out_file:
            tsv_writer = csv.writer(out_file, delimiter='\t')
            tsv_writer.writerows(score)
        logging.info(f"Saved scoring to {outfile_prediction}")
    if len(SCORE) > 0:
        write_tsv(SCORE)

    logging.info(f"Time taken: {time.time() - start_time}")


if __name__ == "__main__":
    utils.setuplogger()
    args = parse_args()
    Path(os.path.expanduser(args.model_dir)).mkdir(parents=True, exist_ok=True)
    if 'train' in args.mode:
        train(args)
    if 'test' in args.mode:
        test(args)
