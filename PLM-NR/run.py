import faulthandler
import signal
faulthandler.register(signal.SIGUSR1.value)
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
import datetime
import time
from dataloader import DataLoaderTrain, DataLoaderTest
from torch.utils.data import Dataset, DataLoader
from preprocess import read_news, read_news_bert, get_doc_input, get_doc_input_bert
from model_bert import ModelBert
from parameters import parse_args
# from torchsummary import summary

from transformers import AutoTokenizer, AutoModel, AutoConfig, get_scheduler, AdamW

finetuneset={
'encoder.layer.10.attention.self.query.weight',
'encoder.layer.10.attention.self.query.bias',
'encoder.layer.10.attention.self.key.weight',
'encoder.layer.10.attention.self.key.bias',
'encoder.layer.10.attention.self.value.weight',
'encoder.layer.10.attention.self.value.bias',
'encoder.layer.10.attention.output.dense.weight',
'encoder.layer.10.attention.output.dense.bias',
'encoder.layer.10.attention.output.LayerNorm.weight',
'encoder.layer.10.attention.output.LayerNorm.bias',
'encoder.layer.10.intermediate.dense.weight',
'encoder.layer.10.intermediate.dense.bias',
'encoder.layer.10.output.dense.weight',
'encoder.layer.10.output.dense.bias',
'encoder.layer.10.output.LayerNorm.weight',
'encoder.layer.10.output.LayerNorm.bias',
'encoder.layer.11.attention.self.query.weight',
'encoder.layer.11.attention.self.query.bias',
'encoder.layer.11.attention.self.key.weight',
'encoder.layer.11.attention.self.key.bias',
'encoder.layer.11.attention.self.value.weight',
'encoder.layer.11.attention.self.value.bias',
'encoder.layer.11.attention.output.dense.weight',
'encoder.layer.11.attention.output.dense.bias',
'encoder.layer.11.attention.output.LayerNorm.weight',
'encoder.layer.11.attention.output.LayerNorm.bias',
'encoder.layer.11.intermediate.dense.weight',
'encoder.layer.11.intermediate.dense.bias',
'encoder.layer.11.output.dense.weight',
'encoder.layer.11.output.dense.bias',
'encoder.layer.11.output.LayerNorm.weight',
'encoder.layer.11.output.LayerNorm.bias',
'pooler.dense.weight',
'pooler.dense.bias',
'rel_pos_bias.weight',
'classifier.weight',
'classifier.bias'}
def train(args):
    # Only support title Turing now
    assert args.enable_hvd  # TODO
    if args.enable_hvd:
        import horovod.torch as hvd

    if args.load_ckpt_name is not None:
        #TODO: choose ckpt_path
        ckpt_path = utils.get_checkpoint(os.path.expanduser(os.path.expanduser(args.model_dir)), args.load_ckpt_name)
    else:
        ckpt_path = utils.latest_checkpoint(os.path.expanduser(os.path.expanduser(args.model_dir)))

    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(
        args.enable_hvd, args.enable_gpu)

    pretrain_lm_path = os.path.expanduser(args.pretrain_lm_path)  # or by name "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(pretrain_lm_path))
    config = AutoConfig.from_pretrained(os.path.expanduser(pretrain_lm_path), output_hidden_states=True)
    bert_model = AutoModel.from_pretrained(os.path.expanduser(pretrain_lm_path), config=config)

    #bert_model.load_state_dict(torch.load('../bert_encoder_part.pkl'))
    # freeze parameters
    for name,param in bert_model.named_parameters():
        if name not in finetuneset:
            param.requires_grad = False

    news, news_index, category_dict, domain_dict, subcategory_dict = read_news_bert(
        os.path.join(os.path.expanduser(args.root_data_dir),
                    f'{args.dataset}/{args.train_dir}/news.tsv'), 
        args,
        tokenizer
    )

    news_title, news_title_type, news_title_attmask, \
    news_abstract, news_abstract_type, news_abstract_attmask, \
    news_body, news_body_type, news_body_attmask, \
    news_category, news_domain, news_subcategory = get_doc_input_bert(
        news, news_index, category_dict, domain_dict, subcategory_dict, args)

    news_combined = np.concatenate([
        x for x in
        [news_title, news_title_type, news_title_attmask, \
            news_abstract, news_abstract_type, news_abstract_attmask, \
            news_body, news_body_type, news_body_attmask, \
            news_category, news_domain, news_subcategory]
        if x is not None], axis=1)

    model = ModelBert(args, bert_model, len(category_dict), len(domain_dict), len(subcategory_dict))
    word_dict = None

    if args.enable_gpu:
        model = model.cuda()

    if args.enable_incremental:
        assert ckpt_path is not None, 'No ckpt found'
        if args.enable_gpu:
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        category_dict = checkpoint['category_dict']
        subcategory_dict = checkpoint['subcategory_dict']
        word_dict = checkpoint['word_dict']
        domain_dict = checkpoint['domain_dict']
        model.load_state_dict(checkpoint['model_state_dict'])
        logging.info(f"Model loaded from {ckpt_path} for incremental training")

    lr_scaler = hvd.local_size()
    if args.optimizer == 'Adam':
        optimizer = optim.Adam(model.parameters(), lr=args.lr)
    elif args.optimizer == 'AdamW':
        optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay, correct_bias=args.correct_bias)
    else:
        optimizer = AdamW(model.parameters(), lr=args.lr)

    num_training_steps = args.epochs * args.max_steps_per_epoch
    if args.enable_lr_scheduler:
        lr_scheduler = get_scheduler(
            "linear",
            optimizer=optimizer,
            num_warmup_steps=args.num_warmup_steps,
            num_training_steps=num_training_steps
        )

    if args.enable_hvd:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)
        hvd.broadcast_optimizer_state(optimizer, root_rank=0)
        compression = hvd.Compression.none
        optimizer = hvd.DistributedOptimizer(
            optimizer,
            named_parameters=model.named_parameters(),
            compression=compression,
            op=hvd.Average)


    dataloader = DataLoaderTrain(
        news_index=news_index,
        news_combined=news_combined,
        word_dict=word_dict,
        data_dir=os.path.join(os.path.expanduser(args.root_data_dir),
                            f'{args.dataset}/{args.train_dir}'),
        filename_pat=args.filename_pat,
        args=args,
        worker_size=hvd_size,
        worker_rank=hvd_rank,
        cuda_device_idx=hvd_local_rank,
        enable_prefetch=True,
        enable_shuffle=True,
        enable_gpu=args.enable_gpu,
    )

    outfile = os.path.join(os.path.expanduser(args.model_dir), "history_{}.tsv".format(datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")))

    def write_history(LOSS, ACC, outfile):
        # format the data: loss, acc
        data = [(float(loss), float(acc)) for score, acc in zip(LOSS, ACC)]
        # save the prediction result
        def write_tsv(data):
            with open(outfile, 'a') as out_file:
                tsv_writer = csv.writer(out_file, delimiter='\t')
                tsv_writer.writerows(data)
        write_tsv(data)

    logging.info('Training...')
    LOSS, ACC = [], []
    for ep in range(args.epochs):
        loss = 0.0
        accuracy = 0.0
        for cnt, (log_ids, log_mask, input_ids, targets) in enumerate(dataloader):
            if cnt > args.max_steps_per_epoch:
                break

            if args.enable_gpu:
                log_ids = log_ids.cuda(non_blocking=True)
                log_mask = log_mask.cuda(non_blocking=True)
                input_ids = input_ids.cuda(non_blocking=True)
                targets = targets.cuda(non_blocking=True)

            bz_loss, y_hat = model(input_ids, log_ids, log_mask, targets)
            # summary(model, [input_ids.shape, log_ids.shape, log_mask.shape, targets.shape], batch_size=16, device='cuda' if args.enable_gp else 'cpu')
            optimizer.zero_grad()
            bz_loss.backward()
            optimizer.step()
            if args.enable_lr_scheduler:
                lr_scheduler.step()

            loss += (bz_loss.data.float() - loss) / (cnt + 1)
            accuracy += (utils.acc(targets, y_hat) - accuracy) / (cnt + 1)
            if cnt % args.log_steps == 0:
                LOSS.append(loss.data)
                ACC.append(accuracy)
                logging.info(
                    '[{}] Ed: {}, train_loss: {:.5f}, acc: {:.5f}'.format(
                        hvd_rank, cnt * args.batch_size, loss.data,
                        accuracy))

            # save model minibatch
            logging.info('[{}] Ed: {} {} {}'.format(hvd_rank, cnt, args.save_steps, cnt % args.save_steps))
            if hvd_rank == 0 and cnt % args.save_steps == 0:
                ckpt_path = os.path.join(os.path.expanduser(args.model_dir), f'epoch-{ep+1}-{cnt}-{loss}-{accuracy}.pt')
                torch.save(
                    {
                        'model_state_dict': model.state_dict(),
                        'category_dict': category_dict,
                        'word_dict': word_dict,
                        'domain_dict': domain_dict,
                        'subcategory_dict': subcategory_dict
                    }, ckpt_path)
                logging.info(f"Model saved to {ckpt_path}")
                write_history(LOSS, ACC, outfile)
                LOSS, ACC = [], []
                logging.info(f"Training history saved to {outfile}")

        logging.info('epoch: {} loss: {} accuracy {}'.format(ep + 1, loss, accuracy))

        # save model last of epoch
        if hvd_rank == 0:
            ckpt_path = os.path.join(os.path.expanduser(args.model_dir), f'epoch-{ep+1}.pt')
            torch.save(
                {
                    'model_state_dict': model.state_dict(),
                    'category_dict': category_dict,
                    'word_dict': word_dict,
                    'domain_dict': domain_dict,
                    'subcategory_dict': subcategory_dict
                }, ckpt_path)
            logging.info(f"Model saved to {ckpt_path}")
        # save history last of epoch
        if len(LOSS) != 0:
            write_history(LOSS, ACC, outfile)
            LOSS, ACC = [], []
            logging.info(f"Training history saved to {outfile}")        

    dataloader.join()


def test(args):
    start_time = time.time()

    if args.enable_hvd:
        import horovod.torch as hvd

    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(
        args.enable_hvd, args.enable_gpu)

    if args.load_ckpt_name is not None:
        #TODO: choose ckpt_path
        ckpt_path = utils.get_checkpoint(os.path.expanduser(args.model_dir), args.load_ckpt_name)
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
    pretrain_lm_path = os.path.expanduser(args.pretrain_lm_path)  # or by name "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(pretrain_lm_path))
    config = AutoConfig.from_pretrained(os.path.expanduser(pretrain_lm_path), output_hidden_states=True)
    bert_model = AutoModel.from_pretrained(os.path.expanduser(pretrain_lm_path), config=config)
    model = ModelBert(args, bert_model, len(category_dict), len(domain_dict), len(subcategory_dict))

    if args.enable_gpu:
        model.cuda()

    model.load_state_dict(checkpoint['model_state_dict'])
    logging.info(f"Model loaded from {ckpt_path}")

    if args.enable_hvd:
        hvd.broadcast_parameters(model.state_dict(), root_rank=0)

    model.eval()
    torch.set_grad_enabled(False)

    news, news_index, category_dict, domain_dict, subcategory_dict = read_news_bert(
        os.path.join(os.path.expanduser(args.root_data_dir),
                    f'{args.dataset}/{args.test_dir}/news.tsv'), 
        args,
        tokenizer
    )

    news_title, news_title_type, news_title_attmask, \
    news_abstract, news_abstract_type, news_abstract_attmask, \
    news_body, news_body_type, news_body_attmask, \
    news_category, news_domain, news_subcategory = get_doc_input_bert(
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
                                batch_size=args.batch_size * 4,
                                num_workers=args.num_workers,
                                collate_fn=news_collate_fn)

    news_scoring = []
    with torch.no_grad():
        for input_ids in tqdm(news_dataloader):
            if args.enable_gpu:
                input_ids = input_ids.cuda()
            news_vec = model.news_encoder(input_ids)
            news_vec = news_vec.to(torch.device("cpu")).detach().numpy()
            news_scoring.extend(news_vec)

    news_scoring = np.array(news_scoring)

    logging.info("news scoring num: {}".format(news_scoring.shape[0]))

    dataloader = DataLoaderTest(
        news_index=news_index,
        news_scoring=news_scoring,
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

    AUC, MRR, nDCG5, nDCG10, SCORE = [], [], [], [], []
    count = 0
    outfile = os.path.join(os.path.expanduser(args.model_dir), "prediction_{}_{}.tsv".format(datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"), hvd_local_rank))

    def print_metrics(hvd_local_rank, cnt, x):
        logging.info("[{}] Ed: {}: {}".format(hvd_local_rank, cnt, \
            '\t'.join(["{:0.2f}".format(i * 100) for i in x])))

    def get_mean(arr):
        return [np.array(i).mean() for i in arr]

    def write_score(SCORE, outfile):
        # format the score: ImpressionID [Rank-of-News1,Rank-of-News2,...,Rank-of-NewsN]
        for score in tqdm(SCORE):
            argsort = np.argsort(-score[1])
            ranks = np.empty_like(argsort)
            ranks[argsort] = np.arange(len(score[1]))
            score[1] = (ranks + 1).tolist()

        # save the prediction result
        def write_tsv(score):
            with open(outfile, 'a') as out_file:
                tsv_writer = csv.writer(out_file, delimiter='\t')
                tsv_writer.writerows(score)
        write_tsv(SCORE)

    with torch.no_grad():
        for cnt, (impression_ids, log_vecs, log_mask, news_vecs, news_bias, labels) in enumerate(dataloader):
            # logging.info(f"start new batch {cnt}")
            count = cnt

            if args.enable_gpu:
                log_vecs = log_vecs.cuda(non_blocking=True)
                log_mask = log_mask.cuda(non_blocking=True)

            user_vecs = model.user_encoder(log_vecs, log_mask).to(torch.device("cpu")).detach().numpy()

            for impression_id, user_vec, news_vec, bias, label in zip(
                    impression_ids, user_vecs, news_vecs, news_bias, labels):

                if label.mean() == 0 or label.mean() == 1:
                    continue

                score = np.dot(
                    news_vec, user_vec
                )

                # label is -1 is for test set and prediction only
                if(np.all(label == -1)):
                    SCORE.append([impression_id, score])
                    continue

                auc = roc_auc_score(label, score)
                mrr = mrr_score(label, score)
                ndcg5 = ndcg_score(label, score, k=5)
                ndcg10 = ndcg_score(label, score, k=10)

                AUC.append(auc)
                MRR.append(mrr)
                nDCG5.append(ndcg5)
                nDCG10.append(ndcg10)

            if cnt % args.log_steps == 0:
                # print_metrics(hvd_rank, cnt * args.batch_size, [1.0])
                print_metrics(hvd_rank, cnt * args.batch_size, get_mean([AUC, MRR, nDCG5,  nDCG10]))

            if cnt % args.save_steps == 0:
                if len(SCORE) > 0:
                    logging.info("[{}] Ed: {}: saving {} lines to {}".format(hvd_local_rank, cnt, len(SCORE), outfile))
                    write_score(SCORE, outfile)
                    SCORE = []

    # stop scoring
    logging.info("Stop scoring")
    dataloader.join()

    # save the last batch of scores
    if len(SCORE) > 0:
        logging.info("[{}] Ed: {}: saving {} lines to {}".format(hvd_local_rank, cnt, len(SCORE), outfile))
        write_score(SCORE, outfile)
        SCORE = []

    # print and save metrics
    logging.info("Print final metrics")
    print_metrics(hvd_rank, count * args.batch_size, get_mean([AUC, MRR, nDCG5,  nDCG10]))

    logging.info(f"Time taken: {time.time() - start_time}")


if __name__ == "__main__":
    utils.setuplogger()
    args = parse_args()
    Path(os.path.expanduser(args.model_dir)).mkdir(parents=True, exist_ok=True)
    if 'train' in args.mode:
        train(args)
    if 'test' in args.mode:
        test(args)
