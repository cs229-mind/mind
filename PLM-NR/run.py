import faulthandler
import signal
faulthandler.register(signal.SIGUSR1.value)
import pickle
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
from preprocess import read_news, read_user, read_news_lm, get_doc_input, get_doc_input_lm
from model_bert import Model
from parameters import parse_args
from transformers import AutoTokenizer, AutoModel, AutoConfig, get_scheduler, AdamW
from torch.utils.tensorboard import SummaryWriter



def train(args):

    assert args.enable_hvd
    if args.enable_hvd:
        import horovod.torch as hvd

    if args.load_ckpt_train is not None:
        ckpt_path = utils.get_checkpoint(os.path.join(os.path.expanduser(args.model_dir), args.load_ckpt_train))
    else:
        ckpt_path = utils.latest_checkpoint(os.path.expanduser(args.model_dir))

    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(
        args.enable_hvd, args.enable_gpu)

    pretrain_lm_path = os.path.expanduser(args.pretrain_lm_path)  # or by name "bert-base-uncased"
    tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(pretrain_lm_path))
    config = AutoConfig.from_pretrained(os.path.expanduser(pretrain_lm_path), output_hidden_states=True)
    language_model = AutoModel.from_pretrained(os.path.expanduser(pretrain_lm_path), config=config)

    # auto adjust hyper parameter by pre-trained model config
    args.num_layers = config.num_hidden_layers if args.num_layers is None else args.num_layers
    args.word_embedding_dim = config.hidden_size if args.word_embedding_dim is None else args.word_embedding_dim
    finetuneset = {
    f'encoder.layer.{args.num_layers-2}.attention.self.query.weight',
    f'encoder.layer.{args.num_layers-2}.attention.self.query.bias',
    f'encoder.layer.{args.num_layers-2}.attention.self.key.weight',
    f'encoder.layer.{args.num_layers-2}.attention.self.key.bias',
    f'encoder.layer.{args.num_layers-2}.attention.self.value.weight',
    f'encoder.layer.{args.num_layers-2}.attention.self.value.bias',
    f'encoder.layer.{args.num_layers-2}.attention.output.dense.weight',
    f'encoder.layer.{args.num_layers-2}.attention.output.dense.bias',
    f'encoder.layer.{args.num_layers-2}.attention.output.LayerNorm.weight',
    f'encoder.layer.{args.num_layers-2}.attention.output.LayerNorm.bias',
    f'encoder.layer.{args.num_layers-2}.intermediate.dense.weight',
    f'encoder.layer.{args.num_layers-2}.intermediate.dense.bias',
    f'encoder.layer.{args.num_layers-2}.output.dense.weight',
    f'encoder.layer.{args.num_layers-2}.output.dense.bias',
    f'encoder.layer.{args.num_layers-2}.output.LayerNorm.weight',
    f'encoder.layer.{args.num_layers-2}.output.LayerNorm.bias',
    f'encoder.layer.{args.num_layers-1}.attention.self.query.weight',
    f'encoder.layer.{args.num_layers-1}.attention.self.query.bias',
    f'encoder.layer.{args.num_layers-1}.attention.self.key.weight',
    f'encoder.layer.{args.num_layers-1}.attention.self.key.bias',
    f'encoder.layer.{args.num_layers-1}.attention.self.value.weight',
    f'encoder.layer.{args.num_layers-1}.attention.self.value.bias',
    f'encoder.layer.{args.num_layers-1}.attention.output.dense.weight',
    f'encoder.layer.{args.num_layers-1}.attention.output.dense.bias',
    f'encoder.layer.{args.num_layers-1}.attention.output.LayerNorm.weight',
    f'encoder.layer.{args.num_layers-1}.attention.output.LayerNorm.bias',
    f'encoder.layer.{args.num_layers-1}.intermediate.dense.weight',
    f'encoder.layer.{args.num_layers-1}.intermediate.dense.bias',
    f'encoder.layer.{args.num_layers-1}.output.dense.weight',
    f'encoder.layer.{args.num_layers-1}.output.dense.bias',
    f'encoder.layer.{args.num_layers-1}.output.LayerNorm.weight',
    f'encoder.layer.{args.num_layers-1}.output.LayerNorm.bias',
    'pooler.dense.weight',
    'pooler.dense.bias',
    'rel_pos_bias.weight',
    'classifier.weight',
    'classifier.bias'}

    #language_model.load_state_dict(torch.load('../bert_encoder_part.pkl'))
    # freeze parameters
    for name,param in language_model.named_parameters():
        if args.fineune_options == 0:
            param.requires_grad = False
        elif args.fineune_options == -2:
            if name not in finetuneset:
                param.requires_grad = False
        else:  #args.fineune_options == -12:
            continue

    # save 1~2 minutes time, manually delete the cache file if cache is outdated
    user_cache_path = os.path.join(os.path.expanduser(args.root_data_dir), f'{args.dataset}/{args.train_dir}/user_cache.pkl')
    if os.path.exists(user_cache_path):
        user_dict = pickle.load(open(user_cache_path, "rb"))
    else:
        user_dict = read_user(
            os.path.join(os.path.expanduser(args.root_data_dir),
                        f'{args.dataset}/{args.train_dir}/'),
            args.filename_pat,
            args
        )
        pickle.dump(user_dict, open(user_cache_path, "wb"))

    # save 1~2 minutes time, manually delete the cache file if cache is outdated
    news_cache_path = os.path.join(os.path.expanduser(args.root_data_dir), f'{args.dataset}/{args.train_dir}/news_cache.pkl')
    if os.path.exists(news_cache_path):
        news, news_index, category_dict, domain_dict, subcategory_dict = pickle.load(open(news_cache_path, "rb"))
    else:
        news, news_index, category_dict, domain_dict, subcategory_dict = read_news_lm(
            os.path.join(os.path.expanduser(args.root_data_dir),
                        f'{args.dataset}/{args.train_dir}/news.tsv'), 
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

    model = Model(args, language_model, len(user_dict), len(category_dict), len(domain_dict), len(subcategory_dict))
    word_dict = None

    if args.enable_gpu:
        model = model.cuda()

    if args.enable_incremental:
        if ckpt_path is None:
            logging.warning('No ckpt found! Warm start is skipped!!!')
        else:
            # this is not an error, set both to cpu to avoid conflict in gpu process for now
            if args.enable_gpu:
                checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
            else:
                checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
            user_dict = checkpoint['user_dict']
            category_dict = checkpoint['category_dict']
            subcategory_dict = checkpoint['subcategory_dict']
            word_dict = checkpoint['word_dict']
            domain_dict = checkpoint['domain_dict']
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
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
        user_dict=user_dict,
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

    LOSS, ACC, VERBOSE = [], [], []
    outfile = os.path.join(os.path.expanduser(args.model_dir), "history_{}.tsv".format(datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")))
    def write_history(LOSS, ACC, VERBOSE, outfile):
        # format the data: loss, acc
        data = [(round(float(l), 5), round(float(a), 5), v) for l, a, v in zip(LOSS, ACC, VERBOSE)]
        # save the prediction result
        def write_tsv(data):
            with open(outfile, 'a') as out_file:
                tsv_writer = csv.writer(out_file, delimiter='\t')
                tsv_writer.writerows(data)
        write_tsv(data)

    with torch.autograd.detect_anomaly() if args.enable_detect_anomaly else utils.dummy_context_mgr():
        logging.info('Training...')
        # track the training progress
        if hvd_rank == 0:
            writer = SummaryWriter()
        for ep in range(args.epochs):
            loss = 0.0
            accuracy = 0.0
            for cnt, (user_ids, log_ids, log_mask, input_ids, targets) in enumerate(dataloader):
                if cnt > args.max_steps_per_epoch:
                    break
                global_step = args.max_steps_per_epoch * ep + cnt

                if args.enable_gpu:
                    log_ids = log_ids.cuda(non_blocking=True)
                    log_mask = log_mask.cuda(non_blocking=True)
                    input_ids = input_ids.cuda(non_blocking=True)
                    user_ids = user_ids.cuda(non_blocking=True)
                    targets = targets.cuda(non_blocking=True)

                bz_loss, y_hat = model(input_ids, user_ids, log_ids, log_mask, targets)
                if global_step == 0 and hvd_rank == 0:
                    writer.add_graph(model, input_to_model=[input_ids, user_ids, log_ids, log_mask, targets], verbose=False)
                optimizer.zero_grad()
                bz_loss.backward()

                if args.clip_grad_norm is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad_norm)

                optimizer.step()
                if args.enable_lr_scheduler:
                    lr_scheduler.step()

                loss += (bz_loss.data.float() - loss) / (cnt + 1)
                accuracy += (utils.acc(targets, y_hat) - accuracy) / (cnt + 1)
                if hvd_rank == 0:
                    writer.add_scalar("Loss", loss.data, global_step)
                    writer.add_scalar("Accuracy", accuracy, global_step)
                    writer.add_scalars('Summary', {'Loss': loss.data, 'Accuracy': accuracy}, global_step)
                if cnt % args.log_steps == 0:
                    LOSS.append(loss.detach().cpu().numpy())
                    ACC.append(accuracy.detach().cpu().numpy())
                    VERBOSE.append('[{}] Ed: {}-{}-{}'.format(hvd_rank, ep, cnt * args.batch_size, datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S")))
                    logging.info(
                        '[{}] Ed: {} {}, train_loss: {:.5f}, acc: {:.5f}'.format(
                            hvd_rank, ep, cnt * args.batch_size, loss.data,
                            accuracy))

                # save model for every num of save steps
                logging.info('[{}] Ed: {} {} {}'.format(hvd_rank, cnt, args.save_steps, cnt % args.save_steps))
                def save_model(LOSS, ACC, VERBOSE, eva=True):
                    # save model
                    ckpt_path = os.path.join(os.path.expanduser(args.model_dir), f'epoch-{ep+1}-{cnt}-{loss:.5f}-{accuracy:.5f}.pt')
                    torch.save(
                        {
                            'model_state_dict': model.state_dict(),
                            'user_dict': user_dict,                        
                            'category_dict': category_dict,
                            'word_dict': word_dict,
                            'domain_dict': domain_dict,
                            'subcategory_dict': subcategory_dict
                        }, ckpt_path)
                    logging.info(f"Model saved to {ckpt_path}")

                    # save history
                    if len(LOSS) != 0:
                        write_history(LOSS, ACC, VERBOSE, outfile)
                        LOSS, ACC, VERBOSE = [], [], []
                        logging.info(f"Training history saved to {outfile}")

                    # evaluate the model for each save
                    metrics = None
                    if eva and args.enable_evaluation:
                        prev_test_dir = args.test_dir
                        args.test_dir = 'dev'
                        logging.info(f"Evaluation on data in dir {args.test_dir} started")
                        model.eval()
                        torch.set_grad_enabled(False)
                        metrics = test(args, model, user_dict, category_dict, word_dict, domain_dict, subcategory_dict, tokenizer, ckpt_path)
                        utils.add_metrics(writer, metrics, global_step)
                        utils.add_weight_histograms(model, writer, args, global_step)
                        model.train()
                        torch.set_grad_enabled(True)
                        args.test_dir = prev_test_dir
                        logging.info(f"Evaluation on data in dir {args.test_dir} finished with final metrics: {metrics}")

                if hvd_rank == 0 and cnt % args.save_steps == 0 and cnt != 0:
                    save_model(LOSS, ACC, VERBOSE)

            logging.info('epoch: {} loss: {:.5f} accuracy {:.5f}'.format(ep + 1, loss, accuracy))

            # # save model last of epoch
            # if hvd_rank == 0:
            #     save_model(LOSS, ACC, VERBOSE)

        if hvd_rank == 0:
            list_element_dict = {'category': category_dict, 'domain': domain_dict, 'subcategory': subcategory_dict, 'user_id': user_dict}
            utils.add_embedding(model, writer, list_element_dict, args, global_step)

    dataloader.join()


def test(args, model=None, user_dict=None, category_dict=None, word_dict=None, domain_dict=None, subcategory_dict=None, tokenizer=None, ckpt_path=None):
    start_time = time.time()

    if args.enable_hvd:
        import horovod.torch as hvd

    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(
        args.enable_hvd, args.enable_gpu)

    if model is None:
        if args.load_ckpt_test is not None:
            ckpt_path = utils.get_checkpoint(os.path.expanduser(args.model_dir), args.load_ckpt_test)
        else:
            ckpt_path = utils.latest_checkpoint(os.path.expanduser(args.model_dir))

        assert ckpt_path is not None, 'No ckpt found'
        # this is not an error, set both to cpu to avoid conflict in gpu process for now    
        if args.enable_gpu:
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))
        else:
            checkpoint = torch.load(ckpt_path, map_location=torch.device('cpu'))

        subcategory_dict = checkpoint['subcategory_dict'] if subcategory_dict is None else subcategory_dict
        category_dict = checkpoint['category_dict'] if category_dict is None else category_dict
        word_dict = checkpoint['word_dict'] if word_dict is None else word_dict
        domain_dict = checkpoint['domain_dict'] if domain_dict is None else domain_dict
        user_dict = checkpoint['user_dict'] if user_dict is None and 'user_dict' in checkpoint else user_dict if user_dict is not None else {}
        pretrain_lm_path = os.path.expanduser(args.pretrain_lm_path)  # or by name "bert-base-uncased"
        tokenizer = AutoTokenizer.from_pretrained(os.path.expanduser(pretrain_lm_path)) if tokenizer is None else tokenizer

        config = AutoConfig.from_pretrained(os.path.expanduser(pretrain_lm_path), output_hidden_states=True)
        language_model = AutoModel.from_pretrained(os.path.expanduser(pretrain_lm_path), config=config)

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

    AUC, MRR, nDCG5, nDCG10, SCORE = [], [], [], [], []
    count = 0

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
        return mean_metrics

    def get_mean(arr):
        return [np.array(i).mean() for i in arr]

    outfile_prediction = os.path.join(os.path.expanduser(args.model_dir), "prediction_{}_{}_{}.tsv".format(ckpt_path.rsplit('/',1)[1].rsplit('.',1)[0], datetime.datetime.utcnow().strftime("%Y%m%d%H%M%S"), hvd_local_rank))
    def write_score(SCORE):
        # format the score: ImpressionID [Rank-of-News1,Rank-of-News2,...,Rank-of-NewsN]
        for score in tqdm(SCORE):
            argsort = np.argsort(-score[1])
            ranks = np.empty_like(argsort)
            ranks[argsort] = np.arange(len(score[1]))
            score[1] = (ranks + 1).tolist()
        # save the prediction result
        def write_tsv(score):
            with open(outfile_prediction, 'a') as out_file:
                tsv_writer = csv.writer(out_file, delimiter='\t')
                tsv_writer.writerows(score)
            logging.info(f"Saved scoring to {outfile_prediction}")
        write_tsv(SCORE)

    with torch.no_grad():
        for cnt, (impression_ids, user_ids, log_vecs, log_mask, news_vecs, news_bias, labels) in enumerate(dataloader):
            # logging.info(f"start new batch {cnt}")
            count = cnt

            if args.enable_gpu:
                user_ids = user_ids.cuda(non_blocking=True)
                log_vecs = log_vecs.cuda(non_blocking=True)
                log_mask = log_mask.cuda(non_blocking=True)

            user_vecs = model.user_encoder(user_ids, log_vecs, log_mask)

            for impression_id, user_vec, news_vec, bias, label in zip(
                    impression_ids, user_vecs, news_vecs, news_bias, labels):

                if label.mean() == 0 or label.mean() == 1:
                    continue

                user_vec = user_vec.unsqueeze(0)
                news_vec = news_vec.unsqueeze(0)
                score = model.scoring(news_vec, user_vec).to(torch.device("cpu")).detach().numpy().squeeze(0)

                # label is -1 is for test set and prediction only
                if(np.all(label == -1)):
                    SCORE.append([impression_id, score])
                    continue

                if np.isnan(np.min(score)):
                    logging.error(f"Scores {score} have NAN, correct your model!")
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
                print_metrics(hvd_rank, cnt * args.batch_size, get_mean([AUC, MRR, nDCG5,  nDCG10]), save=False)

            if cnt % args.save_steps == 0:
                if len(SCORE) > 0:
                    logging.info("[{}] Ed: {}: saving {} lines to {}".format(hvd_local_rank, cnt, len(SCORE), outfile_prediction))
                    write_score(SCORE)
                    SCORE = []

    # stop scoring
    logging.info("Stop scoring")
    dataloader.join()

    # save the last batch of scores
    if len(SCORE) > 0:
        logging.info("[{}] Ed: {}: saving {} lines to {}".format(hvd_local_rank, cnt, len(SCORE), outfile_prediction))
        write_score(SCORE)
        SCORE = []

    # print and save metrics
    final_metrics = [0.0, 0.0, 0.0, 0.0]
    if len(AUC) > 0:
        logging.info("Print final metrics")
        final_metrics = print_metrics(hvd_rank, count * args.batch_size, get_mean([AUC, MRR, nDCG5,  nDCG10]))

    logging.info(f"Time taken: {time.time() - start_time}")

    return final_metrics


if __name__ == "__main__":
    utils.setuplogger()
    args = parse_args()
    Path(os.path.expanduser(args.model_dir)).mkdir(parents=True, exist_ok=True)
    if 'train' in args.mode:
        train(args)
    if 'test' in args.mode:
        test(args)
