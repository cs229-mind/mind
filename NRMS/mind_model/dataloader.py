import sys
import traceback
import logging
import time
import random
from queue import Queue
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import torch
from torch.utils.data import IterableDataset
from .streaming import StreamSampler, StreamSamplerTest
from . import utils
from .preprocess import read_news, read_news_bert, get_doc_input, get_doc_input_bert
from torch.utils.data import Dataset, DataLoader
import os
import tqdm


def news_sample(news, ratio):
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)


class DataLoaderTrain(IterableDataset):
    def __init__(self,
                 data_dir,
                 filename_pat,
                 args,
                 worker_size,
                 worker_rank,
                 cuda_device_idx,
                 news_index,
                 user_dict,
                 news_combined,
                 word_dict,
                 enable_prefetch=True,
                 enable_shuffle=False,
                 enable_gpu=True):
        self.data_dir = data_dir
        self.filename_pat = filename_pat

        self.npratio = args.npratio
        self.user_log_length = args.user_log_length
        self.user_attributes = args.user_attributes
        self.batch_size = args.batch_size

        self.worker_rank = worker_rank
        self.worker_size = worker_size
        self.cuda_device_idx = cuda_device_idx
        # data loader only cares about the config after tokenization.
        self.sampler = None

        self.shuffle_buffer_size = args.shuffle_buffer_size

        self.enable_prefetch = enable_prefetch
        self.enable_shuffle = enable_shuffle
        self.enable_gpu = enable_gpu
        self.epoch = -1

        self.news_combined = news_combined
        self.news_index = news_index
        self.user_dict = user_dict
        self.word_dict = word_dict

    def start(self):
        self.epoch += 1
        self.sampler = StreamSampler(
            data_dir=self.data_dir,
            filename_pat=self.filename_pat,
            batch_size=self.batch_size,
            worker_rank=self.worker_rank,
            worker_size=self.worker_size,
            enable_shuffle=self.enable_shuffle,
            shuffle_buffer_size=self.shuffle_buffer_size,
            shuffle_seed=self.epoch,  # epoch id as shuffle random seed
        )
        self.sampler.__iter__()

    def start_async(self):
        self.aval_count = 0
        self.stopped = False
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

    def trans_to_nindex(self, nids):
        return [self.news_index[i] if i in self.news_index else 0 for i in nids]

    def trans_to_uindex(self, uids):
        return [self.user_dict[i] if i in self.user_dict else 0 for i in uids]

    def pad_to_fix_len(self, x, fix_length, padding_front=True, padding_value=0):
        if padding_front:
            pad_x = [padding_value] * (fix_length-len(x)) + x[-fix_length:]
            mask = [0] * (fix_length-len(x)) + [1] * min(fix_length, len(x))
        else:
            pad_x = x[:fix_length] + [padding_value]*(fix_length-len(x))
            mask = [1] * min(fix_length, len(x)) + [0] * (len(x) - fix_length)
        return pad_x, mask

    def _produce(self):
        # need to reset cuda device in produce thread.
        if self.enable_gpu:
            torch.cuda.set_device(self.cuda_device_idx)
        try:
            self.epoch += 1
            self.sampler = StreamSampler(
                data_dir=self.data_dir,
                filename_pat=self.filename_pat,
                batch_size=self.batch_size,
                worker_rank=self.worker_rank,
                worker_size=self.worker_size,
                enable_shuffle=self.enable_shuffle,
                shuffle_seed=self.epoch,  # epoch id as shuffle random seed
            )
            # t0 = time.time()
            for batch in self.sampler:
                if self.stopped:
                    # put a None object to communicate the end of queue
                    self.outputs.put(None)
                    break
                # print(f"!!!!!!!!!!!! put start {self.aval_count}")
                context = self._process(batch)
                self.outputs.put(context)
                self.aval_count += 1
                # print(f"!!!!!!!!!!!! put end {self.aval_count}")
                # logging.info(f"_produce cost:{time.time()-t0}")
                # t0 = time.time()
            # put a None object to communicate the end of queue
            self.outputs.put(None)
        except:
            # put a None object to communicate the end of queue
            self.outputs.put(None)
            traceback.print_exc(file=sys.stdout)
            self.pool.shutdown(wait=False)
            raise

    def parse_sent(self, sent, fix_length):
        sent = [self.word_dict[w] if w in self.word_dict else 0 for w in utils.word_tokenize(sent)]
        sent, _ = self.pad_to_fix_len(sent, fix_length, padding_front=False)
        return sent

    def parse_sents(self, sents, max_sents_num, max_sent_length, padding_front=True):
        sents, sents_mask = self.pad_to_fix_len(sents, max_sents_num, padding_value='')
        sents = [self.parse_sent(s, max_sent_length) for s in sents]
        sents = np.stack(sents, axis=0)
        sents_mask = np.array(sents_mask)
        return sents, sents_mask

    def _process(self, batch):
        batch_size = len(batch)
        #print(batch)
        batch_poss, batch = batch
        batch_poss = [x.numpy().decode(encoding="utf-8") for x in batch_poss]
        batch = [x.numpy().decode(encoding="utf-8").split("\t") for x in batch]
        label = 0
        user_id_batch, user_feature_batch, log_mask_batch, news_feature_batch, label_batch = [], [], [], [], []

        for poss, line in zip(batch_poss, batch):
            user_id = line[1]
            #user_id = self.trans_to_uindex([user_id])

            click_docs = line[3].split()

            click_docs, log_mask = self.pad_to_fix_len(self.trans_to_nindex(click_docs),
                                             self.user_log_length)

            user_feature = self.news_combined[click_docs]

            sess_news = [i.split('-') for i in line[4].split()]
            sess_neg = [i[0] for i in sess_news if i[-1] == '0']

            poss = self.trans_to_nindex([poss])
            sess_neg = self.trans_to_nindex(sess_neg)

            if len(sess_neg) > 0:
                neg_index = news_sample(list(range(len(sess_neg))),
                                        self.npratio)
                sam_negs = [sess_neg[i] for i in neg_index]
            else:
                sam_negs = [0] * self.npratio
            sample_news = poss + sam_negs

            news_feature = self.news_combined[sample_news]
            user_id_batch.append(user_id)
            user_feature_batch.append(user_feature)
            log_mask_batch.append(log_mask)
            news_feature_batch.append(news_feature)
            label_batch.append(label)

        if self.enable_gpu:
            user_id_batch = torch.LongTensor(user_id_batch).cuda()
            user_feature_batch = torch.LongTensor(user_feature_batch).cuda()
            log_mask_batch = torch.FloatTensor(log_mask_batch).cuda()
            news_feature_batch = torch.LongTensor(news_feature_batch).cuda()
            label_batch = torch.LongTensor(label_batch).cuda()
        else:
            user_id_batch = torch.LongTensor(user_id_batch)
            user_feature_batch = torch.LongTensor(user_feature_batch)
            log_mask_batch = torch.FloatTensor(log_mask_batch)
            news_feature_batch = torch.LongTensor(news_feature_batch)
            label_batch = torch.LongTensor(label_batch)

        return user_id_batch, user_feature_batch, log_mask_batch, news_feature_batch, label_batch

    def __iter__(self):
        """Implement IterableDataset method to provide data iterator."""
        logging.info("DataLoader __iter__()")
        if self.enable_prefetch:
            self.join()
            self.start_async()
        else:
            self.start()
        return self

    def __next__(self):
        if self.sampler and self.sampler.reach_end() and self.aval_count == 0:
            raise StopIteration
        if self.enable_prefetch:
            # print(f"!!!!!!!!!!!! get start {self.aval_count}")
            next_batch = self.outputs.get()
            # print(f"!!!!!!!!!!!! get end {self.aval_count}")
            # print(f"!!!!!!!!!!!! get {next_batch}")
            self.outputs.task_done()
            self.aval_count -= 1
            # a None object from producer means the end of queue
            if next_batch is None:
                raise StopIteration
        else:
            next_batch = self._process(self.sampler.__next__())
        return next_batch

    def join(self):
        self.stopped = True
        if self.sampler:
            if self.enable_prefetch:
                while self.outputs.qsize() > 0:
                    self.outputs.get()
                    self.outputs.task_done()
                self.outputs.join()
                self.pool.shutdown(wait=True)
                logging.info("shut down pool.")
            self.sampler = None


class DataLoaderTest(DataLoaderTrain):
    def __init__(self,
                 data_dir,
                 filename_pat,
                 args,
                 worker_size,
                 worker_rank,
                 cuda_device_idx,
                 news_index,
                 user_dict,
                 news_scoring,
                 word_dict,
                 news_bias_scoring=None,
                 enable_prefetch=True,
                 enable_shuffle=False,
                 enable_gpu=True):
        self.data_dir = data_dir
        self.filename_pat = filename_pat

        self.npratio = args.npratio
        self.user_log_length = args.user_log_length
        self.batch_size = args.batch_size

        self.worker_rank = worker_rank
        self.worker_size = worker_size
        self.cuda_device_idx = cuda_device_idx
        # data loader only cares about the config after tokenization.
        self.sampler = None

        self.enable_prefetch = enable_prefetch
        self.enable_shuffle = enable_shuffle
        self.enable_gpu = enable_gpu
        self.epoch = -1

        self.news_scoring = news_scoring
        self.news_bias_scoring = news_bias_scoring
        self.news_index = news_index
        self.user_dict = user_dict
        self.word_dict = word_dict

    def start_async(self):
        self.aval_count = 0
        self.stopped = False
        self.outputs = Queue(10)
        self.pool = ThreadPoolExecutor()
        self.pool.submit(self._produce)

    def start(self):
        self.epoch += 1
        self.sampler = StreamSamplerTest(
            data_dir=self.data_dir,
            filename_pat=self.filename_pat,
            batch_size=self.batch_size,
            worker_rank=self.worker_rank,
            worker_size=self.worker_size,
            enable_shuffle=self.enable_shuffle,
            shuffle_seed=self.epoch,  # epoch id as shuffle random seed
        )
        self.sampler.__iter__()

    def _produce(self):
        # need to reset cuda device in produce thread.
        if self.enable_gpu:
            torch.cuda.set_device(self.cuda_device_idx)
        try:
            self.epoch += 1
            self.sampler = StreamSamplerTest(
                data_dir=self.data_dir,
                filename_pat=self.filename_pat,
                batch_size=self.batch_size,
                worker_rank=self.worker_rank,
                worker_size=self.worker_size,
                enable_shuffle=self.enable_shuffle,
                shuffle_seed=self.epoch,  # epoch id as shuffle random seed
            )
            # t0 = time.time()
            for batch in self.sampler:
                if self.stopped:
                    # put a None object to communicate the end of queue
                    self.outputs.put(None)
                    break
                # print(f"!!!!!!!!!!!! put start {self.aval_count}")
                context = self._process(batch)
                self.outputs.put(context)
                self.aval_count += 1
                # print(f"!!!!!!!!!!!! put end {self.aval_count}")
                # logging.info(f"_produce cost:{time.time()-t0}")
                # t0 = time.time()
            # put a None object to communicate the end of queue
            self.outputs.put(None)
        except:
            # put a None object to communicate the end of queue
            self.outputs.put(None)
            self.aval_count += 1
            traceback.print_exc(file=sys.stdout)
            self.pool.shutdown(wait=False)
            raise

    def _process(self, batch):
        batch_size = len(batch)
        batch = [x.numpy().decode(encoding="utf-8").split("\t") for x in batch]

        impression_id_batch, user_id_batch, user_feature_batch, log_mask_batch, news_feature_batch, news_bias_batch, label_batch = [], [], [], [], [], [], []

        for line in batch:
            impression_id = line[0]

            user_id = line[1]
            user_id = self.trans_to_uindex([user_id])

            click_docs = line[3].split()

            click_docs, log_mask  = self.pad_to_fix_len(self.trans_to_nindex(click_docs),
                                             self.user_log_length)
            user_feature = self.news_scoring[click_docs]

            sample_news = self.trans_to_nindex([i.split('-')[0] for i in line[4].split()])
            # validation(dev) set has label '<news_id>_(1 for click and 0 for non-click)'
            # test set has no label '<news_id> without _(1 for click and 0 for non-click)', put a dummy label -1
            labels = [int(i.split('-')[1]) if len(i.split('-')) > 1 else -1 for i in line[4].split()]

            news_feature = self.news_scoring[sample_news]
            if self.news_bias_scoring is not None:
                news_bias = self.news_bias_scoring[sample_news]
            else:
                news_bias = [0] * len(sample_news)

            impression_id_batch.append(impression_id)
            user_id_batch.append(user_id)
            user_feature_batch.append(user_feature)
            log_mask_batch.append(log_mask)
            news_feature_batch.append(news_feature)
            news_bias_batch.append(news_bias)
            label_batch.append(np.array(labels))

        if self.enable_gpu:
            user_id_batch = torch.LongTensor(user_id_batch).cuda()
            user_feature_batch = torch.FloatTensor(user_feature_batch).cuda()
            log_mask_batch = torch.FloatTensor(log_mask_batch).cuda()
        else:
            user_id_batch = torch.LongTensor(user_id_batch)
            user_feature_batch = torch.FloatTensor(user_feature_batch)
            log_mask_batch = torch.FloatTensor(log_mask_batch)

        return impression_id_batch, user_id_batch, user_feature_batch, log_mask_batch, news_feature_batch, news_bias_batch, label_batch





def test_iterator(model, valid_dir, args, tokenizer):
    valid_news_file = os.path.join(valid_dir, r'news.tsv')
    news, news_index, category_dict, domain_dict, subcategory_dict = read_news_bert(
        valid_news_file,
        args,
        tokenizer
    )

    news_title, news_title_type, news_title_attmask, \
    news_abstract, news_abstract_type, news_abstract_attmask, \
    news_body, news_body_type, news_body_attmask, \
    news_category, news_domain, news_subcategory = get_doc_input_bert(
        news, news_index, category_dict, domain_dict, subcategory_dict, args)

    word_dict, user_dict = None, None
    # args.enable_hvd = False
    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(
        args.enable_hvd, args.enable_gpu)
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

    dev_dataloader = DataLoaderTest(
        news_index=news_index,
        news_scoring=news_scoring,
        user_dict=user_dict,
        word_dict=word_dict,
        news_bias_scoring=None,
        data_dir=valid_dir,
        filename_pat=args.filename_pat,
        args=args,
        worker_size=hvd_size,
        worker_rank=hvd_rank,
        cuda_device_idx=hvd_local_rank,
        enable_prefetch=True,
        enable_shuffle=False,
        enable_gpu=args.enable_gpu,
    )
    return {"iterator": dev_dataloader,
            "category_dict": category_dict,
            "domain_dict": domain_dict,
            "subcategory_dict": subcategory_dict,
            }


def train_iterator(train_dir, args, tokenizer):
    train_news_file = os.path.join(train_dir, r'news.tsv')

    news, news_index, category_dict, domain_dict, subcategory_dict = read_news_bert(
        train_news_file,
        args,
        tokenizer
    )

    news_title, news_title_type, news_title_attmask, \
    news_abstract, news_abstract_type, news_abstract_attmask, \
    news_body, news_body_type, news_body_attmask, \
    news_category, news_domain, news_subcategory = get_doc_input_bert(
        news, news_index, category_dict, domain_dict, subcategory_dict, args)

    word_dict, user_dict = None, None
    # args.enable_hvd = False
    hvd_size, hvd_rank, hvd_local_rank = utils.init_hvd_cuda(
        args.enable_hvd, args.enable_gpu)
    news_combined = np.concatenate([
        x for x in
        [news_title, news_title_type, news_title_attmask, \
         news_abstract, news_abstract_type, news_abstract_attmask, \
         news_body, news_body_type, news_body_attmask, \
         news_category, news_domain, news_subcategory]
        if x is not None], axis=1)
    dataloader = DataLoaderTrain(
        news_index=news_index,
        news_combined=news_combined,
        user_dict=user_dict,
        word_dict=word_dict,
        data_dir=train_dir,
        filename_pat=args.filename_pat,
        args=args,
        worker_size=hvd_size,
        worker_rank=hvd_rank,
        cuda_device_idx=hvd_local_rank,
        enable_prefetch=True,
        enable_shuffle=True,
        enable_gpu=args.enable_gpu,
    )
    return {"iterator": dataloader,
            "category_dict": category_dict,
            "domain_dict": domain_dict,
            "subcategory_dict": subcategory_dict,
            }
