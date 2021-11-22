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
from streaming import StreamSampler, StreamSamplerTest
import utils
from dataloader import DataLoaderTrain


def news_sample(news, ratio):
    if ratio > len(news):
        return news + [0] * (ratio - len(news))
    else:
        return random.sample(news, ratio)

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

    def start(self):
        self.epoch += 1
        self.sampler = StreamSamplerTest(
            data_dir=self.data_dir,
            filename_pat=self.filename_pat,
            batch_size=self.batch_size * 8,
            worker_rank=self.worker_rank,
            worker_size=self.worker_size,
            enable_shuffle=self.enable_shuffle,
            shuffle_seed=self.epoch,  # epoch id as shuffle random seed
        )
        self.sampler.__iter__()

    def start_async(self):
        self.aval_count = 0
        self.stopped = False
        self.outputs = Queue(100)
        self.pool = ThreadPoolExecutor(1)
        self.pool.submit(self._produce)

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
                # context = self._process(batch)
                context = utils.parallel(self._process, batch)
                # self.outputs.put(context)
                # self.aval_count += 1
                # print(f"!!!!!!!!!!!! put end {self.aval_count}")
                # logging.info(f"_produce cost:{time.time()-t0}")
                # t0 = time.time()
            # put a None object to communicate the end of queue
            self.outputs.put(None)
        except:
            # put a None object for producer to communicate the end of queue 
            self.outputs.put(None)            
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

            # if self.enable_gpu:
            #     user_feature_batch = torch.FloatTensor(user_feature_batch).cuda()
            #     log_mask_batch = torch.FloatTensor(log_mask_batch).cuda()
            # else:
            #     user_feature_batch = torch.FloatTensor(user_feature_batch)
            #     log_mask_batch = torch.FloatTensor(log_mask_batch)

        batch = [(impression_id, user_id, user_feature, log_mask, news_feature, news_bias, label) \
                for impression_id, user_id, user_feature, log_mask, news_feature, news_bias, label in zip(
                    impression_id_batch, user_id_batch, user_feature_batch, log_mask_batch, news_feature_batch, news_bias_batch, label_batch)]

        # print(f"!!!!!!!!!!!! put start {self.aval_count}")
        self.outputs.put(batch)
        self.aval_count += 1
        # print(f"!!!!!!!!!!!! put end {self.aval_count}")

        return batch
