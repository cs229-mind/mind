import logging
import os
import sys
import torch
import numpy as np
import argparse
import re
import multiprocessing
from spacy.util import minibatch
from joblib import Parallel, delayed
from functools import partial
from ltr.metrics import ndcg, dcg, mrr
# TODO: this is a temp workaround to make tensorboard work for add_embeddings
import tensorboard as tb
import tensorflow as tf
tf.io.gfile = tb.compat.tensorflow_stub.io.gfile


def word_tokenize(sent):
    pat = re.compile(r'[\w]+|[.,!?;|]')
    if isinstance(sent, str):
        return pat.findall(sent.lower())
    else:
        return []

def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


def init_hvd_cuda(enable_hvd=True, enable_gpu=True):
    hvd = None
    if enable_hvd:
        import horovod.torch as hvd

        hvd.init()
        logging.info(
            f"hvd_size:{hvd.size()}, hvd_rank:{hvd.rank()}, hvd_local_rank:{hvd.local_rank()}"
        )

    hvd_size = hvd.size() if enable_hvd else 1
    hvd_rank = hvd.rank() if enable_hvd else 0
    hvd_local_rank = hvd.local_rank() if enable_hvd else 0

    if enable_gpu:
        torch.cuda.set_device(hvd_local_rank)

    return hvd_size, hvd_rank, hvd_local_rank


class dummy_context_mgr():
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc_value, traceback):
        return False


def setuplogger():
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    handler = logging.StreamHandler(sys.stdout)
    handler.setLevel(logging.INFO)
    formatter = logging.Formatter("[%(levelname)s %(asctime)s] %(message)s")
    handler.setFormatter(formatter)
    root.addHandler(handler)


def dump_args(args):
    for arg in dir(args):
        if not arg.startswith("_"):
            logging.info(f"args[{arg}]={getattr(args, arg)}")


def acc(y_true, y_hat):
    if y_true.shape == y_hat.shape:
        return torch.mean(ndcg_score(y_true, y_hat))
    y_hat = torch.argmax(y_hat, dim=-1)
    tot = y_true.shape[0]
    hit = torch.sum(y_true == y_hat)
    return hit.data.float() * 1.0 / tot


def dcg_score(y_true, y_hat, k=10):
    if y_true.shape == y_hat.shape:
        return torch.mean(dcg(y_true, y_hat))
    order = np.argsort(y_hat)[::-1]
    y_true = np.take(y_true, order[:k])
    gains = 2**y_true - 1
    discounts = np.log2(np.arange(len(y_true)) + 2)
    return np.sum(gains / discounts)


def ndcg_score(y_true, y_hat, k=10):
    if y_true.shape == y_hat.shape:
        return torch.mean(ndcg(y_true, y_hat))
    best = dcg_score(y_true, y_true, k)
    actual = dcg_score(y_true, y_hat, k)
    return actual / best


def mrr_score(y_true, y_hat):
    if y_true.shape == y_hat.shape:
        return torch.mean(mrr(y_true, y_hat))    
    order = np.argsort(y_hat)[::-1]
    y_true = np.take(y_true, order)
    rr_score = y_true / (np.arange(len(y_true)) + 1)
    return np.sum(rr_score) / np.sum(y_true)


def load_matrix(embedding_file_path, word_dict, word_embedding_dim):
    embedding_matrix = np.random.uniform(size=(len(word_dict) + 1,
                                               word_embedding_dim))
    have_word = []
    if embedding_file_path is not None:
        with open(embedding_file_path, 'rb') as f:
            while True:
                line = f.readline()
                if len(line) == 0:
                    break
                line = line.split()
                word = line[0].decode()
                if word in word_dict:
                    index = word_dict[word]
                    tp = [float(x) for x in line[1:]]
                    embedding_matrix[index] = np.array(tp)
                    have_word.append(word)
    return embedding_matrix, have_word


def latest_checkpoint(directory):
    if not os.path.exists(directory):
        return None
    logging.info(f'finding latest_checkpoint in directory {directory} :')        
    logging.info(os.listdir(directory))
    if len(os.listdir(directory))==0:
        return None

    def sorted_ls(directory):
        mtime = lambda f: os.stat(os.path.join(directory, f)).st_mtime
        return list(sorted(os.listdir(directory), key=mtime))

    all_files = sorted_ls(directory)
    all_checkpoints = [file_name for file_name in all_files if file_name.startswith('epoch')]
    if len(all_checkpoints) == 0:
        return None
    logging.info(f'found latest_checkpoint in directory {directory} : {all_checkpoints[-1]}')
    return os.path.join(directory, all_checkpoints[-1])


def get_checkpoint(directory, ckpt_name):
    ckpt_path = os.path.join(directory, ckpt_name)
    if os.path.exists(ckpt_path):
        return ckpt_path
    else:
        return None


# execute function to process the batch by multi-threading
def parallel(func, batch, batch_size=None, n_jobs=multiprocessing.cpu_count() - 1, prefer='threads', synchronize=False, *args):
    logging.debug("Start the batch processing, there are {} batch.".format(len(batch)))
    num_of_batches = (multiprocessing.cpu_count() - 1) if (len(batch) > multiprocessing.cpu_count() - 1) else len(batch)    
    if batch_size is None:
        batch_size = int(len(batch) / num_of_batches)
    partitions = minibatch(batch, size=batch_size)
    n_jobs = min(n_jobs, num_of_batches)
    results = []
    if synchronize or len(batch) <= batch_size:
        for batch in partitions:
            results.append(func(*args, batch))
    else:
        executor = Parallel(n_jobs=n_jobs, backend='loky' if prefer == 'processes' else 'threading',
                            prefer='processes' if prefer == 'processes' else "threads")
        results = executor(delayed(partial(func, *args))(batch) for batch in partitions)
    # merge the results from multiple batches back into list of batch
    merged_results = []
    for result in results:
        merged_results.extend(result)
    logging.debug(('sample result after batch processing: {}'.format(merged_results[0])))
    logging.debug(('number of final result after batch processing: {}'.format(len(merged_results))))
    return merged_results


def add_metrics(writer, metrics, global_step):
    logging.debug('adding metrics to tensorboard: {}'.format(metrics))
    AUC, MRR, nDCG5, nDCG10 = metrics
    writer.add_scalar("AUC", AUC, global_step)
    writer.add_scalar("MRR", MRR, global_step)
    writer.add_scalar("nDCG5", nDCG5, global_step)
    writer.add_scalar("nDCG10", nDCG10, global_step)
    writer.add_scalars('Summary', {'AUC': AUC, 'MRR': MRR, 'nDCG5': nDCG5, 'nDCG10': nDCG10}, global_step)
    logging.debug('added metrics to tensorboard: {}'.format(metrics))


def add_embedding(model, writer, list_element_dict, args, global_step):
    if not args.enable_weight_tfboard:
        logging.info('skip adding weight to tfboard, enable_weight_tfboard is set to false')
        return    
    element_names = [element_name for element_name in list_element_dict.keys()]
    logging.info('adding embedding to tensorboard: {}'.format(element_names))
    embeddings = {}
    for element_name, element_dict in list_element_dict.items():
        index2element = dict([(value, key) for (key, value) in element_dict.items()])
        index2element[0] = 'OOV'
        for name, param in model.named_parameters():
            if element_name in name.split('.'):
                embeddings[name] = param
                writer.add_embedding(param, metadata=[label for idx, label in sorted(index2element.items())], global_step=global_step, tag=element_name)
    logging.info('added embedding to tensorboard: {}'.format(element_names))
    embedding_path = os.path.join(os.path.expanduser(args.model_dir), 'embeddings.pt')    
    logging.info('saving embedding to: {}'.format(embedding_path))    
    torch.save(embeddings, embedding_path)
    logging.info('saved embedding to: {}'.format(embedding_path))  


def add_weight_histograms(model, writer, args, global_step):
    if not args.enable_weight_tfboard:
        logging.info('skip adding weight to tfboard, enable_weight_tfboard is set to false')
        return
    logging.debug('adding weight histograms to tensorboard: layer {}'.format(args.num_layers-1))
    layer_number = args.num_layers-1
    layer = 'encoder.layer.{}.output.dense.weight'.format(layer_number)
    for name, param in model.named_parameters(): 
        if layer in name:  
            flattened_weights = param.flatten()
            tag = f"layer_{layer_number}"
            writer.add_histogram(tag, flattened_weights, global_step=global_step, bins='tensorflow')
    logging.debug('added weight histograms to tensorboard: {}'.format(args.num_layers-1))
