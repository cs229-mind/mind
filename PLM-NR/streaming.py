import os
import logging
import fnmatch
import random
import numpy as np
import tensorflow as tf
# tf.data.experimental.enable_debug_mode()
import utils


def get_files(dirname, filename_pat="*", recursive=False):
    print(f'get_files: {dirname}, {filename_pat}')
    if not tf.io.gfile.exists(dirname):
        return None
    files = []
    for x in tf.io.gfile.listdir(dirname):
        path = os.path.join(dirname, x)
        if tf.io.gfile.isdir(path):
            if recursive:
                files.extend(get_files(path, filename_pat))
        elif fnmatch.fnmatch(x, filename_pat):
            files.append(path)
    print(f'files: {files}')
    return files


def get_worker_files(dirname,
                     worker_rank,
                     worker_size,
                     filename_pat="*",
                     shuffle=False,
                     seed=0):
    """Get file paths belong to one worker."""
    all_files = get_files(dirname, filename_pat)
    all_files.sort()
    if shuffle:
        # g_mutex = threading.Lock()
        # g_mutex.acquire()
        random.seed(seed)
        random.shuffle(all_files)
        # g_mutex.release()
    files = []
    for i in range(worker_rank, len(all_files), worker_size):
        files.append(all_files[i])
    logging.info(
        f"worker_rank:{worker_rank}, worker_size:{worker_size}, shuffle:{shuffle}, seed:{seed}, directory:{dirname}, files:{files}"
    )
    return files


class StreamReader:
    def __init__(self, data_paths, batch_size, shuffle=False, shuffle_buffer_size=1000):
        tf.config.experimental.set_visible_devices([], device_type="GPU")
        # logging.info(f"visible_devices:{tf.config.experimental.get_visible_devices()}")
        logging.info(f"data_paths: {data_paths}")
        path_len = len(data_paths)
        # logging.info(f"[StreamReader] path_len:{path_len}, paths: {data_paths}")
        dataset = tf.data.Dataset.list_files(data_paths).interleave(
            lambda x: tf.data.TextLineDataset(x),
            cycle_length=path_len,
            block_length=128,
            num_parallel_calls=min(path_len, 64),
        )
        dataset = dataset.interleave(
            lambda x: tf.data.Dataset.from_tensor_slices(
                self._process_record(x)),
            cycle_length=path_len,
            block_length=1,
            num_parallel_calls=tf.data.AUTOTUNE)

        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)

        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        self.next_batch = iter(dataset)
        # self.session = None

    def _process_record(self, record):
        # iid, uid, time, his, impr
        tf.print('before _process_record')
        tf.print(record)
        records = tf.strings.split([record], '\t').values
        sess = tf.strings.split([records[4]], ' ').values  # (num)
        sess_label = tf.strings.split(sess, '-').values

        sess_poss = tf.gather(sess_label, tf.where(tf.equal(sess_label, '1'))-1)
        record = tf.expand_dims(record, axis=0)
        poss_num = tf.size(sess_poss)

        # return sess_poss[:, 0], tf.compat.v1.repeat(record, poss_num, axis=0)
        tf.print('after _process_record')        
        tf.print(sess_poss[:, 0], tf.tile(record, [poss_num]))
        return sess_poss[:, 0], tf.tile(record, [poss_num])

    def reset(self):
        self.endofstream = False

    def get_next(self):
        try:
            ret = next(self.next_batch)
        except tf.errors.OutOfRangeError:
            self.endofstream = True
            return None
        return ret

    def reach_end(self):
        # print(f"StreamReader reach_end(), {self.endofstream}")
        return self.endofstream


class StreamSampler:
    def __init__(
        self,
        data_dir,
        filename_pat,
        batch_size,
        worker_rank,
        worker_size,
        enable_shuffle=False,
        shuffle_buffer_size=1000,
        shuffle_seed=0,
    ):
        data_paths = get_worker_files(
            data_dir,
            worker_rank,
            worker_size,
            filename_pat,
            shuffle=enable_shuffle,
            seed=shuffle_seed,
        )
        self.stream_reader = StreamReader(
            data_paths, 
            batch_size,
            enable_shuffle,
            shuffle_buffer_size
            )

    def __iter__(self):
        self.stream_reader.reset()
        return self

    def __next__(self):
        """Implement iterator interface."""
        # logging.info(f"[StreamSampler] __next__")
        next_batch = self.stream_reader.get_next()
        if not isinstance(next_batch, np.ndarray) and not isinstance(
                next_batch, tuple):
            raise StopIteration
        print(next_batch.shape)
        return next_batch

    def reach_end(self):
        return self.stream_reader.reach_end()


class StreamReaderTest(StreamReader):
    def __init__(self, data_paths, batch_size, shuffle=False, shuffle_buffer_size=1000):
        tf.config.experimental.set_visible_devices([], device_type="GPU")
        logging.info(f"visible_devices:{tf.config.experimental.get_visible_devices()}")
        path_len = len(data_paths)
        logging.info(f"[StreamReader] path_len:{path_len}, paths: {data_paths}")
        dataset = tf.data.Dataset.list_files(data_paths).interleave(
            lambda x: tf.data.TextLineDataset(x),
            cycle_length=path_len,
            block_length=128,
            num_parallel_calls=min(path_len, 64),
        )

        if shuffle:
            dataset = dataset.shuffle(shuffle_buffer_size, reshuffle_each_iteration=True)
        dataset = dataset.batch(batch_size)
        dataset = dataset.prefetch(1)
        self.next_batch = iter(dataset)
        # self.session = None


class StreamSamplerTest(StreamSampler):
    def __init__(
        self,
        data_dir,
        filename_pat,
        batch_size,
        worker_rank,
        worker_size,
        enable_shuffle=False,
        shuffle_buffer_size=1000,
        shuffle_seed=0,
    ):
        data_paths = get_worker_files(
            data_dir,
            worker_rank,
            worker_size,
            filename_pat,
            shuffle=enable_shuffle,
            seed=shuffle_seed,
        )
        self.stream_reader = StreamReaderTest(
            data_paths, 
            batch_size, 
            enable_shuffle,
            shuffle_buffer_size)

    def __next__(self):
        """Implement iterator interface."""
        # logging.info(f"[StreamSampler] __next__")
        next_batch = self.stream_reader.get_next()
        # print(next_batch.shape)
        return next_batch


if __name__ == "__main__":
    utils.setuplogger()
    print("start")
    sampler = StreamSampler(
        "../MIND/dev",
        "behaviors*.tsv", 4, 0, 1)

    import time
    for i in sampler:
        logging.info("sampler")
        logging.info(i)
        time.sleep(5)
