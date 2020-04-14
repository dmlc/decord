"""DataLoader for video datasets."""
import math
import warnings
import numpy as np
from ..ndarray import cpu, gpu
from ..bridge.mxnet import try_import_mxnet

try_import_mxnet()
from mxnet.gluon.data import DataLoader
from mxnet.gluon.data import sampler as _sampler

class ShardedDataLoader(object):
    def __init__(self, dataset, batch_size=None, shuffle=False, sampler=None,
                 last_batch=None, batch_sampler=None, batchify_fn=None,
                 num_workers=0, pin_memory=False, pin_device_id=0,
                 prefetch=None, thread_pool=False, timeout=120, ctx_list=[cpu()]):
        self._dataset = dataset
        self._pin_memory = pin_memory
        self._pin_device_id = pin_device_id
        self._thread_pool = thread_pool
        self._timeout = timeout
        assert timeout > 0, "timeout must be positive, given {}".format(timeout)

        # try to split the batch into shards for contexts
        num_ctx = len(ctx_list)
        bs_per_ctx = math.ceil(batch_size / float(num_ctx))
        bs_list = [bs_per_ctx] * (num_ctx - 1)
        bs_list += batch_size - sum(bs_list)
        assert bs_list[-1] <= bs_per_ctx
        self._bs_list = np.cumsum([0] + bs_list)

        if pin_memory:
            warnings.warn('pin_memory not supported.')
            pin_memory = False

        if batch_sampler is None:
            if batch_size is None:
                raise ValueError("batch_size must be specified unless " \
                                 "batch_sampler is specified")
            if sampler is None:
                if shuffle:
                    sampler = _sampler.RandomSampler(len(dataset))
                else:
                    sampler = _sampler.SequentialSampler(len(dataset))
            elif shuffle:
                raise ValueError("shuffle must not be specified if sampler is specified")

            batch_sampler = _sampler.BatchSampler(
                sampler, batch_size, last_batch if last_batch else 'keep')
        elif batch_size is not None or shuffle or sampler is not None or \
                last_batch is not None:
            raise ValueError("batch_size, shuffle, sampler and last_batch must " \
                             "not be specified if batch_sampler is specified.")

        self._batch_sampler = batch_sampler
        self._num_workers = num_workers if num_workers >= 0 else 0

    def __iter__(self):
        for batch in self._batch_sampler:
            ret = []
            for start, end in zip(self._bs_list[:-1], self._bs_list[1:]):
                if start < len(batch):
                    break
                indices = batch[start:min(end, len(batch))]
                ret.append(self._batchify_fn([self._dataset[idx] for idx in indices]))
            yield ret
