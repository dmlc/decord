"""DataLoader for video datasets."""
import math
import warnings
from multiprocessing.pool import ThreadPool

import numpy as np
from ..ndarray import cpu, gpu
from ..bridge.mxnet import try_import_mxnet

try_import_mxnet()
from mxnet.gluon.data.dataloader import DataLoader, default_batchify_fn, _thread_worker_initializer
from mxnet.gluon.data import sampler as _sampler
from mxnet.util import is_np_shape, is_np_array


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
        self._ctx_list = ctx_list
        assert timeout > 0, "timeout must be positive, given {}".format(timeout)

        # try to split the batch into shards for contexts
        num_ctx = len(ctx_list)
        bs_per_ctx = math.ceil(batch_size / float(num_ctx))
        bs_list = [bs_per_ctx] * (num_ctx - 1)
        bs_list += [batch_size - sum(bs_list)]
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
        if batchify_fn is None:
            self._batchify_fn = default_batchify_fn
        else:
            self._batchify_fn = batchify_fn

        self._iter = None
        self._pool = None
        if len(ctx_list) == 1 and ctx_list[0] == cpu() and num_workers > 0:
            self._iter = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                    batch_sampler=batch_sampler, last_batch=last_batch,
                                    batchify_fn=batchify_fn, num_workers=num_workers,
                                    pin_memory=pin_memory, pin_device_id=pin_device_id,
                                    prefetch=prefetch, thread_pool=thread_pool, timeout=timeout)
        elif num_workers > 0:
            self._pool = ThreadPool(self._num_workers,
                                    initializer=_thread_worker_initializer,
                                    initargs=(is_np_shape(), is_np_array()))

    def __iter__(self):
        if self._num_workers <= 0:
            for batch in self._batch_sampler:
                ret = []
                for ctx, start, end in zip(self._ctx_list, self._bs_list[:-1], self._bs_list[1:]):
                    if start >= len(batch):
                        break
                    indices = batch[start:min(end, len(batch))]
                    ret.append(self._batchify_fn([self._dataset[(idx, ctx)] for idx in indices]))
                yield ret
        elif self._iter is not None:
            for ret in self.iter:
                if not isinstance(ret, list):
                    ret = [ret]
                yield ret
        else:
            assert self._pool is not None
            for batch in self._batch_sampler:
                async_rets = []
                for ctx, start, end in zip(self._ctx_list, self._bs_list[:-1], self._bs_list[1:]):
                    if start >= len(batch):
                        break
                    indices = batch[start:min(end, len(batch))]
                    async_rets.append(self._pool.map_async(self._dataset.__getitem__, [(idx, ctx) for idx in indices]))
                tmp_rets = [r.get(self._timeout) for r in async_rets]
                ret = self._pool.map(self._batchify_fn, tmp_rets)
                yield ret

    def __len__(self):
        return len(self._batch_sampler)
