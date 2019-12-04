"""Video Loader."""
from __future__ import absolute_import

import ctypes
import numpy as np

from ._ffi.base import c_array, c_str
from ._ffi.function import _init_api
from .base import DECORDError
from . import ndarray as _nd
from .ndarray import DECORDContext
from .bridge import bridge_out

VideoLoaderHandle = ctypes.c_void_p


class VideoLoader(object):
    """Multiple video loader with advanced shuffling and batching methods.

    Parameters
    ----------
    uris : list of str
        List of video paths.
    ctx : decord.Context or list of Context
        The context to decode the video file, can be decord.cpu() or decord.gpu().
        If ctx is a list, videos will be evenly split over many ctxs.
    shape : tuple
        Returned shape of the batch images, e.g., (2, 320, 240, 3) as (Batch, H, W, 3)
    interval : int
        Intra-batch frame interval.
    skip : int
        Inter-batch frame interval.
    shuffle : int
        Shuffling strategy. Can be
        `0`:  all sequential, no seeking, following initial filename order
        `1`:  random filename order, no random access for each video, very efficient
        `2`:  random order
        `3`:  random frame access in each video only.

    """
    def __init__(self, uris, ctx, shape, interval, skip, shuffle, prefetch=0):
        self._handle = None
        assert isinstance(uris, (list, tuple))
        assert (len(uris) > 0)
        uri = ','.join([x.strip() for x in uris])
        if isinstance(ctx, DECORDContext):
            ctx = [ctx]
        for _ctx in ctx:
            assert isinstance(_ctx, DECORDContext)
        device_types = _nd.array([x.device_type for x in ctx])
        device_ids = _nd.array([x.device_id for x in ctx])
        assert isinstance(shape, (list, tuple))
        assert len(shape) == 4, "expected shape: [bs, height, width, 3], given {}".format(shape)
        self._handle = _CAPI_VideoLoaderGetVideoLoader(
            uri, device_types, device_ids, shape[0], shape[1], shape[2], shape[3], interval, skip, shuffle, prefetch)
        assert self._handle is not None
        self._len = _CAPI_VideoLoaderLength(self._handle)
        self._curr = 0

    def __del__(self):
        if self._handle:
            _CAPI_VideoLoaderFree(self._handle)

    def __len__(self):
        """Get number of batches in each epoch.

        Returns
        -------
        int
            number of batches in each epoch.

        """
        return self._len

    def reset(self):
        """Reset loader for next epoch.

        """
        assert self._handle is not None
        self._curr = 0
        _CAPI_VideoLoaderReset(self._handle)

    def __next__(self):
        """Get the next batch.

        Returns
        -------
        ndarray, ndarray
            Frame data and corresponding indices in videos.
            Indices are [(n0, k0), (n1, k1)...] where n0 is the index of video, k0 is the index
            of frame in video n0.

        """
        assert self._handle is not None
        # avoid calling CAPI HasNext
        if self._curr >= self._len:
            raise StopIteration
        _CAPI_VideoLoaderNext(self._handle)
        data = _CAPI_VideoLoaderNextData(self._handle)
        indices = _CAPI_VideoLoaderNextIndices(self._handle)
        self._curr += 1
        return bridge_out(data), bridge_out(indices)

    def next(self):
        """Alias of __next__ for python2.

        """
        return self.__next__()

    def __iter__(self):
        assert self._handle is not None
        # if (self._curr >= self._len):
        #     self.reset()
        # else:
        #     err_msg = "Call __iter__ of VideoLoader during previous iteration is forbidden. \
        #         Consider using cached iterator by 'vl = iter(video_loader)' and reuse it."
        #     raise RuntimeError(err_msg)
        return self


_init_api("decord.video_loader")
