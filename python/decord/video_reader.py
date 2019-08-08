"""Video Reader."""
from __future__ import absolute_import

import ctypes
import numpy as np

from ._ffi.base import c_array, c_str
from ._ffi.function import _init_api
from ._ffi.ndarray import DECORDContext
from .base import DECORDError
from . import ndarray as _nd
from .ndarray import cpu, gpu
from .bridge import bridge_out

VideoReaderHandle = ctypes.c_void_p


class VideoReader(object):
    def __init__(self, uri, ctx=cpu(0), width=-1, height=-1):
        assert isinstance(ctx, DECORDContext)
        self._handle = None
        self._handle = _CAPI_VideoReaderGetVideoReader(
            uri, ctx.device_type, ctx.device_id, width, height)
        if self._handle is None:
            raise RuntimeError("Error reading " + uri + "...")
        self._num_frame = _CAPI_VideoReaderGetFrameCount(self._handle)
        assert self._num_frame > 0
        self._key_indices = _CAPI_VideoReaderGetKeyIndices(self._handle)

    @property
    def num_frame(self):
        return self._num_frame

    def __del__(self):
        if self._handle:
            _CAPI_VideoReaderFree(self._handle)

    def __len__(self):
        return self._num_frame

    def __getitem__(self, idx):
        if idx >= self._num_frame:
            raise IndexError("Index: {} out of bound: {}".format(idx, self._num_frame))
        self.seek_accurate(idx)
        return self.next()

    def next(self):
        assert self._handle is not None
        arr = _CAPI_VideoReaderNextFrame(self._handle)
        if not arr.shape:
            raise StopIteration()
        return bridge_out(arr)

    def get_batch(self, indices):
        assert self._handle is not None
        indices = _nd.array(np.array(indices))
        arr = _CAPI_VideoReaderGetBatch(self._handle, indices)
        return bridge_out(arr)

    def get_key_indices(self):
        return bridge_out(self._key_indices)

    def seek(self, pos):
        assert self._handle is not None
        assert pos >= 0 and pos < self._num_frame
        success = _CAPI_VideoReaderSeek(self._handle, pos)
        if not success:
            raise RuntimeError("Failed to seek to frame {}".format(pos))

    def seek_accurate(self, pos):
        assert self._handle is not None
        assert pos >= 0 and pos < self._num_frame
        success = _CAPI_VideoReaderSeekAccurate(self._handle, pos)
        if not success:
            raise RuntimeError("Failed to seek_accurate to frame {}".format(pos))

    def skip_frames(self, num=1):
        assert self._handle is not None
        assert num > 0
        _CAPI_VideoReaderSkipFrames(self._handle, num)

_init_api("decord.video_reader")
