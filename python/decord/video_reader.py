"""Video Reader."""
from __future__ import absolute_import

import ctypes
import numpy as np

from ._ffi.base import c_array, c_str
from ._ffi.function import _init_api
from .base import DECORDError

VideoReaderHandle = ctypes.c_void_p


class VideoReader(object):
    def __init__(self, uri, width=-1, height=-1):
        self._handle = _CAPI_VideoGetVideoReader(
            uri, width, height)

    def next(self):
        assert self._handle is not None
        arr = _CAPI_VideoNextFrame(self._handle)
        if not arr.shape:
            raise StopIteration()
        return arr

    def get_key_indices(self):
        assert self._handle is not None
        indices = _CAPI_VideoGetKeyIndices(self._handle)
        if not indices.shape:
            raise RuntimeError("No key frame indices found.")
        return indices

_init_api("decord.video_reader")