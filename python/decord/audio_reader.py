"""Audio Reader."""
from __future__ import absolute_import

import ctypes
import numpy as np
import math

from ._ffi.base import c_array, c_str
from ._ffi.function import _init_api
from ._ffi.ndarray import DECORDContext
from .base import DECORDError
from . import ndarray as _nd
from .ndarray import cpu, gpu
from .bridge import bridge_out

AudioReaderHandle = ctypes.c_void_p

class AudioReader(object):

    def __init__(self, uri, ctx=cpu(0), sample_rate=44100):
        self._handle = None
        assert isinstance(ctx, DECORDContext)
        if hasattr(uri, 'read'):
            ba = bytearray(uri.read())
            uri = '{} bytes'.format(len(ba))
            self._handle = _CAPI_AudioReaderGetAudioReader(
                ba, ctx.device_type, ctx.device_id, sample_rate, 2)
        else:
            self._handle = _CAPI_AudioReaderGetAudioReader(
                uri, ctx.device_type, ctx.device_id, sample_rate, 0)
        if self._handle is None:
            raise RuntimeError("Error reading " + uri + "...")
        self._array = _CAPI_AudioReaderGetNDArray(self._handle)
        self._array = bridge_out(self._array)
        self._array = self._array.asnumpy()
        self._duration = _CAPI_AudioReaderGetDuration(self._handle)
        self._num_samples_per_channel = _CAPI_AudioReaderGetNumSamplesPerChannel(self._handle)
        self._num_channels = _CAPI_AudioReaderGetNumChannels(self._handle)
        self.sample_rate = sample_rate


    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.get_batch(list(range(*idx.indices(self._num_samples_per_channel))))
        if idx < 0:
            idx += self._num_samples_per_channel
        if idx >= self._num_samples_per_channel or idx < 0:
            raise IndexError("Index: {} out of bound: {}".format(idx, self._num_samples_per_channel))
        return self._array[:, idx]
        
    def get_batch(self, indices):
        return self._array[:, indices]

    def size(self):
        return (self._num_channels, self._num_samples_per_channel)

    def duration(self):
        return self._duration

    def get_num_padding(self):
        self._num_padding = _CAPI_AudioReaderGetNumPaddingSamples(self._handle)
        return self._num_padding

    def get_info(self):
        _CAPI_AudioReaderGetInfo(self._handle)

    def __time_to_sample(self, timestamp):
        return math.ceil(timestamp * self.sample_rate)
        
    def __times_to_samples(self, timestamps):
        return [self.__time_to_sample(timestamp) for timestamp in timestamps]

_init_api("decord.audio_reader")
