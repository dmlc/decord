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
    """Individual audio reader with convenient indexing function.

    Parameters
    ----------
    uri: str
        Path of file.
    ctx: decord.Context
        The context to decode the file, can be decord.cpu() or decord.gpu().
    sample_rate: int, default is -1
        Desired output sample rate of the audio, unchanged if `-1` is specified.
    mono: bool, default is True
        Desired output channel layout of the audio. 
        Setting `True` will return the audio as mono layout. 
        Setting `False` will return the audio channel layout intact.

    """

    def __init__(self, uri, ctx=cpu(0), sample_rate=-1, mono=True):
        self._handle = None
        assert isinstance(ctx, DECORDContext)
        is_mono = 1 if mono else 0
        if hasattr(uri, 'read'):
            ba = bytearray(uri.read())
            uri = '{} bytes'.format(len(ba))
            self._handle = _CAPI_AudioReaderGetAudioReader(
                ba, ctx.device_type, ctx.device_id, sample_rate, 2, is_mono)
        else:
            self._handle = _CAPI_AudioReaderGetAudioReader(
                uri, ctx.device_type, ctx.device_id, sample_rate, 0, is_mono)
        if self._handle is None:
            raise RuntimeError("Error reading " + uri + "...")
        self._array = _CAPI_AudioReaderGetNDArray(self._handle)
        self._array = self._array.asnumpy()
        self._duration = _CAPI_AudioReaderGetDuration(self._handle)
        self._num_samples_per_channel = _CAPI_AudioReaderGetNumSamplesPerChannel(self._handle)
        self._num_channels = _CAPI_AudioReaderGetNumChannels(self._handle)
        self.sample_rate = sample_rate
        self._num_padding = None

    def __len__(self):
        """Get length of the audio. The length refer to the shape's first dimension. In this case,
        the length is the number of channels.
        Returns
        -------
        int
            The number of channels in the audio track.
        """
        return self.shape[0]

    def __del__(self):
        if self._handle:
            _CAPI_AudioReaderFree(self._handle)

    def __getitem__(self, idx):
        """Get sample at `idx`. idx is the index of resampled audio, unit is sample.

        Parameters
        ----------
        idx : int or slice
            The sample index, can be negative which means it will index backwards,
            or slice of sample indices.

        Returns
        -------
        ndarray
            Samples of shape CxS,
            where C is the number of channels, S is the number of samples of the index or slice.
        """
        assert self._handle is not None
        if isinstance(idx, slice):
            return self.get_batch(range(*idx.indices(self._num_samples_per_channel)))
        if idx < 0:
            idx += self._num_samples_per_channel
        if idx >= self._num_samples_per_channel or idx < 0:
            raise IndexError("Index: {} out of bound: {}".format(idx, self._num_samples_per_channel))
        return bridge_out(_nd.array(self._array[:, idx]))
        
    def get_batch(self, indices):
        """Get entire batch of samples.

        Parameters
        ----------
        indices : list of integers
            A list of frame indices. If negative indices detected, the indices will be indexed from backward
        Returns
        -------
        ndarray
            Samples of shape CxS,
            where C is the number of channels, S is the number of samples of the slice.

        """
        assert self._handle is not None
        indices = self._validate_indices(indices)
        indices = list(indices)
        return bridge_out(_nd.array(self._array[:, indices]))

    @property
    def shape(self):
        """Get shape of the entire audio samples.

        Returns
        -------
        (int, int)
            The number of channels, and the number of samples in each channel.

        """
        return (self._num_channels, self._num_samples_per_channel)

    def duration(self):
        """Get duration of the audio.

        Returns
        -------
        double
            Duration of the audio in secs.

        """
        return self._duration

    def __get_num_padding(self):
        """Get number of samples needed to pad the audio to start at time 0."""
        if self._num_padding is None:
            self._num_padding = _CAPI_AudioReaderGetNumPaddingSamples(self._handle)
        return self._num_padding

    def add_padding(self):
        """Pad the audio samples so that it starts at time 0.

        Returns
        -------
        int
            Number of samples padded

        """
        self._array = np.pad(self._array, ((0, 0), (self.__get_num_padding(), 0)), 'constant', constant_values=0)
        self._duration += self.__get_num_padding() * self.sample_rate
        return self.__get_num_padding()

    def get_info(self):
        """Log out the basic info about the audio stream."""
        _CAPI_AudioReaderGetInfo(self._handle)

    def _time_to_sample(self, timestamp):
        """Convert time in seconds to sample index"""
        return math.ceil(timestamp * self.sample_rate)
        
    def _times_to_samples(self, timestamps):
        """Convert times in seconds to sample indices"""
        return [self._time_to_sample(timestamp) for timestamp in timestamps]

    def _validate_indices(self, indices):
        """Validate int64 integers and convert negative integers to positive by backward search"""
        assert self._handle is not None
        indices = np.array(indices, dtype=np.int64)
        # process negative indices
        indices[indices < 0] += self._num_samples_per_channel
        if not (indices >= 0).all():
            raise IndexError(
                'Invalid negative indices: {}'.format(indices[indices < 0] + self._num_samples_per_channel))
        if not (indices < self._num_samples_per_channel).all():
            raise IndexError('Out of bound indices: {}'.format(indices[indices >= self._num_samples_per_channel]))
        return indices

_init_api("decord.audio_reader")
