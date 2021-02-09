"""AV Reader."""
from __future__ import absolute_import

import ctypes
import numpy as np
import math
from .video_reader import VideoReader
from .audio_reader import AudioReader

from .ndarray import cpu, gpu
from . import ndarray as _nd
from .bridge import bridge_out

class AVReader(object):
    """Individual audio video reader with convenient indexing function.

    Parameters
    ----------
    uri: str
        Path of file.
    ctx: decord.Context
        The context to decode the file, can be decord.cpu() or decord.gpu().
    sample_rate: int, default is -1
        Desired output sample rate of the audio, unchanged if `-1` is specified.
    mono: bool, default is True
        Desired output channel layout of the audio. `True` is mono layout. `False` is unchanged.
    width : int, default is -1
        Desired output width of the video, unchanged if `-1` is specified.
    height : int, default is -1
        Desired output height of the video, unchanged if `-1` is specified.
    num_threads : int, default is 0
        Number of decoding thread, auto if `0` is specified.
    fault_tol : int, default is -1
        The threshold of corupted and recovered frames. This is to prevent silent fault
        tolerance when for example 50% frames of a video cannot be decoded and duplicate
        frames are returned. You may find the fault tolerant feature sweet in many cases,
        but not for training models. Say `N = # recovered frames`
        If `fault_tol` < 0, nothing will happen.
        If 0 < `fault_tol` < 1.0, if N > `fault_tol * len(video)`, raise `DECORDLimitReachedError`.
        If 1 < `fault_tol`, if N > `fault_tol`, raise `DECORDLimitReachedError`.

    """

    def __init__(self, uri, ctx=cpu(0), sample_rate=44100, mono=True, width=-1, height=-1, num_threads=0, fault_tol=-1):
        self.__audio_reader = AudioReader(uri, ctx, sample_rate, mono)
        self.__audio_reader.add_padding()
        if hasattr(uri, 'read'):
            uri.seek(0)
        self.__video_reader = VideoReader(uri, ctx, width, height, num_threads, fault_tol)

    def __len__(self):
        """Get length of the video. Note that sometimes FFMPEG reports inaccurate number of frames,
        we always follow what FFMPEG reports.
        Returns
        -------
        int
            The number of frames in the video file.
        """
        return len(self.__video_reader)

    def __getitem__(self, idx):
        """Get audio samples and video frame at `idx`.

        Parameters
        ----------
        idx : int or slice
            The frame index, can be negative which means it will index backwards,
            or slice of frame indices.

        Returns
        -------
        (ndarray/list of ndarray, ndarray)
            First element is samples of shape CxS or a list of length N containing samples of shape CxS,
            where N is the number of frames, C is the number of channels, 
            S is the number of samples of the corresponding frame.

            Second element is Frame of shape HxWx3 or batch of image frames with shape NxHxWx3,
            where N is the length of the slice.
        """
        assert self.__video_reader is not None and self.__audio_reader is not None
        if isinstance(idx, slice):
            return self.get_batch(range(*idx.indices(len(self.__video_reader))))
        if idx < 0:
            idx += len(self.__video_reader)
        if idx >= len(self.__video_reader) or idx < 0:
            raise IndexError("Index: {} out of bound: {}".format(idx, len(self.__video_reader)))
        audio_start_idx, audio_end_idx = self.__video_reader.get_frame_timestamp(idx)
        audio_start_idx = self.__audio_reader._time_to_sample(audio_start_idx)
        audio_end_idx = self.__audio_reader._time_to_sample(audio_end_idx)
        return (self.__audio_reader[audio_start_idx:audio_end_idx], self.__video_reader[idx])
        
    def get_batch(self, indices):
        """Get entire batch of audio samples and video frames.

        Parameters
        ----------
        indices : list of integers
            A list of frame indices. If negative indices detected, the indices will be indexed from backward
        Returns
        -------
        (list of ndarray, ndarray)
            First element is a list of length N containing samples of shape CxS,
            where N is the number of frames, C is the number of channels, 
            S is the number of samples of the corresponding frame.

            Second element is Frame of shape HxWx3 or batch of image frames with shape NxHxWx3,
            where N is the length of the slice.

        """
        assert self.__video_reader is not None and self.__audio_reader is not None
        indices = self._validate_indices(indices)
        audio_arr = []
        prev_video_idx = None
        prev_audio_end_idx = None
        for idx in list(indices):
            frame_start_time, frame_end_time = self.__video_reader.get_frame_timestamp(idx)
            # timestamp and sample conversion could have some error that could cause non-continuous audio
            # we detect if retrieving continuous frame and make the audio continuous
            if prev_video_idx and idx == prev_video_idx+1:
                audio_start_idx = prev_audio_end_idx
            else:
                audio_start_idx = self.__audio_reader._time_to_sample(frame_start_time)
            audio_end_idx = self.__audio_reader._time_to_sample(frame_end_time)
            audio_arr.append(self.__audio_reader[audio_start_idx:audio_end_idx])
            prev_video_idx = idx
            prev_audio_end_idx = audio_end_idx
        return (audio_arr, self.__video_reader.get_batch(indices))

    def _get_slice(self, sl):
        audio_arr = np.empty(shape=(self.__audio_reader.shape()[0], 0), dtype='float32')
        for idx in list(sl):
            audio_start_idx, audio_end_idx = self.__video_reader.get_frame_timestamp(idx)
            audio_start_idx = self.__audio_reader._time_to_sample(audio_start_idx)
            audio_end_idx = self.__audio_reader._time_to_sample(audio_end_idx)
            audio_arr = np.concatenate((audio_arr, self.__audio_reader[audio_start_idx:audio_end_idx].asnumpy()), axis=1)
        return (bridge_out(_nd.array(audio_arr)), self.__video_reader.get_batch(sl))

    def _validate_indices(self, indices):
        """Validate int64 integers and convert negative integers to positive by backward search"""
        assert self.__video_reader is not None and self.__audio_reader is not None
        indices = np.array(indices, dtype=np.int64)
        # process negative indices
        indices[indices < 0] += len(self.__video_reader)
        if not (indices >= 0).all():
            raise IndexError(
                'Invalid negative indices: {}'.format(indices[indices < 0] + len(self.__video_reader)))
        if not (indices < len(self.__video_reader)).all():
            raise IndexError('Out of bound indices: {}'.format(indices[indices >= len(self.__video_reader)]))
        return indices
