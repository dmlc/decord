"""AV Reader."""
from __future__ import absolute_import

import ctypes
import numpy as np
import math
from .video_reader import VideoReader
from .audio_reader import AudioReader

from .ndarray import cpu, gpu

class AVReader(object):

    def __init__(self, uri, ctx=cpu(0), sample_rate=44100, width=-1, height=-1, num_threads=0, fault_tol=-1):
        self.__audio_reader = AudioReader(uri, ctx, sample_rate)
        self.__audio_reader.add_padding()
        if hasattr(uri, 'read'):
            uri.seek(0)
        self.__video_reader = VideoReader(uri, ctx, width, height, num_threads, fault_tol)

    def __len__(self):
        return len(self.__video_reader)

    def __getitem__(self, idx):
        if isinstance(idx, slice):
            return self.get_batch(range(*idx.indices(len(self.__video_reader))))
        if idx < 0:
            idx += len(self.__video_reader)
        if idx >= len(self.__video_reader) or idx < 0:
            raise IndexError("Index: {} out of bound: {}".format(idx, len(self.__video_reader)))
        audio_start_idx, audio_end_idx = self.__video_reader.get_frame_timestamp(idx)
        audio_start_idx = self.__audio_reader.time_to_sample(audio_start_idx)
        audio_end_idx = self.__audio_reader.time_to_sample(audio_end_idx)
        return (self.__audio_reader[audio_start_idx:audio_end_idx], self.__video_reader[idx])
        
    def get_batch(self, indices):
        audio_arr = []
        for idx in list(indices):
            audio_start_idx, audio_end_idx = self.__video_reader.get_frame_timestamp(idx)
            audio_start_idx = self.__audio_reader.time_to_sample(audio_start_idx)
            audio_end_idx = self.__audio_reader.time_to_sample(audio_end_idx)
            audio_arr.append(self.__audio_reader[audio_start_idx:audio_end_idx])
        return (audio_arr, self.__video_reader.get_batch(indices))

    

