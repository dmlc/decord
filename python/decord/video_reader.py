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
    """Individual video reader with convenient indexing and seeking functions.

    Parameters
    ----------
    uri : str
        Path of video file.
    ctx : decord.Context
        The context to decode the video file, can be decord.cpu() or decord.gpu().
    width : int, default is -1
        Desired output width of the video, unchanged if `-1` is specified.
    height : int, default is -1
        Desired output height of the video, unchanged if `-1` is specified.
    num_threads : int, default is 0
        Number of decoding thread, auto if `0` is specified.

    """
    def __init__(self, uri, ctx=cpu(0), width=-1, height=-1, num_threads=0):
        self._handle = None
        assert isinstance(ctx, DECORDContext)
        if hasattr(uri, 'read'):
            ba = bytearray(uri.read())
            uri = '{} bytes'.format(len(ba))
            self._handle = _CAPI_VideoReaderGetVideoReader(
                ba, ctx.device_type, ctx.device_id, width, height, num_threads, 2)
        else:
            self._handle = _CAPI_VideoReaderGetVideoReader(
                uri, ctx.device_type, ctx.device_id, width, height, num_threads, 0)
        if self._handle is None:
            raise RuntimeError("Error reading " + uri + "...")
        self._num_frame = _CAPI_VideoReaderGetFrameCount(self._handle)
        assert self._num_frame > 0, "Invalid frame count: {}".format(self._num_frame)
        self._key_indices = None
        self._frame_pts = None
        self._avg_fps = None

    def __del__(self):
        if self._handle:
            _CAPI_VideoReaderFree(self._handle)

    def __len__(self):
        """Get length of the video. Note that sometimes FFMPEG reports inaccurate number of frames,
        we always follow what FFMPEG reports.

        Returns
        -------
        int
            The number of frames in the video file.

        """
        return self._num_frame

    def __getitem__(self, idx):
        """Get frame at `idx`.

        Parameters
        ----------
        idx : int or slice
            The frame index, can be negative which means it will index backwards,
            or slice of frame indices.

        Returns
        -------
        ndarray
            Frame of shape HxWx3 or batch of image frames with shape NxHxWx3,
            where N is the length of the slice.
        """
        if isinstance(idx, slice):
            return self.get_batch(range(*idx.indices(len(self))))
        if idx < 0:
            idx += self._num_frame
        if idx >= self._num_frame or idx < 0:
            raise IndexError("Index: {} out of bound: {}".format(idx, self._num_frame))
        self.seek_accurate(idx)
        return self.next()

    def next(self):
        """Grab the next frame.

        Returns
        -------
        ndarray
            Frame with shape HxWx3.

        """
        assert self._handle is not None
        arr = _CAPI_VideoReaderNextFrame(self._handle)
        if not arr.shape:
            raise StopIteration()
        return bridge_out(arr)

    def _validate_indices(self, indices):
        """Validate int64 integers and convert negative integers to positive by backward search"""
        assert self._handle is not None
        indices = np.array(indices, dtype=np.int64)
        # process negative indices
        indices[indices < 0] += self._num_frame
        if not (indices >= 0).all():
            raise IndexError(
                'Invalid negative indices: {}'.format(indices[indices < 0] + self._num_frame))
        if not (indices < self._num_frame).all():
            raise IndexError('Out of bound indices: {}'.format(indices[indices >= self._num_frame]))
        indices = _nd.array(indices)
        return indices

    def get_frame_timestamp(self, idx):
        """Get frame playback timestamp in unit(second).

        Parameters
        ----------
        indices: list of integers or slice
            A list of frame indices. If negative indices detected, the indices will be indexed from backward.

        Returns
        -------
        numpy.ndarray
            numpy.ndarray of shape (N, 2), where N is the size of indices. The format is `(start_second, end_second)`.
        """
        assert self._handle is not None
        if isinstance(idx, slice):
            idx = self.get_batch(range(*idx.indices(len(self))))
        idx = self._validate_indices(idx).asnumpy()
        if self._frame_pts is None:
            self._frame_pts = _CAPI_VideoReaderGetFramePTS(self._handle).asnumpy()
        return self._frame_pts[idx, :]


    def get_batch(self, indices):
        """Get entire batch of images. `get_batch` is optimized to handle seeking internally.
        Duplicate frame indices will be optmized by copying existing frames rather than decode
        from video again.

        Parameters
        ----------
        indices : list of integers
            A list of frame indices. If negative indices detected, the indices will be indexed from backward

        Returns
        -------
        ndarray
            An entire batch of image frames with shape NxHxWx3, where N is the length of `indices`.

        """
        assert self._handle is not None
        indices = self._validate_indices(indices)
        arr = _CAPI_VideoReaderGetBatch(self._handle, indices)
        return bridge_out(arr)

    def get_key_indices(self):
        """Get list of key frame indices.

        Returns
        -------
        list
            List of key frame indices.

        """
        if self._key_indices is None:
            self._key_indices = _CAPI_VideoReaderGetKeyIndices(self._handle).asnumpy().tolist()
        return self._key_indices

    def get_avg_fps(self):
        """Get average FPS(frame per second).

        Returns
        -------
        float
            Average FPS.

        """
        if self._avg_fps is None:
            self._avg_fps = _CAPI_VideoReaderGetAverageFPS(self._handle)
        return self._avg_fps

    def seek(self, pos):
        """Fast seek to frame position, this does not guarantee accurate position.
        To obtain accurate seeking, see `accurate_seek`.

        Parameters
        ----------
        pos : integer
            Non negative seeking position.

        """
        assert self._handle is not None
        assert pos >= 0 and pos < self._num_frame
        success = _CAPI_VideoReaderSeek(self._handle, pos)
        if not success:
            raise RuntimeError("Failed to seek to frame {}".format(pos))

    def seek_accurate(self, pos):
        """Accurately seek to frame position, this is slower than `seek`
        but guarantees accurate position.

        Parameters
        ----------
        pos : integer
            Non negative seeking position.

        """
        assert self._handle is not None
        assert pos >= 0 and pos < self._num_frame
        success = _CAPI_VideoReaderSeekAccurate(self._handle, pos)
        if not success:
            raise RuntimeError("Failed to seek_accurate to frame {}".format(pos))

    def skip_frames(self, num=1):
        """Skip reading multiple frames. Skipped frames will still be decoded
        (required by following frames) but it can save image resize/copy operations.


        Parameters
        ----------
        num : int, default is 1
            The number of frames to be skipped.

        """
        assert self._handle is not None
        assert num > 0
        _CAPI_VideoReaderSkipFrames(self._handle, num)

_init_api("decord.video_reader")
