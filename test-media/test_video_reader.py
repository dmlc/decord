"""Tests for decord.VideoReader using big_buck_bunny.mp4."""
import io
import random

import numpy as np
import pytest

from decord import VideoReader, cpu
from decord.base import DECORDError

from conftest import BBB_PATH, PANCAKE_PATH, CORRUPTED_PATH, ROTATION_VIDEOS, UNORDERED_PATH, CTX


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------

class TestVideoReaderProperties:
    def test_frame_count(self, bbb_video):
        assert len(bbb_video) == 1440

    def test_frame_shape(self, bbb_video):
        frame = bbb_video[0]
        assert frame.shape == (360, 640, 3)

    def test_avg_fps(self, bbb_video):
        fps = bbb_video.get_avg_fps()
        assert 23.0 < fps < 25.0  # ~23.96 fps

    def test_key_indices_returns_list(self, bbb_video):
        keys = bbb_video.get_key_indices()
        assert isinstance(keys, list)
        assert len(keys) > 0
        # First keyframe should be frame 0
        assert keys[0] == 0
        # All indices should be valid
        for k in keys:
            assert 0 <= k < len(bbb_video)

    def test_key_indices_are_sorted(self, bbb_video):
        keys = bbb_video.get_key_indices()
        assert keys == sorted(keys)

    def test_pancake_frame_count(self, pancake_video):
        assert len(pancake_video) == 310


# ---------------------------------------------------------------------------
# Single frame access
# ---------------------------------------------------------------------------

class TestVideoReaderFrameAccess:
    def test_first_frame(self, bbb_video):
        frame = bbb_video[0]
        arr = frame.asnumpy()
        assert arr.dtype == np.uint8
        assert arr.shape == (360, 640, 3)

    def test_last_frame(self, bbb_video):
        frame = bbb_video[len(bbb_video) - 1]
        assert frame.shape == (360, 640, 3)

    def test_negative_index(self, bbb_video):
        frame_pos = bbb_video[len(bbb_video) - 1]
        frame_neg = bbb_video[-1]
        assert np.array_equal(frame_pos.asnumpy(), frame_neg.asnumpy())

    def test_out_of_bounds_raises(self, bbb_video):
        with pytest.raises(IndexError):
            bbb_video[1440]
        with pytest.raises(IndexError):
            bbb_video[-1441]

    def test_pixel_values_in_range(self, bbb_video):
        frame = bbb_video[100].asnumpy()
        assert frame.min() >= 0
        assert frame.max() <= 255

    def test_different_frames_differ(self, bbb_video):
        """Frames far apart should have different content."""
        frame_a = bbb_video[0].asnumpy()
        frame_b = bbb_video[720].asnumpy()  # ~halfway
        assert not np.array_equal(frame_a, frame_b)


# ---------------------------------------------------------------------------
# Sequential reading
# ---------------------------------------------------------------------------

class TestVideoReaderSequential:
    def test_read_first_30_frames(self, bbb_video):
        for i in range(30):
            frame = bbb_video[i]
            assert frame.shape == (360, 640, 3)

    def test_sequential_consistency(self, bbb_video):
        """Reading frame N twice should give same result."""
        frame1 = bbb_video[50].asnumpy()
        frame2 = bbb_video[50].asnumpy()
        assert np.array_equal(frame1, frame2)


# ---------------------------------------------------------------------------
# Slice access
# ---------------------------------------------------------------------------

class TestVideoReaderSlice:
    def test_slice_all(self, pancake_video):
        """Slice all frames from the smaller pancake video."""
        frames = pancake_video[:]
        assert frames.shape[0] == 310

    def test_slice_range(self, bbb_video):
        frames = bbb_video[10:20]
        assert frames.shape == (10, 360, 640, 3)

    def test_slice_with_step(self, bbb_video):
        frames = bbb_video[0:100:10]
        assert frames.shape[0] == 10

    def test_slice_from_start(self, bbb_video):
        frames = bbb_video[:5]
        assert frames.shape[0] == 5

    def test_slice_to_end(self, bbb_video):
        frames = bbb_video[1435:]
        assert frames.shape[0] == 5

    def test_slice_negative(self, bbb_video):
        frames = bbb_video[-5:]
        assert frames.shape[0] == 5


# ---------------------------------------------------------------------------
# Batch access
# ---------------------------------------------------------------------------

class TestVideoReaderBatch:
    def test_get_batch_sequential(self, bbb_video):
        indices = list(range(10))
        frames = bbb_video.get_batch(indices)
        assert frames.shape == (10, 360, 640, 3)

    def test_get_batch_random(self, bbb_video):
        random.seed(42)
        indices = random.sample(range(1440), 20)
        frames = bbb_video.get_batch(indices)
        assert frames.shape == (20, 360, 640, 3)

    def test_get_batch_single(self, bbb_video):
        frames = bbb_video.get_batch([500])
        assert frames.shape == (1, 360, 640, 3)

    def test_get_batch_duplicates(self, bbb_video):
        """Duplicate indices should return duplicate frames."""
        frames = bbb_video.get_batch([100, 100, 100])
        assert frames.shape[0] == 3
        arr = frames.asnumpy()
        assert np.array_equal(arr[0], arr[1])
        assert np.array_equal(arr[1], arr[2])

    def test_get_batch_negative_indices(self, bbb_video):
        frames = bbb_video.get_batch([-1, -2, -3])
        assert frames.shape[0] == 3

    def test_batch_matches_individual(self, bbb_video):
        """Batch result should match individual frame reads."""
        indices = [0, 100, 500, 1000]
        batch = bbb_video.get_batch(indices).asnumpy()
        for i, idx in enumerate(indices):
            individual = bbb_video[idx].asnumpy()
            assert np.array_equal(batch[i], individual)


# ---------------------------------------------------------------------------
# Seeking
# ---------------------------------------------------------------------------

class TestVideoReaderSeeking:
    def test_seek(self, bbb_video):
        bbb_video.seek(100)
        frame = bbb_video.next()
        assert frame.shape == (360, 640, 3)

    def test_seek_accurate(self, bbb_video):
        bbb_video.seek_accurate(100)
        frame = bbb_video.next()
        assert frame.shape == (360, 640, 3)

    def test_seek_to_start(self, bbb_video):
        bbb_video.seek(0)
        frame = bbb_video.next()
        assert frame.shape == (360, 640, 3)

    def test_seek_to_near_end(self, bbb_video):
        bbb_video.seek_accurate(1439)
        frame = bbb_video.next()
        assert frame.shape == (360, 640, 3)

    def test_skip_frames(self, bbb_video):
        bbb_video.seek(0)
        bbb_video.skip_frames(5)
        frame = bbb_video.next()
        assert frame.shape == (360, 640, 3)


# ---------------------------------------------------------------------------
# Frame timestamps
# ---------------------------------------------------------------------------

class TestVideoReaderTimestamps:
    def test_timestamp_shape(self, bbb_video):
        ts = bbb_video.get_frame_timestamp(range(10))
        assert ts.shape == (10, 2)

    def test_first_frame_starts_at_zero(self, bbb_video):
        ts = bbb_video.get_frame_timestamp([0])
        assert ts[0, 0] == pytest.approx(0.0, abs=0.01)

    def test_timestamps_are_monotonic(self, bbb_video):
        ts = bbb_video.get_frame_timestamp(range(100))
        starts = ts[:, 0]
        assert all(starts[i] <= starts[i + 1] for i in range(len(starts) - 1))

    def test_timestamps_match_fps(self, bbb_video):
        """Frame interval should roughly match 1/fps."""
        ts = bbb_video.get_frame_timestamp(range(10))
        fps = bbb_video.get_avg_fps()
        expected_interval = 1.0 / fps
        for i in range(1, 10):
            actual_interval = ts[i, 0] - ts[i - 1, 0]
            assert actual_interval == pytest.approx(expected_interval, rel=0.1)

    def test_last_frame_timestamp_reasonable(self, bbb_video):
        ts = bbb_video.get_frame_timestamp([1439])
        # ~60 second video, last frame should be near 60s
        assert 58.0 < ts[0, 0] < 61.0


# ---------------------------------------------------------------------------
# Resize
# ---------------------------------------------------------------------------

class TestVideoReaderResize:
    def test_resize(self):
        vr = VideoReader(BBB_PATH, ctx=CTX, width=320, height=180)
        frame = vr[0]
        assert frame.shape == (180, 320, 3)

    def test_resize_width_only(self):
        vr = VideoReader(BBB_PATH, ctx=CTX, width=320)
        frame = vr[0]
        assert frame.shape[2] == 3
        assert frame.shape[1] == 320

    def test_resize_height_only(self):
        vr = VideoReader(BBB_PATH, ctx=CTX, height=180)
        frame = vr[0]
        assert frame.shape[0] == 180


# ---------------------------------------------------------------------------
# BytesIO / file-like objects
# ---------------------------------------------------------------------------

class TestVideoReaderBytesIO:
    def test_read_from_bytes_io(self):
        with open(BBB_PATH, 'rb') as f:
            vr = VideoReader(f, ctx=CTX)
            assert len(vr) == 1440

    def test_bytes_io_matches_file(self, bbb_video):
        with open(BBB_PATH, 'rb') as f:
            vr_bio = VideoReader(f, ctx=CTX)
            frame_file = bbb_video[50].asnumpy().astype('float')
            frame_bio = vr_bio[50].asnumpy().astype('float')
            assert np.mean(np.abs(frame_file - frame_bio)) < 2


# ---------------------------------------------------------------------------
# Rotation handling
# ---------------------------------------------------------------------------

class TestVideoReaderRotation:
    def test_landscape_no_rotation(self):
        vr = VideoReader(ROTATION_VIDEOS[0], ctx=CTX)
        assert vr[0].shape == (320, 568, 3)

    def test_landscape_180_rotation(self):
        vr = VideoReader(ROTATION_VIDEOS[180], ctx=CTX)
        assert vr[0].shape == (320, 568, 3)

    def test_portrait_90_rotation(self):
        vr = VideoReader(ROTATION_VIDEOS[90], ctx=CTX)
        assert vr[0].shape == (568, 320, 3)

    def test_portrait_270_rotation(self):
        vr = VideoReader(ROTATION_VIDEOS[270], ctx=CTX)
        assert vr[0].shape == (568, 320, 3)

    def test_rotated_with_resize(self):
        vr = VideoReader(ROTATION_VIDEOS[90], ctx=CTX, height=300, width=200)
        assert vr[0].shape == (300, 200, 3)


# ---------------------------------------------------------------------------
# Corrupted video
# ---------------------------------------------------------------------------

class TestVideoReaderCorrupted:
    def test_corrupted_batch_raises(self):
        vr = VideoReader(CORRUPTED_PATH, ctx=CTX)
        with pytest.raises(DECORDError):
            vr.get_batch(range(40))


# ---------------------------------------------------------------------------
# Unordered PTS
# ---------------------------------------------------------------------------

class TestVideoReaderUnorderedPTS:
    def test_unordered_timestamps_sorted(self):
        vr = VideoReader(UNORDERED_PATH, ctx=CTX)
        ts = vr.get_frame_timestamp(range(4))
        starts = ts[:, 0]
        assert all(starts[i] <= starts[i + 1] for i in range(len(starts) - 1))

    def test_unordered_timestamps_values(self):
        vr = VideoReader(UNORDERED_PATH, ctx=CTX)
        ts = vr.get_frame_timestamp(range(4))
        assert np.allclose(ts[:, 0], [0.0, 0.03125, 0.0625, 0.09375])


# ---------------------------------------------------------------------------
# Resource cleanup
# ---------------------------------------------------------------------------

class TestVideoReaderCleanup:
    def test_del(self):
        vr = VideoReader(BBB_PATH, ctx=CTX)
        _ = vr[0]
        del vr  # should not raise

    def test_multiple_readers(self):
        """Multiple simultaneous readers should not conflict."""
        vr1 = VideoReader(BBB_PATH, ctx=CTX)
        vr2 = VideoReader(BBB_PATH, ctx=CTX)
        assert len(vr1) == len(vr2)
        f1 = vr1[100].asnumpy()
        f2 = vr2[100].asnumpy()
        assert np.array_equal(f1, f2)
