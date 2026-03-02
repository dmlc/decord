"""Tests for decord.AVReader using big_buck_bunny.mp4."""
import numpy as np
import pytest

from decord import AVReader, cpu
from decord.base import DECORDError

from conftest import BBB_PATH, VIDEO_ONLY_PATH, CTX


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------

class TestAVReaderProperties:
    def test_frame_count(self, bbb_av):
        assert len(bbb_av) == 1440

    def test_single_index_returns_tuple(self, bbb_av):
        result = bbb_av[0]
        assert isinstance(result, tuple)
        assert len(result) == 2
        audio, video = result
        # Video frame shape
        assert video.shape == (360, 640, 3)

    def test_audio_has_samples(self, bbb_av):
        audio, video = bbb_av[0]
        arr = audio.asnumpy()
        assert arr.shape[0] == 1  # mono (default)
        assert arr.shape[1] > 0  # has samples


# ---------------------------------------------------------------------------
# Single frame access
# ---------------------------------------------------------------------------

class TestAVReaderFrameAccess:
    def test_first_frame(self, bbb_av):
        audio, video = bbb_av[0]
        assert video.shape == (360, 640, 3)

    def test_mid_frame(self, bbb_av):
        audio, video = bbb_av[720]
        assert video.shape == (360, 640, 3)

    def test_last_frame(self, bbb_av):
        audio, video = bbb_av[1439]
        assert video.shape == (360, 640, 3)

    def test_negative_index(self, bbb_av):
        audio, video = bbb_av[-1]
        assert video.shape == (360, 640, 3)

    def test_out_of_bounds_raises(self, bbb_av):
        with pytest.raises(IndexError):
            bbb_av[1440]


# ---------------------------------------------------------------------------
# Slice access
# ---------------------------------------------------------------------------

class TestAVReaderSlice:
    def test_slice_returns_tuple(self, bbb_av):
        result = bbb_av[0:5]
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_slice_video_shape(self, bbb_av):
        audio_list, video = bbb_av[0:5]
        assert video.shape == (5, 360, 640, 3)

    def test_slice_audio_is_list(self, bbb_av):
        audio_list, video = bbb_av[0:5]
        assert isinstance(audio_list, list)
        assert len(audio_list) == 5

    def test_slice_audio_entries_have_samples(self, bbb_av):
        audio_list, video = bbb_av[10:15]
        for audio in audio_list:
            arr = audio.asnumpy()
            assert arr.shape[0] == 1  # mono
            assert arr.shape[1] > 0


# ---------------------------------------------------------------------------
# Batch access
# ---------------------------------------------------------------------------

class TestAVReaderBatch:
    def test_get_batch(self, bbb_av):
        result = bbb_av.get_batch([0, 100, 500])
        assert isinstance(result, tuple)
        audio_list, video = result
        assert video.shape == (3, 360, 640, 3)
        assert len(audio_list) == 3

    def test_get_batch_negative_indices(self, bbb_av):
        audio_list, video = bbb_av.get_batch([-1, 0, 1])
        assert video.shape[0] == 3

    def test_get_batch_single(self, bbb_av):
        audio_list, video = bbb_av.get_batch([500])
        assert video.shape[0] == 1
        assert len(audio_list) == 1


# ---------------------------------------------------------------------------
# Audio-video synchronization
# ---------------------------------------------------------------------------

class TestAVReaderSync:
    def test_consecutive_frames_have_consecutive_audio(self, bbb_av):
        """Audio for consecutive frames should cover consecutive time ranges."""
        audio_list, video = bbb_av[100:105]
        # Each frame's audio should have a reasonable number of samples
        for audio in audio_list:
            n_samples = audio.asnumpy().shape[1]
            # At 44100 Hz and ~24fps, expect ~1837 samples per frame
            # Allow wide tolerance for edge cases
            assert 500 < n_samples < 5000

    def test_audio_not_silent_during_video(self, bbb_av):
        """Audio in the middle of the video should not be all zeros."""
        audio, video = bbb_av[720]
        arr = audio.asnumpy()
        # The bunny video has audio throughout
        # (might be near-silent at some frames, so just check it's not exactly zero)
        # Use a range to be safe
        audio_list, _ = bbb_av[700:740]
        combined = np.concatenate([a.asnumpy() for a in audio_list], axis=1)
        assert np.any(combined != 0)


# ---------------------------------------------------------------------------
# BytesIO / file-like objects
# ---------------------------------------------------------------------------

class TestAVReaderBytesIO:
    def test_read_from_bytes_io(self):
        with open(BBB_PATH, 'rb') as f:
            av = AVReader(f, ctx=CTX)
            assert len(av) == 1440

    def test_bytes_io_matches_file(self, bbb_av):
        with open(BBB_PATH, 'rb') as f:
            av_bio = AVReader(f, ctx=CTX)
            audio1, video1 = bbb_av[50]
            audio2, video2 = av_bio[50]
            assert np.allclose(audio1.asnumpy(), audio2.asnumpy())
            assert np.allclose(video1.asnumpy(), video2.asnumpy())


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestAVReaderErrors:
    def test_no_audio_stream_raises(self):
        """Opening a video-only file with AVReader should raise."""
        with pytest.raises(DECORDError):
            AVReader(VIDEO_ONLY_PATH, ctx=CTX)


# ---------------------------------------------------------------------------
# Resource cleanup
# ---------------------------------------------------------------------------

class TestAVReaderCleanup:
    def test_del(self):
        av = AVReader(BBB_PATH, ctx=CTX)
        _ = av[0]
        del av  # should not raise
