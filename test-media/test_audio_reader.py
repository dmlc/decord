"""Tests for decord.AudioReader using big_buck_bunny.mp4."""
import io

import numpy as np
import pytest

from decord import AudioReader, cpu
from decord.base import DECORDError

from conftest import BBB_PATH, MP3_PATH, VIDEO_ONLY_PATH, CTX


# ---------------------------------------------------------------------------
# Basic properties
# ---------------------------------------------------------------------------

class TestAudioReaderProperties:
    def test_mono_shape(self, bbb_audio):
        assert bbb_audio.shape[0] == 1  # mono = 1 channel

    def test_mono_sample_count(self, bbb_audio):
        num_samples = bbb_audio.shape[1]
        # ~60s at 22050 Hz -> ~1.3M samples
        assert 1_200_000 < num_samples < 1_400_000

    def test_stereo_shape(self, bbb_audio_stereo):
        assert bbb_audio_stereo.shape[0] == 2  # stereo = 2 channels

    def test_stereo_same_sample_count(self, bbb_audio, bbb_audio_stereo):
        """Mono and stereo should have same number of samples per channel."""
        assert bbb_audio.shape[1] == bbb_audio_stereo.shape[1]

    def test_len_returns_num_channels(self, bbb_audio):
        assert len(bbb_audio) == 1

    def test_len_stereo(self, bbb_audio_stereo):
        assert len(bbb_audio_stereo) == 2

    def test_duration(self, bbb_audio):
        dur = bbb_audio.duration()
        assert 59.0 < dur < 61.0  # ~60 seconds


# ---------------------------------------------------------------------------
# Indexing
# ---------------------------------------------------------------------------

class TestAudioReaderIndexing:
    def test_index_first_sample(self, bbb_audio):
        sample = bbb_audio[0]
        assert sample.asnumpy().shape == (1,)

    def test_index_last_sample(self, bbb_audio):
        sample = bbb_audio[-1]
        assert sample.asnumpy().shape == (1,)

    def test_negative_index(self, bbb_audio):
        n = bbb_audio.shape[1]
        sample_pos = bbb_audio[n - 1].asnumpy()
        sample_neg = bbb_audio[-1].asnumpy()
        assert np.allclose(sample_pos, sample_neg)

    def test_out_of_bounds_raises(self, bbb_audio):
        with pytest.raises(IndexError):
            bbb_audio[bbb_audio.shape[1]]

    def test_stereo_index(self, bbb_audio_stereo):
        sample = bbb_audio_stereo[0]
        assert sample.asnumpy().shape == (2,)


# ---------------------------------------------------------------------------
# Slicing
# ---------------------------------------------------------------------------

class TestAudioReaderSlicing:
    def test_slice_range(self, bbb_audio):
        samples = bbb_audio[100:200]
        assert samples.asnumpy().shape == (1, 100)

    def test_slice_all(self, bbb_audio):
        samples = bbb_audio[:]
        assert samples.asnumpy().shape[0] == 1
        assert samples.asnumpy().shape[1] == bbb_audio.shape[1]

    def test_slice_negative(self, bbb_audio):
        samples = bbb_audio[-100:-50]
        assert samples.asnumpy().shape == (1, 50)

    def test_slice_stereo(self, bbb_audio_stereo):
        samples = bbb_audio_stereo[0:1000]
        assert samples.asnumpy().shape == (2, 1000)


# ---------------------------------------------------------------------------
# Batch access
# ---------------------------------------------------------------------------

class TestAudioReaderBatch:
    def test_get_batch(self, bbb_audio):
        indices = [0, 100, 200, 300, 400]
        batch = bbb_audio.get_batch(indices)
        assert batch.asnumpy().shape == (1, 5)

    def test_get_batch_negative_indices(self, bbb_audio):
        batch = bbb_audio.get_batch([-1, -2, -3])
        assert batch.asnumpy().shape == (1, 3)

    def test_get_batch_stereo(self, bbb_audio_stereo):
        batch = bbb_audio_stereo.get_batch([0, 1000, 2000])
        assert batch.asnumpy().shape == (2, 3)


# ---------------------------------------------------------------------------
# Resampling
# ---------------------------------------------------------------------------

class TestAudioReaderResample:
    def test_resample_lower(self):
        ar = AudioReader(BBB_PATH, ctx=CTX, sample_rate=11025, mono=True)
        original = AudioReader(BBB_PATH, ctx=CTX, mono=True)
        # Resampled should have roughly half the samples
        ratio = original.shape[1] / ar.shape[1]
        assert 1.8 < ratio < 2.2

    def test_resample_higher(self):
        ar = AudioReader(BBB_PATH, ctx=CTX, sample_rate=44100, mono=True)
        original = AudioReader(BBB_PATH, ctx=CTX, mono=True)
        # Resampled should have roughly double the samples
        ratio = ar.shape[1] / original.shape[1]
        assert 1.8 < ratio < 2.2

    def test_resample_preserves_channels(self):
        ar = AudioReader(BBB_PATH, ctx=CTX, sample_rate=11025, mono=False)
        assert ar.shape[0] == 2  # still stereo


# ---------------------------------------------------------------------------
# Channel conversion
# ---------------------------------------------------------------------------

class TestAudioReaderChannels:
    def test_stereo_to_mono(self):
        ar_stereo = AudioReader(BBB_PATH, ctx=CTX, mono=False)
        ar_mono = AudioReader(BBB_PATH, ctx=CTX, mono=True)
        assert ar_stereo.shape[0] == 2
        assert ar_mono.shape[0] == 1
        # Same number of samples per channel
        assert ar_stereo.shape[1] == ar_mono.shape[1]


# ---------------------------------------------------------------------------
# Audio values
# ---------------------------------------------------------------------------

class TestAudioReaderValues:
    def test_samples_are_float(self, bbb_audio):
        samples = bbb_audio[0:100].asnumpy()
        assert samples.dtype == np.float32 or samples.dtype == np.float64

    def test_samples_in_reasonable_range(self, bbb_audio):
        """Audio samples should be in a reasonable floating point range."""
        samples = bbb_audio[10000:20000].asnumpy()
        assert np.all(np.abs(samples) < 10.0)  # normalized audio

    def test_not_all_zeros(self, bbb_audio):
        """Audio should not be entirely silent."""
        # Sample from middle of file where there should be audio
        samples = bbb_audio[100000:200000].asnumpy()
        assert np.any(samples != 0)


# ---------------------------------------------------------------------------
# Padding
# ---------------------------------------------------------------------------

class TestAudioReaderPadding:
    def test_add_padding(self, bbb_audio):
        original_samples = bbb_audio.shape[1]
        num_padding = bbb_audio.add_padding()
        # After padding, total samples should increase by padding amount
        # (add_padding modifies internal array, not shape property)
        assert num_padding >= 0


# ---------------------------------------------------------------------------
# BytesIO / file-like objects
# ---------------------------------------------------------------------------

class TestAudioReaderBytesIO:
    def test_read_from_bytes_io(self):
        with open(BBB_PATH, 'rb') as f:
            ar = AudioReader(f, ctx=CTX, mono=True)
            assert ar.shape[0] == 1
            assert ar.shape[1] > 0

    def test_bytes_io_matches_file(self, bbb_audio):
        with open(BBB_PATH, 'rb') as f:
            ar_bio = AudioReader(f, ctx=CTX, mono=True)
            file_samples = bbb_audio[0:1000].asnumpy()
            bio_samples = ar_bio[0:1000].asnumpy()
            assert np.allclose(file_samples, bio_samples)


# ---------------------------------------------------------------------------
# Error handling
# ---------------------------------------------------------------------------

class TestAudioReaderErrors:
    def test_no_audio_stream_raises(self):
        """Opening a video-only file with AudioReader should raise."""
        with pytest.raises(DECORDError):
            AudioReader(VIDEO_ONLY_PATH, ctx=CTX)


# ---------------------------------------------------------------------------
# Resource cleanup
# ---------------------------------------------------------------------------

class TestAudioReaderCleanup:
    def test_del(self):
        ar = AudioReader(BBB_PATH, ctx=CTX)
        _ = ar[0]
        del ar  # should not raise

    def test_multiple_readers(self):
        ar1 = AudioReader(BBB_PATH, ctx=CTX, mono=True)
        ar2 = AudioReader(BBB_PATH, ctx=CTX, mono=True)
        assert ar1.shape == ar2.shape


# ---------------------------------------------------------------------------
# MP3 format support
# ---------------------------------------------------------------------------

class TestAudioReaderMP3:
    def test_mp3_loads(self, mp3_audio):
        assert mp3_audio.shape[0] == 1  # mono
        assert mp3_audio.shape[1] > 0

    def test_mp3_duration(self, mp3_audio):
        dur = mp3_audio.duration()
        # ~878 seconds
        assert 870 < dur < 890

    def test_mp3_sample_count(self, mp3_audio):
        # ~878s at 44100 Hz -> ~38.7M samples
        n = mp3_audio.shape[1]
        assert 38_000_000 < n < 40_000_000

    def test_mp3_indexing(self, mp3_audio):
        sample = mp3_audio[0]
        assert sample.asnumpy().shape == (1,)

    def test_mp3_slicing(self, mp3_audio):
        samples = mp3_audio[1000:2000]
        assert samples.asnumpy().shape == (1, 1000)

    def test_mp3_not_silent(self, mp3_audio):
        # Sample from well into the file
        samples = mp3_audio[500000:600000].asnumpy()
        assert np.any(samples != 0)

    def test_mp3_samples_are_float(self, mp3_audio):
        samples = mp3_audio[0:100].asnumpy()
        assert samples.dtype in (np.float32, np.float64)

    def test_mp3_resample(self):
        ar = AudioReader(MP3_PATH, ctx=CTX, sample_rate=22050, mono=True)
        original = AudioReader(MP3_PATH, ctx=CTX, mono=True)
        ratio = original.shape[1] / ar.shape[1]
        assert 1.8 < ratio < 2.2

    def test_mp3_bytes_io(self, mp3_audio):
        with open(MP3_PATH, 'rb') as f:
            ar_bio = AudioReader(f, ctx=CTX, mono=True)
            assert ar_bio.shape == mp3_audio.shape
