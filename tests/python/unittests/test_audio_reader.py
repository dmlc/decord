import os
import numpy as np
from decord import AudioReader, cpu, gpu
from decord.base import DECORDError

CTX = cpu(0)

def get_single_channel_reader():
    return AudioReader(os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'audio', 'count_down.mov'), CTX)

def get_double_channels_reader():
    return AudioReader(os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'audio', 'sample-mov-file.mov'), CTX, mono=False)

def get_resampled_reader():
    return AudioReader(os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'audio', 'count_down.mov'), CTX, 4410)

def get_channel_change_reader():
    return AudioReader(os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'audio', 'sample-mov-file.mov'), CTX)

def test_single_channel_audio_reader():
    ar = get_single_channel_reader()
    assert ar.shape == (1, 394176)

def test_double_channels_audio_reader():
    ar = get_double_channels_reader()
    assert ar.shape == (2, 5555200)

"""def test_no_audio_stream():
    from nose.tools import assert_raises
    assert_raises(DECORDError, AudioReader, os.path.join(os.path.dirname(__file__), '..', '..', 'test_data', 'video_0.mov'), CTX)"""

def test_bytes_io():
    fn = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'audio', 'count_down.mov')
    with open(fn, 'rb') as f:
        ar = AudioReader(f)
        assert ar.shape == (1, 394176)
        ar2 = get_single_channel_reader()
        assert np.allclose(ar[10].asnumpy(), ar2[10].asnumpy())

def test_resample():
    ar = get_resampled_reader()
    assert ar.shape == (1, 39418)

def test_channel_change():
    ar = get_channel_change_reader()
    assert ar.shape == (1, 5555200)

def test_index():
    ar = get_double_channels_reader()
    ar[0]
    ar[-1]

def test_indices():
    ar = get_double_channels_reader()
    ar[:]
    ar[-20:-10]

def test_get_batch():
    ar = get_double_channels_reader()
    ar.get_batch([-1,0,1,2,3])

def test_get_info():
    ar = get_double_channels_reader()
    ar.get_info()

def test_add_padding():
    ar = get_single_channel_reader()
    num_channels = ar.shape[0]
    num_padding = ar.add_padding()
    assert np.array_equal(ar[:num_padding].asnumpy(), np.zeros((num_channels, num_padding)))

def test_free():
    ar = get_single_channel_reader()
    del ar


# --- Tests for issue #10: stereo (mono=False) stability ---

def test_stereo_all_finite():
    """mono=False must produce only finite values in every channel."""
    ar = get_double_channels_reader()
    arr = ar[:].asnumpy()
    assert np.isfinite(arr).all(), (
        f"non-finite count: {(~np.isfinite(arr)).sum()}, "
        f"ch0 max {np.nanmax(np.abs(arr[0]))}, ch1 max {np.nanmax(np.abs(arr[1]))}"
    )

def test_stereo_no_extreme_values():
    """Sample values should stay in a plausible range (not ~1e38 garbage)."""
    ar = get_double_channels_reader()
    arr = ar[:].asnumpy()
    assert np.abs(arr).max() <= 1.0 + 1e-3, f"abs max {np.abs(arr).max()} exceeds plausible audio range"

def test_stereo_deterministic():
    """Repeated decodes of the same file must produce identical output."""
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'audio', 'sample-mov-file.mov')
    a1 = AudioReader(path, CTX, mono=False)[:].asnumpy()
    a2 = AudioReader(path, CTX, mono=False)[:].asnumpy()
    assert a1.shape == a2.shape
    assert np.array_equal(a1, a2), f"max diff {np.abs(a1 - a2).max()}"

def test_stereo_channels_differ():
    """Left and right channels should not be identical (rules out silent second channel)."""
    ar = get_double_channels_reader()
    arr = ar[:].asnumpy()
    assert not np.array_equal(arr[0], arr[1])

def test_stereo_bytesio():
    """mono=False through BytesIO must match the file-path result."""
    import io
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'audio', 'sample-mov-file.mov')
    with open(path, 'rb') as f:
        data = f.read()
    ar_file = AudioReader(path, CTX, mono=False)
    ar_bytes = AudioReader(io.BytesIO(data), CTX, mono=False)
    a_file = ar_file[:].asnumpy()
    a_bytes = ar_bytes[:].asnumpy()
    assert a_file.shape == a_bytes.shape
    assert np.allclose(a_file, a_bytes, atol=1e-6), f"max diff {np.abs(a_file - a_bytes).max()}"

def test_stereo_resampled_all_finite():
    """Resampling stereo audio should also produce all-finite values."""
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'audio', 'sample-mov-file.mov')
    ar = AudioReader(path, CTX, sample_rate=16000, mono=False)
    arr = ar[:].asnumpy()
    assert arr.shape[0] == 2
    assert np.isfinite(arr).all(), f"non-finite count: {(~np.isfinite(arr)).sum()}"
    assert np.abs(arr).max() <= 1.0 + 1e-3

def test_av_reader_stereo():
    """AVReader with mono=False should produce finite audio for every frame."""
    from decord import AVReader
    path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'audio', 'sample-mov-file.mov')
    avr = AVReader(path, CTX, sample_rate=44100, mono=False)
    audio_samples, frames = avr[0]
    arr = audio_samples.asnumpy()
    assert arr.shape[0] == 2
    assert np.isfinite(arr).all()

    audio_list, frame_batch = avr.get_batch([0, 1, 2])
    for i, a in enumerate(audio_list):
        a_np = a.asnumpy()
        assert a_np.shape[0] == 2, f"frame {i}: expected 2 channels, got {a_np.shape[0]}"
        assert np.isfinite(a_np).all(), f"frame {i}: contains non-finite values"


if __name__ == '__main__':
    import nose
    nose.runmodule()