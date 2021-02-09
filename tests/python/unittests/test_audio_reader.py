import os
import numpy as np
from decord import AudioReader, cpu, gpu
from decord.base import DECORDError

CTX = cpu(0)

def get_single_channel_reader():
    return AudioReader(os.path.join(os.path.dirname(__file__), '..', '..', 'cpp', 'audio', 'count_down.mov'), CTX)

def get_double_channels_reader():
    return AudioReader(os.path.join(os.path.dirname(__file__), '..', '..', 'cpp', 'audio', 'sample-mov-file.mov'), CTX, mono=False)

def get_resampled_reader():
    return AudioReader(os.path.join(os.path.dirname(__file__), '..', '..', 'cpp', 'audio', 'count_down.mov'), CTX, 4410)

def get_channel_change_reader():
    return AudioReader(os.path.join(os.path.dirname(__file__), '..', '..', 'cpp', 'audio', 'sample-mov-file.mov'), CTX)

def test_single_channel_audio_reader():
    ar = get_single_channel_reader()
    assert ar.shape() == (1, 482240)

def test_double_channels_audio_reader():
    ar = get_double_channels_reader()
    assert ar.shape() == (2, 5555200)

def test_no_audio_stream():
    from nose.tools import assert_raises
    assert_raises(DECORDError, AudioReader, os.path.join(os.path.dirname(__file__), '..', '..', 'test_data', 'video_0.mov'), CTX)

def test_bytes_io():
    fn = os.path.join(os.path.dirname(__file__), '..', '..', 'cpp', 'audio', 'count_down.mov')
    with open(fn, 'rb') as f:
        ar = AudioReader(f)
        assert ar.shape() == (1, 482240)
        ar2 = get_single_channel_reader()
        assert np.allclose(ar[10].asnumpy(), ar2[10].asnumpy())

def test_resample():
    ar = get_resampled_reader()
    assert ar.shape() == (1, 48224)

def test_channel_change():
    ar = get_channel_change_reader()
    assert ar.shape() == (1, 5555200)

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
    num_channels = ar.shape()[0]
    num_padding = ar.add_padding()
    assert np.array_equal(ar[:num_padding].asnumpy(), np.zeros((num_channels, num_padding)))

def test_free():
    ar = get_single_channel_reader()
    del ar

if __name__ == '__main__':
    import nose
    nose.runmodule()
