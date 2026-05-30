import os
import pytest
import numpy as np
from decord import AVReader, cpu, gpu
from decord.base import DECORDError

CTX = cpu(0)


# Correctly constructs the path relative to the current file
def get_normal_av_reader():
    # A common practice is to have a `tests/resources` directory.
    video_path = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'audio', 'count_down.mov')
    return AVReader(video_path, CTX)

def test_normal_av_reader():
    av = get_normal_av_reader()
    assert len(av) == 143

def test_bytes_io():
    fn = os.path.join(os.path.dirname(__file__), '..', '..', 'resources', 'audio', 'count_down.mov')
    with open(fn, 'rb') as f:
        av = AVReader(f)
        assert len(av) == 143
        av2 = get_normal_av_reader()
        audio, video = av[10]
        audio2, video2 = av2[10]
        assert np.allclose(audio.asnumpy(), audio2.asnumpy())
        assert np.allclose(video.asnumpy(), video2.asnumpy())

"""def test_no_audio_stream():
    from nose.tools import assert_raises
    assert_raises(DECORDError, AVReader, os.path.join(os.path.dirname(__file__), '..', '..', 'test_data', 'video_0.mov'), CTX)"""

def test_index():
    av = get_normal_av_reader()
    audio, video = av[0]

def test_indices():
    av = get_normal_av_reader()
    audio, video = av[:]

def test_get_batch():
    av = get_normal_av_reader()
    av.get_batch([-1,0,1,2,3])

@pytest.mark.skip(reason="Cannot test audio playback in a headless CI environment")
def test_sync():
    av = get_normal_av_reader()
    import simpleaudio
    audio = av[25:40][0]
    buffer = np.array([], dtype='float32')
    for samples in audio:
        buffer = np.append(buffer, samples.asnumpy())
    play = simpleaudio.play_buffer(buffer, 1, 4, 44100)
    play.wait_done()

if __name__ == '__main__':
    import nose
    nose.runmodule()