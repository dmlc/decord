import os
import numpy as np
from decord import AVReader, cpu, gpu
from decord.base import DECORDError

CTX = cpu(0)

def get_normal_av_reader():
    return AVReader('/Users/weisy/Developer/yinweisu/decord/tests/cpp/audio/count_down.mov', CTX)

def test_normal_av_reader():
    av = get_normal_av_reader()
    assert len(av) == 328

def test_bytes_io():
    fn = os.path.join(os.path.dirname(__file__), '..', '..', 'cpp', 'audio', 'count_down.mov')
    with open(fn, 'rb') as f:
        av = AVReader(f)
        assert len(av) == 328
        av2 = get_normal_av_reader()
        audio, video = av[10]
        audio2, video2 = av2[10]
        assert np.allclose(audio.asnumpy(), audio2.asnumpy())
        assert np.allclose(video.asnumpy(), video2.asnumpy())

def test_no_audio_stream():
    from nose.tools import assert_raises
    assert_raises(DECORDError, AVReader, os.path.join(os.path.dirname(__file__), '..', '..', 'test_data', 'video_0.mov'), CTX)

def test_index():
    av = get_normal_av_reader()
    audio, video = av[0]

def test_indices():
    av = get_normal_av_reader()
    audio, video = av[:]

def test_get_batch():
    av = get_normal_av_reader()
    av.get_batch([-1,0,1,2,3])

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
