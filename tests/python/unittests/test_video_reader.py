import os
import random
from decord import VideoReader
from decord.base import DECORDError

def _get_default_test_video():
    return VideoReader(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples', 'flipping_a_pancake.mkv')))

def _get_corrupted_test_video():
    return VideoReader(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'test_data', 'corrupted.mp4')))

def test_video_reader_len():
    vr = _get_default_test_video()
    assert len(vr) == 311

def test_video_reader_read_sequential():
    vr = _get_default_test_video()
    for i in range(len(vr)):
        frame = vr[i]

def test_video_reader_read_slice():
    vr = _get_default_test_video()
    frames = vr[:]
    assert frames.shape[0] == len(vr) 
    
    vr = _get_default_test_video()
    frames = vr[:10]
    assert frames.shape[0] == 10

def test_video_reader_read_random():
    vr = _get_default_test_video()
    lst = list(range(len(vr)))
    random.shuffle(lst)
    num = min(len(lst), 10)
    rand_lst = lst[:num]
    for i in rand_lst:
        frame = vr[i]

def test_video_get_batch():
    vr = _get_default_test_video()
    lst = list(range(len(vr)))
    random.shuffle(lst)
    num = min(len(lst), 10)
    rand_lst = lst[:num]
    frames = vr.get_batch(rand_lst)

def test_video_corrupted_get_batch():
    from nose.tools import assert_raises
    vr = _get_corrupted_test_video()
    assert_raises(DECORDError, vr.get_batch, range(40))

if __name__ == '__main__':
    import nose
    nose.runmodule()
