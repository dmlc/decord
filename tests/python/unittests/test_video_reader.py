import os
import random
from decord import VideoReader

def _get_default_test_video():
    return VideoReader(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples', 'flipping_a_pancake.mkv')))

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

if __name__ == '__main__':
    import nose
    nose.runmodule()
