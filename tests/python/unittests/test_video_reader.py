import os
import random
import numpy as np
from decord import VideoReader, cpu, gpu
from decord.base import DECORDError

CTX = cpu(0)

def _get_default_test_video(ctx=CTX):
    return VideoReader(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples', 'flipping_a_pancake.mkv')), ctx=ctx)

def _get_corrupted_test_video(ctx=CTX):
    return VideoReader(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'test_data', 'corrupted.mp4')), ctx=ctx)

def _get_rotated_test_video(rot, height=-1, width=-1, ctx=CTX):
    return VideoReader(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'test_data', f'video_{rot}.mov')), height=height, width=width, ctx=ctx)

def _get_unordered_test_video(ctx=CTX):
    # video with frames not ordered by pts
    return VideoReader(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'test_data', 'unordered.mov')), ctx=ctx)

def test_video_reader_len():
    vr = _get_default_test_video()
    assert len(vr) == 310

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
    vr = _get_corrupted_test_video(ctx=cpu(0))
    assert_raises(DECORDError, vr.get_batch, range(40))

def test_rotated_video():
    # Input videos are all h=320 w=568 in metadata, but
    # rotation should be applied to recover correctly
    # displayed image (from rotation metadata).
    for rot in [0, 180]:
        # shot in landscape; correct video orientation has
        # same shape as "original" frame
        vr = _get_rotated_test_video(rot, ctx=cpu(0))
        assert vr[0].shape == (320, 568, 3)
        assert vr[:].shape == (3, 320, 568, 3)
    for rot in [90, 270]:
        # shot in portrait mode; correct video orientation has
        # swapped width and height (height>>width)
        vr = _get_rotated_test_video(rot, ctx=cpu(0))
        assert vr[0].shape == (568, 320, 3), vr[0].shape
        assert vr[:].shape == (3, 568, 320, 3)
        # resize is applied in target shape
        vr = _get_rotated_test_video(rot, height=300, width=200, ctx=cpu(0))
        assert vr[0].shape == (300, 200, 3), vr[0].shape

def test_frame_timestamps():
    vr = _get_default_test_video()
    frame_ts = vr.get_frame_timestamp(range(5))
    assert np.allclose(frame_ts[:,0], [0.0, 0.033, 0.067, 0.1, 0.133])

    vr = _get_unordered_test_video()
    '''ffprobe output:
        pts_time=0.000000 dts_time=-0.062500
        pts_time=0.093750 dts_time=-0.031250
        pts_time=0.031250 dts_time=0.000000
        pts_time=0.062500 dts_time=0.031250
    '''
    frame_ts = vr.get_frame_timestamp(range(4))
    assert np.allclose(frame_ts[:,0], [0.0, 0.03125, 0.0625, 0.09375]), frame_ts[:,0]

def test_bytes_io():
    fn = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples', 'flipping_a_pancake.mkv'))
    with open(fn, 'rb') as f:
        vr = VideoReader(f)
        assert len(vr) == 310
        vr2 = _get_default_test_video()
        assert np.allclose(vr[10].asnumpy(), vr2[10].asnumpy())
        

if __name__ == '__main__':
    import nose
    nose.runmodule()
