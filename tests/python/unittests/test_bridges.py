import os
import numpy as np
import decord
from decord import VideoReader
from decord.bridge import *

def _get_default_test_video():
    return VideoReader(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..', 'examples', 'flipping_a_pancake.mkv')))

def test_mxnet_bridge():
    try:
        from decord.bridge.mxnet import try_import_mxnet
        mxnet = try_import_mxnet()
        vr = _get_default_test_video()
        with use_mxnet():
            frame = vr[0]
            assert isinstance(frame, mxnet.nd.NDArray)
            native_frame = bridge_in(frame)
        assert isinstance(native_frame, decord.nd.NDArray), type(native_frame)
    except ImportError:
        print('Skip test mxnet bridge as mxnet is not found')

def test_torch_bridge():
    try:
        from decord.bridge.torchdl import try_import_torch
        import torch
        vr = _get_default_test_video()
        with use_torch():
            frame = vr[0]
            assert isinstance(frame, torch.Tensor), type(frame)
            native_frame = bridge_in(frame)
        assert isinstance(native_frame, decord.nd.NDArray), type(native_frame)
    except ImportError:
        print('Skip test torchdl bridge as torch is not found')

def test_tf_bridge():
    try:
        from decord.bridge.tf import try_import_tfdl
        import tensorflow as tf
        vr = _get_default_test_video()
        with use_tensorflow():
            frame = vr[0]
            assert isinstance(frame, tf.Tensor), type(frame)
            native_frame = bridge_in(frame)
        assert isinstance(native_frame, decord.nd.NDArray), type(native_frame)
    except ImportError:
        print('Skip test tensorflow bridge as tf is not found')

def test_tvm_bridge():
    try:
        from decord.bridge.tvm import try_import_tvm
        tvm = try_import_tvm()
        vr = _get_default_test_video()
        with use_tvm():
            frame = vr[0]
            assert isinstance(frame, tvm.nd.NDArray)
            native_frame = bridge_in(frame)
        assert isinstance(native_frame, decord.nd.NDArray), type(native_frame)
    except ImportError:
        print('Skip test tvm bridge as tvm is not found')

def test_threaded_bridge():
    # issue #85
    from decord import cpu, gpu
    from multiprocessing.dummy import Pool as ThreadPool

    video_paths = [
      os.path.expanduser('~/Dev/decord/examples/flipping_a_pancake.mkv'), #list of paths to video
      ]

    def process_path(path):
        vr = VideoReader(path, ctx=cpu(0))

        for i in range(len(vr)):
            frame = vr[i]

    pool = ThreadPool(1)
    pool.map(process_path, video_paths)

if __name__ == '__main__':
    import nose
    nose.runmodule()
