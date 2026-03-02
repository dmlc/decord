"""Tests for decord framework bridge integration."""
import numpy as np
import pytest

import decord
from decord import VideoReader, cpu
from decord.bridge import set_bridge, reset_bridge

from conftest import BBB_PATH, CTX


def _torch_available():
    try:
        import torch
        return True
    except ImportError:
        return False


# ---------------------------------------------------------------------------
# Native bridge (default)
# ---------------------------------------------------------------------------

class TestNativeBridge:
    def test_default_returns_decord_ndarray(self):
        vr = VideoReader(BBB_PATH, ctx=CTX)
        frame = vr[0]
        assert isinstance(frame, decord.nd.NDArray)

    def test_ndarray_to_numpy(self):
        vr = VideoReader(BBB_PATH, ctx=CTX)
        frame = vr[0]
        arr = frame.asnumpy()
        assert isinstance(arr, np.ndarray)
        assert arr.dtype == np.uint8

    def test_ndarray_shape(self):
        vr = VideoReader(BBB_PATH, ctx=CTX)
        frame = vr[0]
        assert frame.shape == (360, 640, 3)


# ---------------------------------------------------------------------------
# PyTorch bridge
# ---------------------------------------------------------------------------

class TestTorchBridge:
    @pytest.fixture(autouse=True)
    def _reset_bridge(self):
        yield
        reset_bridge()

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not installed"
    )
    def test_torch_bridge_returns_tensor(self):
        import torch
        vr = VideoReader(BBB_PATH, ctx=CTX)
        set_bridge('torch')
        frame = vr[0]
        assert isinstance(frame, torch.Tensor)

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not installed"
    )
    def test_torch_bridge_shape(self):
        import torch
        vr = VideoReader(BBB_PATH, ctx=CTX)
        set_bridge('torch')
        frame = vr[0]
        assert frame.shape == (360, 640, 3)

    @pytest.mark.skipif(
        not _torch_available(),
        reason="PyTorch not installed"
    )
    def test_torch_bridge_context_manager(self):
        import torch
        from decord.bridge import use_torch
        vr = VideoReader(BBB_PATH, ctx=CTX)
        with use_torch():
            frame = vr[0]
            assert isinstance(frame, torch.Tensor)
        # After context, should be back to native
        frame2 = vr[0]
        assert isinstance(frame2, decord.nd.NDArray)
