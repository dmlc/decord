"""Tests for decord NDArray and context functions."""
import numpy as np
import pytest

import decord
from decord import cpu, gpu
from decord import nd


class TestContext:
    def test_cpu_context(self):
        ctx = cpu(0)
        assert ctx.device_type == 1  # kDLCPU

    def test_cpu_default_id(self):
        ctx = cpu()
        assert ctx.device_id == 0

    def test_gpu_context(self):
        ctx = gpu(0)
        assert ctx.device_type == 2  # kDLGPU


class TestNDArray:
    def test_from_numpy(self):
        arr = np.random.rand(10, 10).astype(np.float32)
        nd_arr = nd.array(arr)
        assert nd_arr.shape == (10, 10)

    def test_to_numpy(self):
        arr = np.random.rand(5, 5).astype(np.float32)
        nd_arr = nd.array(arr)
        result = nd_arr.asnumpy()
        assert isinstance(result, np.ndarray)
        assert np.allclose(arr, result)

    def test_roundtrip_int(self):
        arr = np.array([1, 2, 3, 4, 5], dtype=np.int64)
        nd_arr = nd.array(arr)
        result = nd_arr.asnumpy()
        assert np.array_equal(arr, result)

    def test_shape_property(self):
        arr = np.zeros((3, 4, 5), dtype=np.float32)
        nd_arr = nd.array(arr)
        assert nd_arr.shape == (3, 4, 5)
