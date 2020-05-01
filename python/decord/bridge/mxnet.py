"""DECORD MXNet bridge"""
from __future__ import absolute_import

from .._ffi._ctypes.ndarray import _from_dlpack
from .utils import try_import

def try_import_mxnet():
    """Try import mxnet at runtime.

    Returns
    -------
    mxnet module if found. Raise ImportError otherwise
    """
    msg = "mxnet is required, you can install by pip.\n \
        CPU: `pip install mxnet-mkl`, GPU: `pip install mxnet-cu100mkl`"
    return try_import('mxnet', msg)

def to_mxnet(decord_arr):
    """from decord to mxnet, no copy"""
    mx = try_import_mxnet()
    return mx.nd.from_dlpack(decord_arr.to_dlpack())

def from_mxnet(mxnet_arr):
    """from mxnet to decord, no copy"""
    return _from_dlpack(mxnet_arr.to_dlpack_for_read())
