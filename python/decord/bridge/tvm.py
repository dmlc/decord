"""DECORD TVM bridge"""
from __future__ import absolute_import

from .._ffi._ctypes.ndarray import _from_dlpack
from .utils import try_import

def try_import_tvm():
    """Try import tvm at runtime.

    Returns
    -------
    tvm module if found. Raise ImportError otherwise
    """
    msg = "tvm is required, for installation guide, please checkout:\n \
        https://tvm.apache.org/docs/install/index.html"
    return try_import('tvm', msg)

def to_tvm(decord_arr):
    """from decord to tvm, no copy"""
    tvm = try_import_tvm()
    return tvm.nd.from_dlpack(decord_arr.to_dlpack())

def from_tvm(tvm_arr):
    """from tvm to decord, no copy"""
    return _from_dlpack(tvm_arr.to_dlpack())
