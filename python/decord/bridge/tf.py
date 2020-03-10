"""DECORD tensorflow bridge"""
from __future__ import absolute_import

from .._ffi._ctypes.ndarray import _from_dlpack

def try_import_tfdl():
    """Try to import tensorflow dlpack at runtime.

    Returns
    -------
    tensorflow dlpack module if found. Raise ImportError otherwise
    """
    try:
        return __import__('tensorflow.experimental.dlpack', fromlist=[''])
    except ImportError as e:
        raise ImportError("tensorflow >= 2.2.0 is required.")

def to_tensorflow(decord_arr):
    """from decord to tensorflow, no copy"""
    tfdl = try_import_tfdl()
    return tfdl.from_dlpack(decord_arr.to_dlpack())

def from_tensorflow(tf_tensor):
    """from tensorflow to decord, no copy"""
    tfdl = try_import_tfdl()
    return _from_dlpack(tfdl.to_dlpack(tf_tensor))
