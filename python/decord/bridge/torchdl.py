"""DECORD Pytorch bridge"""
from __future__ import absolute_import

from .._ffi._ctypes.ndarray import _from_dlpack

def try_import_torch():
    """Try import torch at runtime.

    Returns
    -------
    torch module if found. Raise ImportError otherwise
    """
    message = "torch is required, you can install by pip: `pip install torch`"
    try:
        return __import__('torch.utils.dlpack', fromlist=['object'])
    except ImportError as e:
        if not message:
            raise e
        raise ImportError(message)

def to_torch(decord_arr):
    """From decord to torch.
    The tensor will share the memory with the object represented in the dlpack.
    Note that each dlpack can only be consumed once."""
    dlpack = try_import_torch()
    return dlpack.from_dlpack(decord_arr.to_dlpack())

def from_torch(tensor):
    """From torch to decord.
    The dlpack shares the tensors memory.
    Note that each dlpack can only be consumed once."""
    dlpack = try_import_torch()
    return _from_dlpack(dlpack.to_dlpack(tensor))