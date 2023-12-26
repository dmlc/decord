"""DECORD Paddle bridge"""
from __future__ import absolute_import

from .._ffi._ctypes.ndarray import _from_dlpack

def try_import_paddle():
    """Try import paddle at runtime.

    Returns
    -------
    paddle module if found. Raise ImportError otherwise
    """
    msg = "paddle is required, for installation guide, please checkout:\n \
        https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/develop/install/pip/linux-pip.html"
    try:
        return __import__('paddle.utils.dlpack', fromlist=['object'])
    except ImportError as e:
        if not message:
            raise e
        raise ImportError(message)

def to_paddle(decord_arr):
    """From decord to paddle.
    The tensor will share the memory with the object represented in the dlpack.
    Note that each dlpack can only be consumed once."""
    dlpack = try_import_paddle()
    return dlpack.from_dlpack(decord_arr.to_dlpack())

def from_paddle(tensor):
    """From paddle to decord.
    The dlpack shares the tensors memory.
    Note that each dlpack can only be consumed once."""
    dlpack = try_import_paddle()
    return _from_dlpack(dlpack.to_dlpack(tensor))