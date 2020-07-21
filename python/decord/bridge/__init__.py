"""Deep Learning Framework bridges."""
from __future__ import absolute_import
import threading

from .mxnet import to_mxnet, from_mxnet
from .torchdl import to_torch, from_torch
from .tf import to_tensorflow, from_tensorflow
from .tvm import to_tvm, from_tvm

_BRIDGE_TYPES = {
    'native': (lambda x: x, lambda x: x),
    'mxnet': (to_mxnet, from_mxnet),
    'torch': (to_torch, from_torch),
    'tensorflow': (to_tensorflow, from_tensorflow),
    'tvm': (to_tvm, from_tvm),
}

_CURRENT_BRIDGE = threading.local()
_CURRENT_BRIDGE.type = 'native'
_GLOBAL_BRIDGE_TYPE = 'native'  # child threads will derive from the global type but not overwrite

def reset_bridge():
    _CURRENT_BRIDGE.type = 'native'
    if threading.current_thread().name == 'MainThread':
        _GLOBAL_BRIDGE_TYPE = 'native'

def set_bridge(new_bridge):
    assert isinstance(new_bridge, str), (
        "New bridge type must be str. Choices: {}".format(_BRIDGE_TYPES.keys()))
    assert new_bridge in _BRIDGE_TYPES.keys(), (
        "valid bridges: {}".format(_BRIDGE_TYPES.keys()))
    global _CURRENT_BRIDGE
    _CURRENT_BRIDGE.type = new_bridge
    if threading.current_thread().name == 'MainThread':
        _GLOBAL_BRIDGE_TYPE = new_bridge

def bridge_out(native_arr):
    if not hasattr(_CURRENT_BRIDGE, 'type'):
        _CURRENT_BRIDGE.type = _GLOBAL_BRIDGE_TYPE
    return _BRIDGE_TYPES[_CURRENT_BRIDGE.type][0](native_arr)

def bridge_in(arr):
    if not hasattr(_CURRENT_BRIDGE, 'type'):
        _CURRENT_BRIDGE.type = _GLOBAL_BRIDGE_TYPE
    return _BRIDGE_TYPES[_CURRENT_BRIDGE.type][1](arr)

class _BridgeScope(object):
    def __init__(self, bridge_type='native'):
        self._type = bridge_type
        self._prev = None

    def __enter__(self):
        global _CURRENT_BRIDGE
        if not hasattr(_CURRENT_BRIDGE, 'type'):
            _CURRENT_BRIDGE.type = _GLOBAL_BRIDGE_TYPE
        try:
            self._prev = _CURRENT_BRIDGE.type
        except AttributeError:
            self._prev = 'native'
        set_bridge(self._type)

    def __exit__(self, type, value, traceback):
        if self._prev != self._type:
            set_bridge(self._prev)

def use_mxnet():
    return _BridgeScope('mxnet')

def use_torch():
    return _BridgeScope('torch')

def use_tensorflow():
    return _BridgeScope('tensorflow')

def use_tvm():
    return _BridgeScope('tvm')
