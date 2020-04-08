"""Deep Learning Framework bridges."""
from __future__ import absolute_import

from .mxnet import from_decord as to_mxnet
from .mxnet import to_decord as from_mxnet
from .torchdl import to_torch, from_torch
from .tf import to_tensorflow, from_tensorflow

_BRIDGE_TYPES = {
    'native': (lambda x: x, lambda x: x),
    'mxnet': (to_mxnet, from_mxnet),
    'torch': (to_torch, from_torch),
    'tensorflow': (to_tensorflow, from_tensorflow)
}

_CURRENT_BRIDGE = 'native'

def reset_bridge():
    _CURRENT_BRIDGE = 'native'

def set_bridge(new_bridge):
    assert isinstance(new_bridge, str), (
        "New bridge type must be str. Choices: {}".format(_BRIDGE_TYPES.keys()))
    assert new_bridge in _BRIDGE_TYPES.keys(), (
        "valid bridges: {}".format(_BRIDGE_TYPES.keys()))
    global _CURRENT_BRIDGE
    _CURRENT_BRIDGE = new_bridge

def bridge_out(native_arr):
    return _BRIDGE_TYPES[_CURRENT_BRIDGE][0](native_arr)

def bridge_in(arr):
    return _BRIDGE_TYPES[_CURRENT_BRIDGE][1](arr)

class _BridgeScope(object):
    def __init__(self, bridge_type='native'):
        self._type = bridge_type
        self._prev = None

    def __enter__(self):
        self._prev = _CURRENT_BRIDGE
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
