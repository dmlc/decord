"""DECORD logging module.

You can adjust the logging level for ffmpeg.
"""
from ._ffi.function import _init_api

QUIET = -8
PANIC = 0
FATAL = 8
ERROR = 16
WARNING = 24
INFO = 32
VERBOSE = 40
DEBUG = 48
TRACE = 56

# Mimicking stdlib.
CRITICAL = FATAL

def set_level(lvl=ERROR):
    _CAPI_SetLoggingLevel(lvl)

_init_api("decord.logging")
