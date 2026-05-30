# coding: utf-8
# pylint: disable=invalid-name
"""ctypes library and helper functions """
from __future__ import absolute_import

import sys
import os
import ctypes
import numpy as np
from . import libinfo

#----------------------------
# library loading
#----------------------------
if sys.version_info[0] == 3:
    string_types = (str,)
    numeric_types = (float, int, np.float32, np.int32)
    # this function is needed for python3
    # to convert ctypes.char_p .value back to python str
    py_str = lambda x: x.decode('utf-8')
else:
    string_types = (basestring,)
    numeric_types = (float, int, long, np.float32, np.int32)
    py_str = lambda x: x


class DECORDError(Exception):
    """Error thrown by DECORD function"""
    pass  # pylint: disable=unnecessary-pass

class DECORDLimitReachedError(Exception):
    """Limit Reached Error thrown by DECORD function"""
    pass  # pylint: disable=unnecessary-pass

def _preload_cuda_libs():
    """Best-effort: make pip-installed CUDA runtime libraries discoverable.

    The CUDA builds of decord2 (``+cuXXX`` wheels) do NOT bundle the CUDA
    runtime (libcudart / libnvrtc / libcublas, ...). When the user installs it
    via the NVIDIA pip packages (e.g. ``pip install decord2[cu13]`` which pulls
    ``nvidia-cuda-runtime`` / ``nvidia-cuda-nvrtc`` / ``nvidia-cublas``), those
    libraries live under ``site-packages/nvidia/<component>/`` and are not on
    the default loader search path. Expose them here, before the native library
    is loaded, so its CUDA dependencies resolve.

    This is purely additive and silent: if nothing is installed (CPU build, or
    the user relies on a system CUDA install) nothing happens and the previous
    behaviour is preserved. ``nvcuvid`` / the CUDA driver API are provided by
    the GPU driver, not by the pip packages.
    """
    import glob
    import importlib.util

    if sys.platform.startswith("win32"):
        # Windows resolves DLLs via the search path; decord.dll imports
        # nvcuda.dll / nvcuvid.dll which the driver installs in System32, so the
        # only thing we may need is to expose the pip-provided CUDA runtime DLLs
        # (nvidia/<component>/bin).
        try:
            spec = importlib.util.find_spec("nvidia")
        except Exception:  # pylint: disable=broad-except
            spec = None
        if spec is not None and getattr(spec, "submodule_search_locations", None):
            for root in list(spec.submodule_search_locations):
                for dll_dir in glob.glob(os.path.join(root, "*", "bin")):
                    try:
                        os.add_dll_directory(dll_dir)
                    except (OSError, AttributeError):
                        pass
                    os.environ["PATH"] = dll_dir + os.pathsep + os.environ.get("PATH", "")
        return

    if not sys.platform.startswith("linux"):
        return  # macOS: no CUDA.

    # POSIX (Linux): dlopen the relevant CUDA shared objects with RTLD_GLOBAL so
    # the native library's undefined cu*/cuvid*/cudart* symbols resolve from the
    # already-loaded modules, even when they are not on the default search path.

    # 1) Driver-provided libraries (CUDA Driver API + NVDEC). These are NEEDED
    #    by libdecord.so but live with the driver; preloading them guarantees
    #    resolution regardless of DT_NEEDED. No-op on driverless/CPU hosts.
    for driver_lib in ("libcuda.so.1", "libnvcuvid.so.1"):
        try:
            ctypes.CDLL(driver_lib, mode=ctypes.RTLD_GLOBAL)
        except OSError:
            pass

    # 2) CUDA runtime from the nvidia-* pip packages (decord2[cu13]), if present.
    #    Several passes resolve inter-library ordering (e.g. cublas needs cudart).
    try:
        spec = importlib.util.find_spec("nvidia")
    except Exception:  # pylint: disable=broad-except
        spec = None
    if spec is None or not getattr(spec, "submodule_search_locations", None):
        return
    candidates = []
    for root in list(spec.submodule_search_locations):
        candidates.extend(glob.glob(os.path.join(root, "*", "lib", "*.so*")))
    remaining = list(candidates)
    for _ in range(3):
        if not remaining:
            break
        still_failing = []
        for so_path in remaining:
            try:
                ctypes.CDLL(so_path, mode=ctypes.RTLD_GLOBAL)
            except OSError:
                still_failing.append(so_path)
        if len(still_failing) == len(remaining):
            break  # no progress in this pass; give up
        remaining = still_failing


def _load_lib():
    """Load libary by searching possible path."""
    lib_path = libinfo.find_lib_path()
    # Make pip-provided CUDA runtime libs (decord2[cuXXX]) loadable first.
    _preload_cuda_libs()
    os.environ['PATH'] += os.pathsep + os.path.dirname(lib_path[0])
    lib = ctypes.CDLL(lib_path[0], ctypes.RTLD_GLOBAL)
    # DMatrix functions
    lib.DECORDGetLastError.restype = ctypes.c_char_p
    return lib, os.path.basename(lib_path[0])

# version number
__version__ = libinfo.__version__
# library instance of nnvm
_LIB, _LIB_NAME = _load_lib()

# The FFI mode of DECORD
_FFI_MODE = os.environ.get("DECORD_FFI", "auto")

# enable stack trace or not
_ENABLE_STACK_TRACE = int(os.environ.get("DECORD_ENABLE_STACK_TRACE", "0"))

#----------------------------
# helper function in ctypes.
#----------------------------
def check_call(ret):
    """Check the return value of C API call

    This function will raise exception when error occurs.
    Wrap every API call with this function

    Parameters
    ----------
    ret : int
        return value from API calls
    """
    if ret != 0:
        err_str = py_str(_LIB.DECORDGetLastError())
        if not _ENABLE_STACK_TRACE:
            if 'Stack trace' in err_str:
                err_str = err_str.split('Stack trace')[0].strip()
        if 'recovered from nearest frames' in err_str:
            if 'Stack trace' in err_str:
                err_str = err_str.split('Stack trace')[0].strip()
            raise DECORDLimitReachedError(err_str)
        raise DECORDError(err_str)


def c_str(string):
    """Create ctypes char * from a python string
    Parameters
    ----------
    string : string type
        python string

    Returns
    -------
    str : c_char_p
        A char pointer that can be passed to C API
    """
    return ctypes.c_char_p(string.encode('utf-8'))


def c_array(ctype, values):
    """Create ctypes array from a python array

    Parameters
    ----------
    ctype : ctypes data type
        data type of the array we want to convert to

    values : tuple or list
        data content

    Returns
    -------
    out : ctypes array
        Created ctypes array
    """
    return (ctype * len(values))(*values)


def decorate(func, fwrapped):
    """A wrapper call of decorator package, differs to call time

    Parameters
    ----------
    func : function
        The original function

    fwrapped : function
        The wrapped function
    """
    import decorator
    return decorator.decorate(func, fwrapped)
