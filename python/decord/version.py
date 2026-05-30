import os

# Base package version
_BASE_VERSION = "3.4.0"

# Optional PEP 440 local version segment, e.g. "+cu132" for CUDA wheels.
# This is read at build time by setuptools (via pyproject dynamic version)
# and at runtime when importing decord.version. CPU wheels should leave this
# unset so versions remain plain (e.g., "3.3.0").
_suffix = os.environ.get("DECORD_LOCAL_VERSION_SUFFIX") or os.environ.get("LOCAL_VERSION_SUFFIX") or ""
if _suffix:
    normalized = _suffix.strip().lstrip("+").lower().replace("-", ".").replace("_", ".")
    __version__ = f"{_BASE_VERSION}+{normalized}"
else:
    __version__ = _BASE_VERSION
