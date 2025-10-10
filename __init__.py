"""
Top-level resfrac package shim.

This repo contains an inner package directory at resfrac/resfrac/.
To allow imports like `import resfrac.tools ...`, we extend the package
search path to include the inner directory.
"""
import os as _os
import sys as _sys

# Extend package path so `resfrac.tools` resolves to resfrac/resfrac/tools
_inner = _os.path.join(_os.path.dirname(__file__), 'resfrac')
if _os.path.isdir(_inner) and _inner not in __path__:
    __path__.append(_inner)  # type: ignore[name-defined]

# Convenience re-export
try:
    from .resfrac.holo_index import HolographicSublinearIndex  # noqa: F401
except Exception:
    HolographicSublinearIndex = None  # type: ignore

__all__ = [
    name for name, obj in (
        ("HolographicSublinearIndex", HolographicSublinearIndex),
    ) if obj is not None
]
