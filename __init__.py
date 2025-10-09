# resfrac package init

# Convenience re-exports (from inner package)
try:
    from .resfrac.holo_index import HolographicSublinearIndex  # noqa: F401
except Exception:
    HolographicSublinearIndex = None  # type: ignore

__all__ = [
    name for name, obj in (
        ("HolographicSublinearIndex", HolographicSublinearIndex),
    ) if obj is not None
]
