"""Model download/cache management for optional ONNX-based inference
paths. See :mod:`wisense.models.registry`."""

from wisense.models.registry import ModelRegistry, get_default_registry

__all__ = ["ModelRegistry", "get_default_registry"]
