"""Model download, cache, and load management.

WiSense ships **no pretrained model weights** in the pip package itself
(this keeps the install small and is a deliberate design decision
documented in the project README). Every feature module (presence,
fall, activity) works fully without any model, using a statistical
baseline method. The :class:`ModelRegistry` below implements the real
mechanics for the *optional* upgrade path: checking a local cache
directory, downloading a named ``.onnx`` file from a configurable base
URL if it isn't cached, verifying its SHA-256 checksum, and loading it
via ``onnxruntime.InferenceSession``.

No real model hosting URL exists yet for this project -- the
``DEFAULT_MODEL_BASE_URL`` below is intentionally a placeholder value
(clearly marked as such) rather than a fabricated working endpoint.
Model training/publishing is out of scope for this repository; see the
README's "Model Files" section.
"""

from __future__ import annotations

import hashlib
import logging
import os
import urllib.error
import urllib.request
from pathlib import Path
from typing import Dict, Optional

from wisense.exceptions import ModelChecksumError, ModelNotFoundError

logger = logging.getLogger("wisense.models.registry")

# PLACEHOLDER -- set this to your real model-hosting URL before release.
# This is NOT a working endpoint. Downloads against this default will
# fail with a clear ModelNotFoundError; that is intentional rather than
# silently fabricating a fake successful download.
DEFAULT_MODEL_BASE_URL = "https://models.example.invalid/wisense/PLACEHOLDER-set-real-url"

DEFAULT_CACHE_DIR = Path(os.environ.get("WISENSE_MODEL_CACHE", str(Path.home() / ".wisense" / "models")))


class ModelRegistry:
    """Manages a local cache of ``.onnx`` model files, downloading and
    checksum-verifying them from a configurable base URL on demand.

    Parameters
    ----------
    cache_dir:
        Directory models are cached in. Defaults to
        ``~/.wisense/models`` (overridable via the ``WISENSE_MODEL_CACHE``
        environment variable).
    base_url:
        Base URL models are downloaded from, joined with the model
        filename, e.g. ``f"{base_url}/{filename}"``. Defaults to a
        placeholder that will not resolve -- see module docstring.
    """

    def __init__(self, cache_dir: Optional[Path] = None, base_url: Optional[str] = None) -> None:
        self.cache_dir = Path(cache_dir) if cache_dir is not None else DEFAULT_CACHE_DIR
        self.base_url = base_url if base_url is not None else DEFAULT_MODEL_BASE_URL
        self._sessions: Dict[str, object] = {}

    def local_path(self, filename: str) -> Path:
        return self.cache_dir / filename

    def is_cached(self, filename: str) -> bool:
        return self.local_path(filename).exists()

    def ensure_downloaded(self, filename: str, sha256: Optional[str] = None) -> Path:
        """Return the local path to ``filename``, downloading it from
        ``self.base_url`` into ``self.cache_dir`` first if it is not
        already cached.

        Parameters
        ----------
        filename: model file name, e.g. ``"presence_v1.onnx"``.
        sha256:
            Expected SHA-256 hex digest. If given and the (already
            cached or freshly downloaded) file's digest doesn't match,
            raises :class:`~wisense.exceptions.ModelChecksumError`.

        Raises
        ------
        ModelNotFoundError
            If the file is not cached locally and the download fails
            (network error, 404, or ``base_url`` is unset/placeholder).
        ModelChecksumError
            If ``sha256`` is given and does not match.
        """
        path = self.local_path(filename)
        if path.exists():
            logger.debug("Model %s found in cache at %s", filename, path)
        else:
            self._download(filename, path)

        if sha256 is not None:
            actual = self._sha256_of(path)
            if actual.lower() != sha256.lower():
                # Remove the bad file so a retry doesn't just find "cached
                # garbage" again.
                try:
                    path.unlink()
                except OSError:
                    pass
                raise ModelChecksumError(
                    f"checksum mismatch for {filename}: expected {sha256}, got {actual}"
                )
        return path

    def _download(self, filename: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"{self.base_url.rstrip('/')}/{filename}"
        tmp_dest = dest.with_suffix(dest.suffix + ".part")
        logger.info("Downloading model %s from %s", filename, url)
        try:
            with urllib.request.urlopen(url, timeout=30) as response, open(tmp_dest, "wb") as out:
                out.write(response.read())
        except (urllib.error.URLError, urllib.error.HTTPError, ValueError, OSError) as exc:
            if tmp_dest.exists():
                try:
                    tmp_dest.unlink()
                except OSError:
                    pass
            raise ModelNotFoundError(
                f"could not download model {filename!r} from {url!r}: {exc}. "
                "If you have not configured a real model-hosting URL yet, "
                "this is expected -- see ModelRegistry(base_url=...) and "
                "the README's 'Model Files' section. All statistical "
                "(non-ONNX) detection paths work without any model file."
            ) from exc
        tmp_dest.rename(dest)

    @staticmethod
    def _sha256_of(path: Path) -> str:
        h = hashlib.sha256()
        with open(path, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()

    def load_session(self, filename: str, sha256: Optional[str] = None):
        """Return a cached ``onnxruntime.InferenceSession`` for
        ``filename``, downloading/verifying it first if necessary.
        Sessions are cached per-registry-instance so repeated calls
        don't reload the model from disk.

        Raises :class:`~wisense.exceptions.ModelNotFoundError` if
        ``onnxruntime`` is not installed, or if the model cannot be
        obtained.
        """
        if filename in self._sessions:
            return self._sessions[filename]

        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover - exercised only without onnxruntime
            raise ModelNotFoundError(
                "onnxruntime is not installed; install it with "
                "`pip install onnxruntime` to use ONNX inference paths, "
                "or omit `model_path`/`use_model` to use the statistical "
                "baseline detector instead."
            ) from exc

        path = self.ensure_downloaded(filename, sha256=sha256)
        session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
        self._sessions[filename] = session
        return session

    def load_session_from_path(self, path: Path):
        """Load an ``onnxruntime.InferenceSession`` directly from a
        local file path, bypassing the cache/download machinery
        entirely (for users who already have a model file on disk).
        """
        try:
            import onnxruntime as ort
        except ImportError as exc:  # pragma: no cover
            raise ModelNotFoundError(
                "onnxruntime is not installed; install it with `pip install onnxruntime`."
            ) from exc
        path = Path(path)
        if not path.exists():
            raise ModelNotFoundError(f"model file not found: {path}")
        return ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])


_default_registry: Optional[ModelRegistry] = None


def get_default_registry() -> ModelRegistry:
    """Return a process-wide default :class:`ModelRegistry` instance,
    creating it on first call. Feature modules use this so callers
    don't have to construct a registry themselves for the common case."""
    global _default_registry
    if _default_registry is None:
        _default_registry = ModelRegistry()
    return _default_registry
