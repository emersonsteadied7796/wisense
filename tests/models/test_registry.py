import hashlib
import http.server
import threading
from pathlib import Path

import pytest

from wisense.exceptions import ModelChecksumError, ModelNotFoundError
from wisense.models.registry import ModelRegistry


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


class _StaticHTTPServer:
    """Minimal local HTTP server serving files from a directory, used
    to test ModelRegistry's download path without any real network
    access (binds to 127.0.0.1 only)."""

    def __init__(self, directory: Path):
        handler_cls = lambda *args, **kwargs: http.server.SimpleHTTPRequestHandler(
            *args, directory=str(directory), **kwargs
        )
        self.httpd = http.server.HTTPServer(("127.0.0.1", 0), handler_cls)
        self.port = self.httpd.server_address[1]
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    def start(self):
        self.thread.start()

    def stop(self):
        self.httpd.shutdown()
        self.thread.join(timeout=5.0)

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"


def test_is_cached_false_initially(tmp_path: Path):
    registry = ModelRegistry(cache_dir=tmp_path / "cache")
    assert registry.is_cached("model.onnx") is False


def test_ensure_downloaded_pulls_from_server(tmp_path: Path):
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    content = b"fake-onnx-bytes-for-testing"
    (serve_dir / "model.onnx").write_bytes(content)

    server = _StaticHTTPServer(serve_dir)
    server.start()
    try:
        registry = ModelRegistry(cache_dir=tmp_path / "cache", base_url=server.base_url)
        path = registry.ensure_downloaded("model.onnx")
        assert path.exists()
        assert path.read_bytes() == content
        assert registry.is_cached("model.onnx")
    finally:
        server.stop()


def test_ensure_downloaded_uses_cache_without_refetching(tmp_path: Path):
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    (cache_dir / "model.onnx").write_bytes(b"already-here")

    # base_url deliberately invalid -- if the registry tried to
    # download, this would fail; success proves it used the cache.
    registry = ModelRegistry(cache_dir=cache_dir, base_url="http://127.0.0.1:1")
    path = registry.ensure_downloaded("model.onnx")
    assert path.read_bytes() == b"already-here"


def test_ensure_downloaded_verifies_checksum(tmp_path: Path):
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    content = b"checksum-me"
    (serve_dir / "model.onnx").write_bytes(content)

    server = _StaticHTTPServer(serve_dir)
    server.start()
    try:
        registry = ModelRegistry(cache_dir=tmp_path / "cache", base_url=server.base_url)
        path = registry.ensure_downloaded("model.onnx", sha256=_sha256(content))
        assert path.exists()
    finally:
        server.stop()


def test_ensure_downloaded_rejects_bad_checksum(tmp_path: Path):
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    content = b"checksum-me"
    (serve_dir / "model.onnx").write_bytes(content)

    server = _StaticHTTPServer(serve_dir)
    server.start()
    try:
        registry = ModelRegistry(cache_dir=tmp_path / "cache", base_url=server.base_url)
        with pytest.raises(ModelChecksumError):
            registry.ensure_downloaded("model.onnx", sha256="0" * 64)
    finally:
        server.stop()


def test_ensure_downloaded_missing_file_raises_model_not_found(tmp_path: Path):
    serve_dir = tmp_path / "serve"
    serve_dir.mkdir()
    server = _StaticHTTPServer(serve_dir)
    server.start()
    try:
        registry = ModelRegistry(cache_dir=tmp_path / "cache", base_url=server.base_url)
        with pytest.raises(ModelNotFoundError):
            registry.ensure_downloaded("does_not_exist.onnx")
    finally:
        server.stop()


def test_default_base_url_is_placeholder_and_fails_clearly(tmp_path: Path):
    registry = ModelRegistry(cache_dir=tmp_path / "cache")
    with pytest.raises(ModelNotFoundError):
        registry.ensure_downloaded("anything.onnx")


def test_load_session_from_path_missing_file_raises(tmp_path: Path):
    registry = ModelRegistry(cache_dir=tmp_path / "cache")
    with pytest.raises(ModelNotFoundError):
        registry.load_session_from_path(tmp_path / "nope.onnx")
