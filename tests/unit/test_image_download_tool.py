from __future__ import annotations

import json
import os
import urllib.error
from io import BytesIO
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from schemas.errors import (
    IMAGE_DOWNLOAD_TOOL_ERROR,
    IMAGE_DOWNLOAD_TOOL_NOT_FOUND,
    IMAGE_DOWNLOAD_TOOL_PROVIDER_ERROR,
    IMAGE_DOWNLOAD_TOOL_TIMEOUT,
    TOOL_ARGUMENT_ERROR,
)
from tools.impl.image_download_tool import ImageDownloadTool


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _unsplash_response(count: int = 1) -> bytes:
    results = [
        {
            "id": f"abc{i}",
            "urls": {"raw": f"https://images.unsplash.com/photo-{i}?ixid=x", "full": ""},
            "width": 5000,
            "height": 3333,
            "alt_description": f"sample image {i}",
            "description": None,
            "user": {"name": f"Photographer {i}"},
        }
        for i in range(count)
    ]
    return json.dumps({"results": results, "total": count}).encode()


def _pexels_response(count: int = 1) -> bytes:
    photos = [
        {
            "id": i,
            "width": 4000,
            "height": 2667,
            "alt": f"pexels image {i}",
            "photographer": f"Shooter {i}",
            "src": {
                "original": f"https://images.pexels.com/photos/{i}/original.jpg",
                "large2x": "",
                "large": "",
            },
        }
        for i in range(count)
    ]
    return json.dumps({"photos": photos, "total_results": count}).encode()


def _fake_urlopen(search_body: bytes, image_bytes: bytes = b"FAKEIMAGE"):
    """Return a context-manager mock that serves search_body on first call, image_bytes on second."""
    call_count = {"n": 0}

    def _urlopen(req, timeout=None):
        call_count["n"] += 1
        body = search_body if call_count["n"] == 1 else image_bytes
        mock_resp = MagicMock()
        mock_resp.read = MagicMock(side_effect=lambda limit=None: body if limit is None else body[:limit])
        mock_resp.__enter__ = lambda s: s
        mock_resp.__exit__ = MagicMock(return_value=False)
        return mock_resp

    return _urlopen


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def tool():
    return ImageDownloadTool()


@pytest.fixture
def tmp_image_dir(tmp_path, monkeypatch):
    monkeypatch.setattr("tools.impl.image_download_tool._ASSETS_IMAGE_DIR", tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# Tool metadata
# ---------------------------------------------------------------------------

def test_tool_name(tool):
    assert tool.name == "image_download"


def test_tool_schema_has_required_fields(tool):
    schema = tool.schema()
    assert schema["name"] == "image_download"
    props = schema["parameters"]["properties"]
    assert "description" in props
    assert "resolution" in props
    assert "width" in props
    assert "height" in props
    assert "aspect_ratio" in props
    assert "orientation" in props
    assert "count" in props
    assert "provider" in props


# ---------------------------------------------------------------------------
# Argument validation
# ---------------------------------------------------------------------------

def test_empty_description_returns_error(tool):
    result = tool.run({"description": ""})
    assert not result.success
    assert result.error.code == TOOL_ARGUMENT_ERROR


def test_missing_description_returns_error(tool):
    result = tool.run({})
    assert not result.success
    assert result.error.code == TOOL_ARGUMENT_ERROR


def test_no_api_key_returns_provider_error(tool, monkeypatch):
    monkeypatch.delenv("UNSPLASH_ACCESS_KEY", raising=False)
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    result = tool.run({"description": "mountain sunset"})
    assert not result.success
    assert result.error.code == IMAGE_DOWNLOAD_TOOL_PROVIDER_ERROR


# ---------------------------------------------------------------------------
# Resolution / dimension mapping
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("resolution,expected_w,expected_h", [
    ("720p",  1280, 720),
    ("1080p", 1920, 1080),
    ("1440p", 2560, 1440),
    ("4K",    3840, 2160),
])
def test_resolution_maps_to_dimensions(tool, tmp_image_dir, monkeypatch, resolution, expected_w, expected_h):
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "test-key")
    captured: dict = {}

    def fake_search(self_inner, query, orientation, count, timeout):
        return [("https://images.unsplash.com/photo-0?ixid=x", {"photographer": "X", "original_width": 5000, "original_height": 3333, "alt": ""})]

    def fake_download(url, dest, timeout):
        captured["url"] = url
        dest.write_bytes(b"FAKEIMAGE")

    with patch.object(ImageDownloadTool, "_search_unsplash", fake_search), \
         patch.object(ImageDownloadTool, "_download_file", staticmethod(fake_download)):
        result = tool.run({"description": "test", "resolution": resolution})

    assert result.success, result.error
    assert f"w={expected_w}" in captured["url"]
    assert f"h={expected_h}" in captured["url"]


# ---------------------------------------------------------------------------
# Orientation derived from aspect ratio
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("aspect_ratio,expected_orientation", [
    ("16:9",  "landscape"),
    ("9:16",  "portrait"),
    ("1:1",   "squarish"),
    ("4:3",   "landscape"),
])
def test_orientation_derived_from_aspect_ratio(tool, tmp_image_dir, monkeypatch, aspect_ratio, expected_orientation):
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "test-key")
    captured: dict = {}

    def fake_search(self_inner, query, orientation, count, timeout):
        captured["orientation"] = orientation
        return [("https://images.unsplash.com/photo-0", {"photographer": "X", "original_width": 5000, "original_height": 3333, "alt": ""})]

    def fake_download(url, dest, timeout):
        dest.write_bytes(b"FAKEIMAGE")

    with patch.object(ImageDownloadTool, "_search_unsplash", fake_search), \
         patch.object(ImageDownloadTool, "_download_file", staticmethod(fake_download)):
        tool.run({"description": "test", "aspect_ratio": aspect_ratio})

    assert captured.get("orientation") == expected_orientation


# ---------------------------------------------------------------------------
# Unsplash provider
# ---------------------------------------------------------------------------

def test_unsplash_success(tool, tmp_image_dir, monkeypatch):
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "test-key")
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)

    with patch("urllib.request.urlopen", _fake_urlopen(_unsplash_response(), b"IMGDATA")):
        result = tool.run({"description": "mountain lake", "provider": "unsplash"})

    assert result.success
    data = json.loads(result.output)["data"]
    assert data["downloaded"] == 1
    saved_path = Path(data["images"][0]["path"])
    assert saved_path.exists()
    assert saved_path.read_bytes() == b"IMGDATA"


def test_unsplash_no_results_returns_not_found(tool, tmp_image_dir, monkeypatch):
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "test-key")
    empty = json.dumps({"results": [], "total": 0}).encode()

    with patch("urllib.request.urlopen", _fake_urlopen(empty)):
        result = tool.run({"description": "xyzzy impossible query", "provider": "unsplash"})

    assert not result.success
    assert result.error.code == IMAGE_DOWNLOAD_TOOL_NOT_FOUND


def test_unsplash_401_returns_provider_error(tool, tmp_image_dir, monkeypatch):
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "bad-key")

    def _raise_401(req, timeout=None):
        raise urllib.error.HTTPError(url="", code=401, msg="Unauthorized", hdrs=None, fp=None)

    with patch("urllib.request.urlopen", _raise_401):
        result = tool.run({"description": "test", "provider": "unsplash"})

    assert not result.success
    assert result.error.code == IMAGE_DOWNLOAD_TOOL_PROVIDER_ERROR


# ---------------------------------------------------------------------------
# Pexels provider
# ---------------------------------------------------------------------------

def test_pexels_success(tool, tmp_image_dir, monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    monkeypatch.delenv("UNSPLASH_ACCESS_KEY", raising=False)

    with patch("urllib.request.urlopen", _fake_urlopen(_pexels_response(), b"PEXELSIMG")):
        result = tool.run({"description": "city skyline", "provider": "pexels"})

    assert result.success
    data = json.loads(result.output)["data"]
    assert data["downloaded"] == 1
    saved_path = Path(data["images"][0]["path"])
    assert saved_path.read_bytes() == b"PEXELSIMG"


def test_pexels_no_results_returns_not_found(tool, tmp_image_dir, monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    empty = json.dumps({"photos": [], "total_results": 0}).encode()

    with patch("urllib.request.urlopen", _fake_urlopen(empty)):
        result = tool.run({"description": "xyzzy impossible", "provider": "pexels"})

    assert not result.success
    assert result.error.code == IMAGE_DOWNLOAD_TOOL_NOT_FOUND


def test_pexels_403_returns_provider_error(tool, tmp_image_dir, monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "bad-key")

    def _raise_403(req, timeout=None):
        raise urllib.error.HTTPError(url="", code=403, msg="Forbidden", hdrs=None, fp=None)

    with patch("urllib.request.urlopen", _raise_403):
        result = tool.run({"description": "test", "provider": "pexels"})

    assert not result.success
    assert result.error.code == IMAGE_DOWNLOAD_TOOL_PROVIDER_ERROR


# ---------------------------------------------------------------------------
# Auto-provider detection
# ---------------------------------------------------------------------------

def test_auto_detects_unsplash_when_key_present(tool, tmp_image_dir, monkeypatch):
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "test-key")
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    captured: dict = {}

    def fake_unsplash(self_inner, query, orientation, count, timeout):
        captured["provider"] = "unsplash"
        return [("https://images.unsplash.com/photo-0", {"photographer": "X", "original_width": 100, "original_height": 100, "alt": ""})]

    def fake_download(url, dest, timeout):
        dest.write_bytes(b"X")

    with patch.object(ImageDownloadTool, "_search_unsplash", fake_unsplash), \
         patch.object(ImageDownloadTool, "_download_file", staticmethod(fake_download)):
        result = tool.run({"description": "test"})

    assert result.success
    assert captured.get("provider") == "unsplash"


def test_auto_detects_pexels_when_only_pexels_key(tool, tmp_image_dir, monkeypatch):
    monkeypatch.delenv("UNSPLASH_ACCESS_KEY", raising=False)
    monkeypatch.setenv("PEXELS_API_KEY", "test-key")
    captured: dict = {}

    def fake_pexels(self_inner, query, orientation, count, timeout):
        captured["provider"] = "pexels"
        return [("https://images.pexels.com/photos/0/original.jpg", {"photographer": "Y", "original_width": 100, "original_height": 100, "alt": ""})]

    def fake_download(url, dest, timeout):
        dest.write_bytes(b"X")

    with patch.object(ImageDownloadTool, "_search_pexels", fake_pexels), \
         patch.object(ImageDownloadTool, "_download_file", staticmethod(fake_download)):
        result = tool.run({"description": "test"})

    assert result.success
    assert captured.get("provider") == "pexels"


# ---------------------------------------------------------------------------
# Count parameter
# ---------------------------------------------------------------------------

def test_count_downloads_multiple_images(tool, tmp_image_dir, monkeypatch):
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "test-key")

    with patch("urllib.request.urlopen", _fake_urlopen(_unsplash_response(3), b"IMG")):
        result = tool.run({"description": "forest", "provider": "unsplash", "count": 3})

    assert result.success
    data = json.loads(result.output)["data"]
    assert data["downloaded"] == 3
    assert len(data["images"]) == 3


def test_count_capped_at_5(tool, tmp_image_dir, monkeypatch):
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "test-key")
    captured: dict = {}

    def fake_search(self_inner, query, orientation, count, timeout):
        captured["count"] = count
        return [("https://images.unsplash.com/photo-0", {"photographer": "X", "original_width": 100, "original_height": 100, "alt": ""})]

    def fake_download(url, dest, timeout):
        dest.write_bytes(b"X")

    with patch.object(ImageDownloadTool, "_search_unsplash", fake_search), \
         patch.object(ImageDownloadTool, "_download_file", staticmethod(fake_download)):
        tool.run({"description": "test", "count": 99})

    assert captured.get("count") == 5


# ---------------------------------------------------------------------------
# Custom filename
# ---------------------------------------------------------------------------

def test_custom_filename_used(tool, tmp_image_dir, monkeypatch):
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "test-key")

    with patch("urllib.request.urlopen", _fake_urlopen(_unsplash_response(), b"IMG")):
        result = tool.run({"description": "test", "provider": "unsplash", "filename": "my_photo"})

    assert result.success
    fname = json.loads(result.output)["data"]["images"][0]["filename"]
    assert fname.startswith("my_photo")


# ---------------------------------------------------------------------------
# Timeout
# ---------------------------------------------------------------------------

def test_search_timeout_returns_timeout_error(tool, tmp_image_dir, monkeypatch):
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "test-key")

    def _raise_timeout(req, timeout=None):
        raise TimeoutError("timed out")

    with patch("urllib.request.urlopen", _raise_timeout):
        result = tool.run({"description": "test", "provider": "unsplash", "timeout": 1})

    assert not result.success
    assert result.error.code == IMAGE_DOWNLOAD_TOOL_TIMEOUT


# ---------------------------------------------------------------------------
# File size guard
# ---------------------------------------------------------------------------

def test_oversized_image_fails_gracefully(tool, tmp_image_dir, monkeypatch):
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "test-key")

    def fake_search(self_inner, query, orientation, count, timeout):
        return [("https://images.unsplash.com/photo-0", {"photographer": "X", "original_width": 100, "original_height": 100, "alt": ""})]

    # Simulate read() returning more than the limit
    oversized = b"X" * (51 * 1024 * 1024)

    def fake_download(url, dest, timeout):
        from tools.impl.image_download_tool import _MAX_FILE_SIZE_MB
        data = oversized[: _MAX_FILE_SIZE_MB * 1024 * 1024 + 1]
        if len(data) > _MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"Image exceeds {_MAX_FILE_SIZE_MB} MB size limit.")
        dest.write_bytes(data)

    with patch.object(ImageDownloadTool, "_search_unsplash", fake_search), \
         patch.object(ImageDownloadTool, "_download_file", staticmethod(fake_download)):
        result = tool.run({"description": "test", "provider": "unsplash"})

    assert not result.success
    assert result.error.code == IMAGE_DOWNLOAD_TOOL_ERROR


# ---------------------------------------------------------------------------
# Output structure
# ---------------------------------------------------------------------------

def test_output_contains_save_directory(tool, tmp_image_dir, monkeypatch):
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "test-key")

    with patch("urllib.request.urlopen", _fake_urlopen(_unsplash_response(), b"IMG")):
        result = tool.run({"description": "ocean", "provider": "unsplash"})

    assert result.success
    data = json.loads(result.output)["data"]
    assert "save_directory" in data
    assert "images" in data
    assert "downloaded" in data


def test_image_entry_has_expected_keys(tool, tmp_image_dir, monkeypatch):
    monkeypatch.setenv("UNSPLASH_ACCESS_KEY", "test-key")

    with patch("urllib.request.urlopen", _fake_urlopen(_unsplash_response(), b"IMG")):
        result = tool.run({"description": "desert", "provider": "unsplash"})

    assert result.success
    img = json.loads(result.output)["data"]["images"][0]
    assert "filename" in img
    assert "path" in img
    assert "source_url" in img
    assert "photographer" in img
