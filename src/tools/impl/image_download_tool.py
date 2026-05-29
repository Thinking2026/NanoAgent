from __future__ import annotations

import json
import os
import re
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from schemas import (
    IMAGE_DOWNLOAD_TOOL_ERROR,
    IMAGE_DOWNLOAD_TOOL_NOT_FOUND,
    IMAGE_DOWNLOAD_TOOL_PROVIDER_ERROR,
    IMAGE_DOWNLOAD_TOOL_TIMEOUT,
    TOOL_ARGUMENT_ERROR,
    ToolResult,
    build_pipeline_error,
)
from tools.tool_base import BaseTool, build_tool_output

_ASSETS_IMAGE_DIR = Path(__file__).parents[3] / "assets" / "image"
_DEFAULT_TIMEOUT = 30
_MAX_TIMEOUT = 60
_MAX_FILE_SIZE_MB = 50

_RESOLUTION_MAP: dict[str, tuple[int, int]] = {
    "720p":  (1280, 720),
    "1080p": (1920, 1080),
    "1440p": (2560, 1440),
    "2k":    (2048, 1080),
    "4k":    (3840, 2160),
    "8k":    (7680, 4320),
}

_SAFE_FILENAME = re.compile(r"[^\w\-.]")


class ImageDownloadTool(BaseTool):
    name = "image_download"
    description = (
        "Search for images by description and download them to assets/image/. "
        "Supports filtering by resolution (e.g. '1080p', '4K'), aspect ratio (e.g. '16:9'), "
        "and orientation (landscape/portrait/squarish). "
        "Requires UNSPLASH_ACCESS_KEY or PEXELS_API_KEY environment variable. "
        "Returns saved file paths for later reference."
    )
    parameters = {
        "type": "object",
        "properties": {
            "description": {
                "type": "string",
                "description": "Description of the image to search for. Be specific for better results.",
            },
            "resolution": {
                "type": "string",
                "description": "Target resolution: '720p', '1080p', '1440p', '2K', '4K', '8K'.",
                "enum": ["720p", "1080p", "1440p", "2K", "4K", "8K"],
            },
            "width": {
                "type": "integer",
                "description": "Desired width in pixels. Used when resolution is not specified.",
                "minimum": 1,
            },
            "height": {
                "type": "integer",
                "description": "Desired height in pixels. Used when resolution is not specified.",
                "minimum": 1,
            },
            "aspect_ratio": {
                "type": "string",
                "description": "Desired aspect ratio, e.g. '16:9', '4:3', '1:1', '9:16'.",
                "pattern": r"^\d+:\d+$",
            },
            "orientation": {
                "type": "string",
                "description": "Image orientation filter.",
                "enum": ["landscape", "portrait", "squarish"],
            },
            "count": {
                "type": "integer",
                "description": "Number of images to download. Defaults to 1, max 5.",
                "default": 1,
                "minimum": 1,
                "maximum": 5,
            },
            "filename": {
                "type": "string",
                "description": "Custom base filename (without extension). Auto-generated if omitted.",
            },
            "provider": {
                "type": "string",
                "description": "Image provider. Auto-detected from available API keys if omitted.",
                "enum": ["unsplash", "pexels"],
            },
            "timeout": {
                "type": "integer",
                "description": f"HTTP timeout in seconds. Defaults to {_DEFAULT_TIMEOUT}, max {_MAX_TIMEOUT}.",
                "default": _DEFAULT_TIMEOUT,
                "minimum": 1,
                "maximum": _MAX_TIMEOUT,
            },
        },
        "required": ["description"],
        "additionalProperties": False,
    }

    def run(self, arguments: dict[str, Any]) -> ToolResult:
        description = str(arguments.get("description", "")).strip()
        if not description:
            return self._error_result(
                build_pipeline_error(TOOL_ARGUMENT_ERROR, "description is required and must not be empty.")
            )

        resolution = str(arguments["resolution"]).lower() if arguments.get("resolution") else None
        width: int | None = arguments.get("width")
        height: int | None = arguments.get("height")
        aspect_ratio: str | None = arguments.get("aspect_ratio")
        orientation: str | None = arguments.get("orientation")
        count = min(int(arguments.get("count", 1)), 5)
        custom_filename: str | None = arguments.get("filename")
        provider: str | None = arguments.get("provider")
        timeout = min(int(arguments.get("timeout", _DEFAULT_TIMEOUT)), _MAX_TIMEOUT)

        if resolution and not (width or height):
            resolved = _RESOLUTION_MAP.get(resolution)
            if resolved:
                width, height = resolved

        if aspect_ratio and not orientation:
            try:
                w_ratio, h_ratio = map(int, aspect_ratio.split(":"))
                if w_ratio > h_ratio:
                    orientation = "landscape"
                elif h_ratio > w_ratio:
                    orientation = "portrait"
                else:
                    orientation = "squarish"
            except (ValueError, AttributeError):
                pass

        if not provider:
            if os.environ.get("UNSPLASH_ACCESS_KEY"):
                provider = "unsplash"
            elif os.environ.get("PEXELS_API_KEY"):
                provider = "pexels"
            else:
                return self._error_result(
                    build_pipeline_error(
                        IMAGE_DOWNLOAD_TOOL_PROVIDER_ERROR,
                        "No image API key found. Set UNSPLASH_ACCESS_KEY or PEXELS_API_KEY.",
                    )
                )

        _ASSETS_IMAGE_DIR.mkdir(parents=True, exist_ok=True)

        try:
            if provider == "unsplash":
                image_entries = self._search_unsplash(description, orientation, count, timeout)
            else:
                image_entries = self._search_pexels(description, orientation, count, timeout)
        except TimeoutError:
            return self._error_result(
                build_pipeline_error(IMAGE_DOWNLOAD_TOOL_TIMEOUT, f"Image search timed out after {timeout}s.")
            )
        except PermissionError as exc:
            return self._error_result(build_pipeline_error(IMAGE_DOWNLOAD_TOOL_PROVIDER_ERROR, str(exc)))
        except LookupError:
            return self._error_result(
                build_pipeline_error(IMAGE_DOWNLOAD_TOOL_NOT_FOUND, f"No images found for: {description!r}")
            )
        except Exception as exc:
            return self._error_result(build_pipeline_error(IMAGE_DOWNLOAD_TOOL_ERROR, f"Search failed: {exc}"))

        saved: list[dict[str, Any]] = []
        for idx, (img_url, meta) in enumerate(image_entries):
            try:
                download_url = self._build_download_url(provider, img_url, width, height)
                ext = self._guess_extension(download_url)
                fname = self._make_filename(custom_filename, description, idx, ext)
                dest = _ASSETS_IMAGE_DIR / fname
                self._download_file(download_url, dest, timeout)
                saved.append({"filename": fname, "path": str(dest), "source_url": img_url, **meta})
            except Exception as exc:
                saved.append({"error": str(exc), "source_url": img_url})

        successful = [s for s in saved if "error" not in s]
        if not successful:
            errors = "; ".join(s.get("error", "unknown") for s in saved)
            return self._error_result(
                build_pipeline_error(IMAGE_DOWNLOAD_TOOL_ERROR, f"All downloads failed: {errors}")
            )

        return ToolResult(
            output=build_tool_output(
                success=True,
                data={
                    "downloaded": len(successful),
                    "save_directory": str(_ASSETS_IMAGE_DIR),
                    "images": saved,
                },
            ),
            success=True,
        )

    def _search_unsplash(
        self, query: str, orientation: str | None, count: int, timeout: int
    ) -> list[tuple[str, dict[str, Any]]]:
        api_key = os.environ["UNSPLASH_ACCESS_KEY"]
        params: dict[str, Any] = {"query": query, "per_page": count, "content_filter": "high"}
        if orientation:
            params["orientation"] = orientation
        url = "https://api.unsplash.com/search/photos?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"Authorization": f"Client-ID {api_key}", "User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise PermissionError(f"Unsplash API key invalid (HTTP {exc.code})") from exc
            raise Exception(f"Unsplash API error HTTP {exc.code}") from exc

        results = data.get("results", [])
        if not results:
            raise LookupError("no results")

        out: list[tuple[str, dict[str, Any]]] = []
        for item in results:
            raw_url = item["urls"].get("raw") or item["urls"].get("full", "")
            meta: dict[str, Any] = {
                "photographer": item.get("user", {}).get("name", ""),
                "original_width": item.get("width"),
                "original_height": item.get("height"),
                "alt": item.get("alt_description") or item.get("description") or "",
            }
            out.append((raw_url, meta))
        return out

    def _search_pexels(
        self, query: str, orientation: str | None, count: int, timeout: int
    ) -> list[tuple[str, dict[str, Any]]]:
        api_key = os.environ["PEXELS_API_KEY"]
        params: dict[str, Any] = {"query": query, "per_page": count}
        if orientation:
            params["orientation"] = orientation
        url = "https://api.pexels.com/v1/search?" + urllib.parse.urlencode(params)
        req = urllib.request.Request(url, headers={"Authorization": api_key, "User-Agent": "Mozilla/5.0"})
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                data = json.loads(resp.read().decode())
        except urllib.error.HTTPError as exc:
            if exc.code in (401, 403):
                raise PermissionError(f"Pexels API key invalid (HTTP {exc.code})") from exc
            raise Exception(f"Pexels API error HTTP {exc.code}") from exc

        photos = data.get("photos", [])
        if not photos:
            raise LookupError("no results")

        out: list[tuple[str, dict[str, Any]]] = []
        for item in photos:
            src = item.get("src", {})
            raw_url = src.get("original") or src.get("large2x") or src.get("large", "")
            meta: dict[str, Any] = {
                "photographer": item.get("photographer", ""),
                "original_width": item.get("width"),
                "original_height": item.get("height"),
                "alt": item.get("alt", ""),
            }
            out.append((raw_url, meta))
        return out

    @staticmethod
    def _build_download_url(
        provider: str, base_url: str, width: int | None, height: int | None
    ) -> str:
        if provider == "unsplash" and (width or height):
            params: dict[str, Any] = {"fit": "crop", "auto": "format"}
            if width:
                params["w"] = width
            if height:
                params["h"] = height
            sep = "&" if "?" in base_url else "?"
            return base_url + sep + urllib.parse.urlencode(params)
        return base_url

    @staticmethod
    def _guess_extension(url: str) -> str:
        path = urllib.parse.urlparse(url).path.lower()
        for ext in (".jpg", ".jpeg", ".png", ".webp", ".gif"):
            if path.endswith(ext):
                return ext.lstrip(".")
        return "jpg"

    @staticmethod
    def _make_filename(custom: str | None, description: str, idx: int, ext: str) -> str:
        base = _SAFE_FILENAME.sub("_", custom if custom else description[:40]).strip("_")
        suffix = f"_{idx + 1}" if idx > 0 else ""
        ts = int(time.time())
        return f"{base}{suffix}_{ts}.{ext}"

    @staticmethod
    def _download_file(url: str, dest: Path, timeout: int) -> None:
        req = urllib.request.Request(url, headers={"User-Agent": "NanoAgent/1.0 (image-download-tool)"})
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            data = resp.read(_MAX_FILE_SIZE_MB * 1024 * 1024 + 1)
        if len(data) > _MAX_FILE_SIZE_MB * 1024 * 1024:
            raise ValueError(f"Image exceeds {_MAX_FILE_SIZE_MB} MB size limit.")
        dest.write_bytes(data)

    @staticmethod
    def _error_result(error: Any) -> ToolResult:
        return ToolResult(
            output=build_tool_output(success=False, error=error),
            success=False,
            error=error,
        )
