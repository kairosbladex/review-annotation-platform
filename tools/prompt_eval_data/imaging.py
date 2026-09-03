"""Image byte fetching, SHA-256 and stdlib-only JPEG/PNG size parsing."""

from __future__ import annotations

import struct
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from .common import sha256_bytes

DEFAULT_IMAGE_BASE_URL = "https://lxtest.zhuoyilingxi.com:8088/minio-src"


def fetch_bytes(
    object_key: str,
    *,
    image_root: str | None = None,
    base_url: str = DEFAULT_IMAGE_BASE_URL,
    timeout: float = 30.0,
    retries: int = 2,
) -> bytes:
    """Read image bytes from a local directory or an HTTP(S) prefix. Read-only."""
    if image_root:
        return (Path(image_root) / object_key).read_bytes()
    url = f"{base_url.rstrip('/')}/{object_key.lstrip('/')}"
    last_error: Exception | None = None
    for _ in range(max(1, retries + 1)):
        try:
            with urllib.request.urlopen(url, timeout=timeout) as resp:
                return resp.read()
        except Exception as exc:  # noqa: BLE001 - surfaced verbatim in the manifest
            last_error = exc
    raise RuntimeError(f"download failed: {last_error}")


def image_size(data: bytes) -> tuple[int, int]:
    """Return (width, height) for JPEG or PNG bytes without third-party deps."""
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        if data[12:16] != b"IHDR":
            raise ValueError("PNG missing IHDR")
        width, height = struct.unpack(">II", data[16:24])
        return width, height
    if data[:2] == b"\xff\xd8":
        return _jpeg_size(data)
    raise ValueError("unsupported image format (not JPEG/PNG)")


def _jpeg_size(data: bytes) -> tuple[int, int]:
    i = 2
    n = len(data)
    while i + 1 < n:
        if data[i] != 0xFF:
            i += 1
            continue
        marker = data[i + 1]
        if marker == 0xD8 or marker == 0xD9 or 0xD0 <= marker <= 0xD7:
            i += 2
            continue
        if i + 4 > n:
            break
        seg_len = (data[i + 2] << 8) | data[i + 3]
        if 0xC0 <= marker <= 0xCF and marker not in (0xC4, 0xC8, 0xCC):
            if i + 9 > n:
                break
            height = (data[i + 5] << 8) | data[i + 6]
            width = (data[i + 7] << 8) | data[i + 8]
            return width, height
        i += 2 + seg_len
    raise ValueError("JPEG missing SOF segment")


def resolve_images(
    object_keys: list[str],
    *,
    image_root: str | None = None,
    base_url: str = DEFAULT_IMAGE_BASE_URL,
    workers: int = 8,
) -> tuple[dict[str, dict], dict[str, str]]:
    """Fetch every object key, returning ({key: image-info}, {key: error})."""

    def _one(key: str) -> tuple[str, dict | None, str | None]:
        try:
            data = fetch_bytes(key, image_root=image_root, base_url=base_url)
            width, height = image_size(data)
            return key, {"sha256": sha256_bytes(data), "width": width, "height": height}, None
        except Exception as exc:  # noqa: BLE001 - recorded, never raised past here
            return key, None, str(exc)

    images: dict[str, dict] = {}
    errors: dict[str, str] = {}
    with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
        for key, info, error in pool.map(_one, object_keys):
            if info is not None:
                images[key] = info
            else:
                errors[key] = error or "unknown error"
    return images, errors
