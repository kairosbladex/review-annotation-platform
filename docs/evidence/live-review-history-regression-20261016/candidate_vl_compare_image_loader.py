"""Function-only replacement for the hash-pinned live api/vl_compare.py.

The release builder replaces only _load_image, preserving the rest of the
deployed file. Names outside the function are supplied by that live module.
"""
from __future__ import annotations


async def _load_image(item: ImageInput) -> LoadedImage:
    from PIL import UnidentifiedImageError

    def decode(stream) -> Image.Image:
        try:
            with Image.open(stream) as image:
                return image.convert("RGB")
        except ModuleNotFoundError as exc:
            # Ultralytics optionally tries HEIF after Pillow rejects the bytes.
            # An unrelated missing dependency remains an execution failure.
            if exc.name != "pi_heif" or not isinstance(exc.__context__, UnidentifiedImageError):
                raise
            raise HTTPException(status_code=422, detail="无法读取图片，请使用有效的JPEG或PNG图片") from exc
        except (UnidentifiedImageError, OSError, Image.DecompressionBombError) as exc:
            if isinstance(exc, OSError) and exc.errno is not None:
                # Preserve file-access and system I/O errors; only decode errors
                # without an OS errno are classified as invalid image content.
                raise
            raise HTTPException(status_code=422, detail="无法读取图片，请使用有效的JPEG或PNG图片") from exc

    if item.source_type == "base64":
        raw = (item.base64_data or "").split(",", 1)[-1]
        try:
            decoded = base64.b64decode(raw, validate=True)
        except (binascii.Error, ValueError):
            try:
                decoded = base64.b64decode(raw + "===")
            except (binascii.Error, ValueError) as exc:
                raise HTTPException(status_code=422, detail="图片base64编码无效") from exc
        img = decode(BytesIO(decoded))
        return LoadedImage(img, img.width, img.height, item.source_type, item.image_id or item.filename or "")
    if item.source_type == "local":
        with Path(item.path).open("rb") as stream:
            img = decode(stream)
        return LoadedImage(img, img.width, img.height, item.source_type, item.image_id or item.path or "")
    if item.source_type == "http":
        verify_setting = _httpx_verify_setting()
        logger.info("VL compare httpx verify=%s url=%s", verify_setting, item.url)
        async with httpx.AsyncClient(timeout=_env_float("VL_COMPARE_IMAGE_LOAD_TIMEOUT", 8.0), verify=verify_setting) as client:
            resp = await client.get(item.url)
            resp.raise_for_status()
        img = decode(BytesIO(resp.content))
        return LoadedImage(img, img.width, img.height, item.source_type, item.image_id or item.url or "")
    ref = item.url or item.object_name or ""
    if ref.startswith("http://") or ref.startswith("https://"):
        return await _load_image(ImageInput(source_type="http", url=ref, image_id=item.image_id, label=item.label))
    if MINIO_PUBLIC_BASE_URL and item.object_name:
        url = f"{MINIO_PUBLIC_BASE_URL}/{item.object_name.lstrip('/')}"
        return await _load_image(ImageInput(source_type="http", url=url, image_id=item.image_id, label=item.label))
    raise HTTPException(status_code=422, detail="minio source_type requires public url or object_name with MINIO_PUBLIC_BASE_URL")
