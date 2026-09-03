"""Deterministic split assignment: image-level, hash-based, zero intersection."""

from __future__ import annotations

from . import SPLIT_SALT
from .common import sha256_text

SELECTION = "selection"
LOCKED_TEST = "locked_test"


def assign_split(image_sha256: str) -> str:
    """~20% of images land in locked_test; all boxes of an image share one split."""
    digest = int(sha256_text(f"{SPLIT_SALT}|{image_sha256}"), 16)
    return LOCKED_TEST if digest % 5 == 0 else SELECTION
