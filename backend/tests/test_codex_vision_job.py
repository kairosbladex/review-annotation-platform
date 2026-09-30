import importlib.util
from pathlib import Path

import pytest


spec = importlib.util.spec_from_file_location(
    "vision_job", Path(__file__).resolve().parents[2] / "scripts/codex_vision_job.py")
job = importlib.util.module_from_spec(spec)
spec.loader.exec_module(job)


def request(**changes):
    return {"task": "review", "images": ["target.jpg"], "prompt": "Review target",
            "api_key": "test-only", **changes}


@pytest.mark.parametrize("name", ["../secret", "/etc/passwd", "a/b.jpg", ".", ".."])
def test_paths_cannot_escape_job(tmp_path, name):
    with pytest.raises(ValueError):
        job.validate_request(request(images=[name]), tmp_path)


def test_symlink_rejected(tmp_path):
    (tmp_path / "real.jpg").write_bytes(b"image")
    (tmp_path / "target.jpg").symlink_to(tmp_path / "real.jpg")
    with pytest.raises(ValueError):
        job.validate_request(request(), tmp_path)


def test_size_and_credential_guards(tmp_path):
    image = tmp_path / "target.jpg"
    image.write_bytes(b"image")
    assert job.validate_request(request(), tmp_path) == ["target.jpg"]
    with pytest.raises(ValueError):
        job.validate_request(request(api_key=""), tmp_path)
    with image.open("wb") as stream:
        stream.truncate(5_000_001)
    with pytest.raises(ValueError):
        job.validate_request(request(), tmp_path)


@pytest.mark.parametrize("change", [{"task": "dispatch"}, {"images": []},
                                     {"images": ["target.jpg"] * 4},
                                     {"prompt": "x" * 12001}])
def test_request_limits(tmp_path, change):
    (tmp_path / "target.jpg").write_bytes(b"image")
    with pytest.raises(ValueError):
        job.validate_request(request(**change), tmp_path)
