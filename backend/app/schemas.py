from __future__ import annotations

from datetime import datetime
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class ImageReference(BaseModel):
    bucket: str = Field(min_length=1, max_length=255)
    object_key: str = Field(min_length=1, max_length=1024)
    sha256: str | None = Field(default=None, max_length=128)


class CandidateInput(BaseModel):
    target_index: int = Field(ge=0)
    class_id: int | None = None
    class_name: str = Field(min_length=1, max_length=160)
    clean_bbox: list[float] | None = Field(default=None, min_length=4, max_length=4)
    confidence: float | None = Field(default=None, ge=0, le=1)
    bbox: list[float] = Field(min_length=4, max_length=4)
    raw_decision: str | None = None
    yolo_decision: str | None = None
    llm_decision: str | None = None
    reason_code: str | None = None
    error_code: str | None = None
    reasoning: str = ""

    @field_validator("bbox", "clean_bbox")
    @classmethod
    def normalized_bbox(cls, value: list[float]) -> list[float]:
        if value is None:
            return value
        if any(item < 0 or item > 1 for item in value):
            raise ValueError("bbox must use normalized coordinates between 0 and 1")
        if value[2] <= value[0] or value[3] <= value[1]:
            raise ValueError("bbox must have positive width and height")
        return value


class ReviewEventInput(BaseModel):
    event_id: str = Field(min_length=1, max_length=255)
    source_service: str = Field(min_length=1, max_length=120)
    session_id: str | None = Field(default=None, max_length=120)
    capture_time: datetime | None = None
    camera_id: str | None = Field(default=None, max_length=120)
    rule_code: str | None = Field(default=None, max_length=120)
    image: ImageReference
    clean_image: ImageReference | None = None
    clean_event_id: str | None = None
    comparison_image: ImageReference | None = None
    targets: list[CandidateInput] = Field(default_factory=list, max_length=1000)
    metadata: dict[str, Any] = Field(default_factory=dict)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=120)
    password: str = Field(min_length=1, max_length=255)


class AnnotationUpdate(BaseModel):
    label: Literal["true", "false"] | None
    note: str | None = Field(default=None, max_length=2000)
    expected_revision: int | None = Field(default=None, ge=0)


class AgentReviewRequestInput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    cycle_id: str = Field(min_length=1, max_length=120)
    candidate_id: str = Field(min_length=1, max_length=32)
    reason_code: str = Field(min_length=1, max_length=120)
    reason: str = Field(min_length=1, max_length=4000)
    attempt_refs: list[Annotated[str, Field(min_length=1, max_length=120)]] = Field(
        default_factory=list,
        max_length=20,
    )

    @field_validator("cycle_id", "candidate_id", "reason_code", "reason")
    @classmethod
    def reject_blank_values(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("value must not be blank")
        return value

    @field_validator("attempt_refs")
    @classmethod
    def unique_attempt_refs(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value]
        if any(not item for item in normalized):
            raise ValueError("attempt refs must not be blank")
        if len(set(normalized)) != len(normalized):
            raise ValueError("attempt refs must be unique")
        return normalized


class ModelRunRequest(BaseModel):
    sample_ids: list[str] = Field(min_length=1, max_length=10000)
    model_profile_id: str
    options: dict[str, Any] = Field(default_factory=dict)


class AutoAnnotationRequest(BaseModel):
    sample_ids: list[str] = Field(min_length=1, max_length=10000)
    options: dict[str, Any] = Field(default_factory=dict)


class ModelProfileCreate(BaseModel):
    profile_key: str = Field(min_length=1, max_length=120)
    name: str = Field(min_length=1, max_length=160)
    model_name: str = Field(min_length=1, max_length=160)
    model_version: str = Field(min_length=1, max_length=160)
    adapter_type: str = Field(default="http", max_length=80)
    config: dict[str, Any] = Field(default_factory=dict)


class MinioImportRequest(BaseModel):
    source_type: Literal["minio"] = "minio"
    bucket: str = Field(min_length=1, max_length=255)
    prefix: str = Field(default="", max_length=1024)
    manifest_key: str | None = Field(default=None, max_length=1024)


class ExportRequest(BaseModel):
    run_id: str | None = None
    formats: list[Literal["yolo", "jsonl"]] = Field(default_factory=lambda: ["jsonl"], min_length=1)
    filters: dict[str, Any] = Field(default_factory=dict)
