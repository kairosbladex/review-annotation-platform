"""Jev-Omni Tier 2.5（Fast Gate）集成测试。

覆盖四组契约：

1. 双视角合图：1024×512 画幅、8px 中性灰分隔条、ROI 零遮挡；
2. 概率解析与异常兜底：多形态输出归一化、超时/HTTP 错误/图像不可解码时
   100% 安全回退 Tier 3，绝不中断主流程；
3. 级联裁决：刚体类高概率确认、假框低概率拒识、`blue_illegal_roof` 强制穿透、
   灰区与几何冲突回退；
4. 配置传导：`Settings -> build_auto_annotation_options`。
"""

from __future__ import annotations

import base64
import json
from io import BytesIO

import httpx
import pytest
from fastapi.testclient import TestClient
from PIL import Image
from sqlalchemy import select

from app.adapters.jev_omni_classifier import (
    DEFAULT_CANVAS_WIDTH,
    DEFAULT_TARGET_HEIGHT,
    NEUTRAL_GRAY_RGB,
    JevOmniClassifierAdapter,
    JevOmniUnavailableError,
    build_jev_omni_composite_image,
    build_jev_omni_decision_query,
    build_jev_omni_review_views,
    normalise_binary_probabilities,
    separator_column_bounds,
)
from app.config import Settings
from app.main import create_app
from app.models import Annotation, AuditEvent
from app.services.auto_annotation_service import (
    JevOmniTierRunner,
    build_auto_annotation_options,
)
from app.services.ktd_recheck import (
    TIER25_REASON_CONFIRM,
    TIER25_REASON_GRAY,
    TIER25_REASON_REJECT,
    evaluate_jev_omni_tier,
)

from .test_annotation_flow import _login, _review_event
from .test_imports import MemoryObjectStore

_TICK_RED = (235, 64, 52)


def _solid(color: tuple[int, int, int], size: tuple[int, int] = (1024, 1024)) -> Image.Image:
    return Image.new("RGB", size, color)


def _jpeg_bytes(image: Image.Image) -> bytes:
    buffer = BytesIO()
    image.save(buffer, format="JPEG", quality=92)
    return buffer.getvalue()


# ---------------------------------------------------------------------------
# 1. 双视角合图
# ---------------------------------------------------------------------------
def test_composite_is_1024x512_with_8px_neutral_separator_and_no_occlusion():
    context = _solid((12, 200, 40))
    roi = _solid((200, 20, 40), (800, 400))

    composite = build_jev_omni_composite_image(context, roi)

    assert composite.size == (DEFAULT_CANVAS_WIDTH, DEFAULT_TARGET_HEIGHT) == (1024, 512)
    separator_start, separator_end = separator_column_bounds()
    assert separator_end - separator_start == 8
    assert (separator_start, separator_end) == (508, 516)

    pixels = composite.load()
    for x in range(separator_start, separator_end):
        for y in (0, 1, 255, 511):
            assert pixels[x, y] == NEUTRAL_GRAY_RGB

    # 分隔条之外必须立刻是两幅视图的内容，而不是灰边。
    assert pixels[separator_start - 1, 256] != NEUTRAL_GRAY_RGB
    assert pixels[separator_end, 256] != NEUTRAL_GRAY_RGB

    # 左右面板内容保真：上下文为绿、ROI 为红，未被串色或覆盖。
    assert pixels[100, 256] == (12, 200, 40)
    assert pixels[900, 256] == (200, 20, 40)

    # ROI 面板零遮挡：不含任何 L 型角标像素。
    roi_panel = composite.crop((separator_end, 0, composite.width, composite.height))
    assert _TICK_RED not in set(roi_panel.getdata())


def test_review_views_mark_context_with_ticks_and_keep_roi_clean():
    frame = Image.new("RGB", (960, 720), (90, 120, 150))
    for x in range(240, 530):
        for y in range(180, 400):
            frame.putpixel((x, y), (250, 250, 250))

    context, roi = build_jev_omni_review_views(frame, [0.25, 0.25, 0.55, 0.55])

    assert context.size == roi.size == (512, 512)
    assert _TICK_RED in set(context.getdata())
    assert _TICK_RED not in set(roi.getdata())
    # ROI 是零遮挡的高清局部：框内亮色目标必须出现在画面中心。
    assert roi.getpixel((256, 256)) == (250, 250, 250)


def test_review_views_keep_oversized_wide_target_fully_visible():
    """回归：1920×1080 画幅里 1200×300 的超宽目标不能被压到短边而截断两端。"""
    width, height = 1920, 1080
    background = (60, 60, 60)
    target_color = (250, 250, 250)
    frame = Image.new("RGB", (width, height), background)
    # 目标框 1200×300，居中于 (960, 540)：左右各留 360px 余量，完全在画幅内。
    box_left, box_top, box_right, box_bottom = 360, 390, 1560, 690
    frame.paste(
        Image.new("RGB", (box_right - box_left, box_bottom - box_top), target_color),
        (box_left, box_top),
    )
    bbox = [box_left / width, box_top / height, box_right / width, box_bottom / height]

    context, roi = build_jev_omni_review_views(frame, bbox)

    assert context.size == roi.size == (512, 512)

    # ROI 视口边长 = 1200 + 2×8 = 1216（放宽到长边），缩放后目标横向占 ~505px，
    # 左右各留 ~3.4px 背景边；若视口仍被压到短边 1080，左端 60px 目标会被切掉，
    # 表现为最左列直接是目标色、没有背景边。
    roi_row = [roi.getpixel((x, 256)) for x in range(512)]
    target_columns = [x for x, pixel in enumerate(roi_row) if pixel == target_color]
    assert target_columns, "ROI 视图中目标本体完全消失"
    first, last = target_columns[0], target_columns[-1]
    assert first >= 1 and last <= 510, "目标两端顶到视口边缘，说明视口被截断"
    assert roi_row[0] == background and roi_row[511] == background
    assert last - first + 1 >= 500, "目标横向跨度异常，可能被压到短边后裁切"
    # 目标纵向 300px → 视口内 ~126px，中心行必须是目标本体。
    assert roi.getpixel((256, 256)) == target_color

    # 上下文视口（边长 1920，缩放比 512/1920）同样完整包含目标两端：
    # 目标在视图内占 x≈96..416，两侧必须还是背景色。
    scale = 512 / 1920.0
    assert int(box_left * scale) == 96 and int(box_right * scale) == 416
    assert context.getpixel((90, 256)) == background
    assert context.getpixel((422, 256)) == background
    assert context.getpixel((100, 256)) == target_color
    assert context.getpixel((412, 256)) == target_color


def test_decision_query_embeds_category_guidelines_and_fixed_options():
    query = build_jev_omni_decision_query(
        "三轮车", {"candidate_id": "c-1", "detector_confidence": 0.42}
    )

    assert query["options"] == ["Yes", "No"]
    assert query["question"] == (
        "Based on the left context view and right high-resolution ROI view, "
        "is the target indicated by the corner markers a genuine positive instance of 'tricycle'?"
    )
    # 复用 CATEGORY_SPECIFIC_GUIDELINES 原文（三轮车 vs 汽车的核心区分准则）。
    assert "三轮结构" in query["state"]
    assert "严禁将小型四轮轿车" in query["state"]
    # 候选元数据按不可信数据处理，且带 L 型角标/双视角说明。
    assert "c-1" in query["state"]
    assert "不可信" in query["state"]
    assert "L 型角标" in query["state"]


def test_unknown_class_still_produces_a_yes_no_query():
    query = build_jev_omni_decision_query(None)
    assert query["options"] == ["Yes", "No"]
    assert "'unknown'" in query["question"]


# ---------------------------------------------------------------------------
# 2. 概率解析与异常兜底
# ---------------------------------------------------------------------------
@pytest.mark.parametrize(
    "payload",
    [
        {"Yes": 0.91, "No": 0.09},
        {"probs": {"Yes": 0.91, "No": 0.09}},
        {"probabilities": [0.91, 0.09]},
        {"Yes": 0.91},
        {"label": "Yes", "confidence": 0.91},
        [{"label": "Yes", "probability": 0.91}, {"label": "No", "probability": 0.09}],
        {"p_true": 0.91, "p_false": 0.09},
        {"Yes": 2.3, "No": 0.2},  # logits -> softmax
    ],
)
def test_probability_parsing_variants_normalise_to_complementary_pair(payload):
    p_true, p_false, raw_probs = normalise_binary_probabilities(payload)

    assert 0.0 <= p_true <= 1.0
    assert p_true + p_false == 1.0
    assert raw_probs["Yes"] == p_true
    assert raw_probs["No"] == p_false
    assert p_true > 0.5  # 所有用例都指向 Yes


@pytest.mark.parametrize("payload", [{}, {"probs": []}, {"Yes": float("nan")}, "not-json"])
def test_probability_parsing_rejects_unusable_output(payload):
    with pytest.raises(JevOmniUnavailableError):
        normalise_binary_probabilities(payload)


def test_http_mode_posts_composite_to_v1_decide_and_returns_standard_payload():
    captured: dict[str, object] = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200, json={"p_true": 0.93, "p_false": 0.07, "raw_probs": {"Yes": 0.93, "No": 0.07}}
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    adapter = JevOmniClassifierAdapter(
        mode="http", base_url="http://jev-omni.test:8010", client=client
    )
    try:
        result = adapter.decide(
            _solid((10, 180, 30)), _solid((180, 30, 10)), "车辆", {"candidate_id": "c-9"}
        )
    finally:
        adapter.close()

    assert captured["url"] == "http://jev-omni.test:8010/v1/decide"
    body = captured["body"]
    assert body["modality"] == "image"
    assert body["options"] == ["Yes", "No"]
    assert body["model_id"] == "akhilaaa3/Jev-Omni"
    assert "'vehicle'" in body["question"]
    decoded = base64.b64decode(body["image_base64"])
    assert Image.open(BytesIO(decoded)).size == (1024, 512)

    assert set(result) == {"p_true", "p_false", "raw_probs", "latency_ms", "model_id"}
    assert result["p_true"] == pytest.approx(0.93)
    assert result["p_true"] + result["p_false"] == 1.0
    assert result["model_id"] == "akhilaaa3/Jev-Omni"
    assert result["latency_ms"] >= 0.0


@pytest.mark.parametrize(
    "handler",
    [
        pytest.param(lambda request: httpx.Response(500, text="boom"), id="http-500"),
        pytest.param(lambda request: httpx.Response(200, text="not-json"), id="bad-json"),
        pytest.param(
            lambda request: (_ for _ in ()).throw(httpx.ConnectTimeout("timeout")),
            id="timeout",
        ),
        pytest.param(lambda request: httpx.Response(200, json={"unexpected": True}), id="bad-body"),
    ],
)
def test_http_failures_raise_unavailable_error_for_safe_fallback(handler):
    adapter = JevOmniClassifierAdapter(
        mode="http",
        base_url="http://jev-omni.test:8010",
        client=httpx.Client(transport=httpx.MockTransport(handler)),
    )
    with pytest.raises(JevOmniUnavailableError):
        adapter.decide(_solid((1, 2, 3)), _solid((4, 5, 6)), "车")
    adapter.close()


def test_local_mode_without_weights_raises_actionable_error():
    adapter = JevOmniClassifierAdapter(mode="local")
    with pytest.raises(JevOmniUnavailableError) as excinfo:
        adapter.decide(_solid((1, 2, 3)), _solid((4, 5, 6)), "车")
    assert "JEV_OMNI_MODE=http" in str(excinfo.value)


def test_runner_returns_none_when_adapter_fails_so_the_caller_falls_back_to_tier3():
    class Exploding:
        def decide(self, *_args, **_kwargs):
            raise JevOmniUnavailableError("timeout")

    jpeg = _jpeg_bytes(_solid((120, 120, 120), (960, 720)))
    runner = JevOmniTierRunner(Exploding(), lambda ref: jpeg)

    verdict = runner.evaluate(
        {"bucket": "b", "object_key": "k"},
        {"bbox": [0.1, 0.1, 0.3, 0.3], "class_name": "车", "confidence": 0.5},
    )

    assert verdict is None
    assert runner.last_error and "timeout" in runner.last_error


def test_runner_returns_none_when_the_source_image_cannot_be_decoded():
    class NeverCalled:
        def decide(self, *_args, **_kwargs):  # pragma: no cover - 不应被调用
            raise AssertionError("图像不可解码时不得调用模型")

    runner = JevOmniTierRunner(NeverCalled(), lambda ref: b"not-an-image")
    verdict = runner.evaluate(
        {"bucket": "b", "object_key": "k"},
        {"bbox": [0.1, 0.1, 0.3, 0.3], "class_name": "车", "confidence": 0.5},
    )
    assert verdict is None
    assert runner.last_error


def test_runner_caches_the_decoded_frame_across_candidates_of_one_image():
    loads: list[str] = []

    class Fixed:
        def decide(self, *_args, **_kwargs):
            return {
                "p_true": 0.95,
                "p_false": 0.05,
                "raw_probs": {"Yes": 0.95, "No": 0.05},
                "latency_ms": 1.0,
                "model_id": "akhilaaa3/Jev-Omni",
            }

    def loader(ref):
        loads.append(ref["object_key"])
        return _jpeg_bytes(_solid((100, 100, 100), (960, 720)))

    runner = JevOmniTierRunner(Fixed(), loader)
    image_ref = {"bucket": "b", "object_key": "k"}
    for _ in range(3):
        runner.evaluate(image_ref, {"bbox": [0.1, 0.1, 0.2, 0.2], "class_name": "车"})
    assert loads == ["k"]


# ---------------------------------------------------------------------------
# 3. Tier 2.5 级联裁决
# ---------------------------------------------------------------------------
def test_tier25_confirms_rigid_class_above_threshold():
    routing = evaluate_jev_omni_tier({"class_name": "车", "confidence": 0.42}, p_true=0.95)

    assert routing["tier"] == "tier2_5_jev_omni_confirm"
    assert routing["auto_label"] == "true"
    assert routing["requires_vlm"] is False
    assert routing["reason_code"] == TIER25_REASON_CONFIRM
    assert routing["p_true"] == 0.95
    assert routing["p_true"] + routing["p_false"] == 1.0
    assert "自动确认" in routing["explanation"]


def test_tier25_rejects_low_probability_false_box():
    routing = evaluate_jev_omni_tier({"class_name": "车", "confidence": 0.55}, p_true=0.05)

    assert routing["tier"] == "tier2_5_jev_omni_reject"
    assert routing["auto_label"] == "false"
    assert routing["requires_vlm"] is False
    assert routing["reason_code"] == TIER25_REASON_REJECT
    assert "假框自动拒识" in routing["explanation"]
    assert "0.12" in routing["explanation"]


def test_tier25_rejects_non_rigid_low_probability_boxes_but_never_confirms_them():
    rejected = evaluate_jev_omni_tier({"class_name": "道路裂缝"}, p_true=0.04)
    assert rejected["tier"] == "tier2_5_jev_omni_reject"

    not_confirmed = evaluate_jev_omni_tier({"class_name": "道路裂缝"}, p_true=0.97)
    assert not_confirmed["tier"] == "tier3_vlm_verify"
    assert not_confirmed["auto_label"] is None
    assert "不在刚体类白名单" in not_confirmed["explanation"]


def test_blue_illegal_roof_always_passes_through_to_tier3_with_prior():
    for probability in (0.0, 0.05, 0.5, 0.99, 1.0):
        routing = evaluate_jev_omni_tier({"class_name": "违建"}, p_true=probability)
        assert routing["tier"] == "tier3_vlm_verify"
        assert routing["requires_vlm"] is True
        assert routing["auto_label"] is None
        assert routing["reason_code"] == "tier3_scene_dependent_forced_vlm"
        assert routing["p_true"] == probability


def test_gray_zone_and_geometric_conflict_fall_back_to_tier3():
    gray = evaluate_jev_omni_tier({"class_name": "车"}, p_true=0.5)
    assert gray["tier"] == "tier3_vlm_verify"
    assert gray["reason_code"] == TIER25_REASON_GRAY
    assert "灰区" in gray["explanation"]

    conflict = evaluate_jev_omni_tier(
        {"class_name": "车", "geometric_warning": "part_in_whole_containment_warning(containment=0.95)"},
        p_true=0.99,
    )
    assert conflict["tier"] == "tier3_vlm_verify"
    assert conflict["reason_code"] == "tier3_geometric_conflict"
    assert conflict["auto_label"] is None


def test_tier25_rejects_invalid_probabilities():
    for bad in (-0.1, 1.1, float("nan")):
        with pytest.raises(ValueError):
            evaluate_jev_omni_tier({"class_name": "车"}, p_true=bad)


# ---------------------------------------------------------------------------
# 4. 配置传导
# ---------------------------------------------------------------------------
def test_jev_omni_is_disabled_by_default_and_settings_reach_job_options():
    options = build_auto_annotation_options({}, Settings())

    assert options["jev_omni_enabled"] is False
    assert options["jev_omni_mode"] == "http"
    assert options["jev_omni_base_url"] == "http://127.0.0.1:8010"
    assert options["jev_omni_model_id"] == "akhilaaa3/Jev-Omni"
    assert options["jev_omni_confirm_threshold"] == 0.88
    assert options["jev_omni_reject_threshold"] == 0.12


def test_jev_omni_settings_are_threaded_into_options_with_explicit_override():
    settings = Settings(
        jev_omni_enabled=True,
        jev_omni_mode="local",
        jev_omni_base_url="http://gpu-host:8010/",
        jev_omni_confirm_threshold=0.9,
        jev_omni_reject_threshold=0.1,
        jev_omni_timeout_seconds=12.5,
    )

    options = build_auto_annotation_options({}, settings)

    assert options["jev_omni_enabled"] is True
    assert options["jev_omni_mode"] == "local"
    assert options["jev_omni_base_url"] == "http://gpu-host:8010"
    assert options["jev_omni_confirm_threshold"] == 0.9
    assert options["jev_omni_reject_threshold"] == 0.1
    assert options["jev_omni_timeout_seconds"] == 12.5

    overridden = build_auto_annotation_options({"jev_omni_enabled": False}, settings)
    assert overridden["jev_omni_enabled"] is False


def test_jev_omni_env_vars_are_parsed_by_settings(monkeypatch):
    monkeypatch.setenv("JEV_OMNI_ENABLED", "true")
    monkeypatch.setenv("JEV_OMNI_MODE", "HTTP")
    monkeypatch.setenv("JEV_OMNI_BASE_URL", "http://10.0.0.9:8010/")
    monkeypatch.setenv("JEV_OMNI_CONFIRM_THRESHOLD", "0.91")
    monkeypatch.setenv("JEV_OMNI_REJECT_THRESHOLD", "0.09")

    settings = Settings.from_env()

    assert settings.jev_omni_enabled is True
    assert settings.jev_omni_mode == "http"
    assert settings.jev_omni_base_url == "http://10.0.0.9:8010"
    assert settings.jev_omni_confirm_threshold == 0.91
    assert settings.jev_omni_reject_threshold == 0.09


# ---------------------------------------------------------------------------
# 5. 端到端：API 自动标注走 Tier 2.5，且失败时安全回退
# ---------------------------------------------------------------------------
class _RecordingJevOmni:
    """按检测器分数返回固定校准概率的假 Jev-Omni。"""

    def __init__(self, probabilities: dict[float, float]) -> None:
        self.probabilities = probabilities
        self.calls: list[dict] = []

    def decide(self, _context, _roi, class_name, candidate_meta=None):
        meta = dict(candidate_meta or {})
        self.calls.append({"class_name": class_name, **meta})
        confidence = round(float(meta.get("detector_confidence")), 2)
        p_true = self.probabilities[confidence]
        return {
            "p_true": p_true,
            "p_false": 1.0 - p_true,
            "raw_probs": {"Yes": p_true, "No": 1.0 - p_true},
            "latency_ms": 1.0,
            "model_id": "akhilaaa3/Jev-Omni",
        }


class _ExplodingJevOmni:
    def __init__(self) -> None:
        self.calls = 0

    def decide(self, *_args, **_kwargs):
        self.calls += 1
        raise JevOmniUnavailableError("inference service unavailable")


class _InlineDispatcher:
    def __init__(self, runner) -> None:
        self.runner = runner

    def dispatch(self, job_id: str) -> None:
        self.runner(job_id)


def _app(tmp_path, *, store, auto_adapter, jev_omni, enabled=True):
    settings = Settings(
        database_url=f"sqlite:///{tmp_path / 'test.sqlite3'}",
        service_api_key="service-secret",
        admin_username="admin",
        admin_password="admin-password",
        jwt_secret="a-test-jwt-secret-that-is-long-enough",
        minio_endpoint="",
        auto_annotation_api_key="env-secret",
        auto_annotation_cascade_routing=True,
        tier1_auto_confirm_threshold=0.85,
        tier2_auto_reject_threshold=0.12,
        jev_omni_enabled=enabled,
        jev_omni_mode="http",
        jev_omni_base_url="http://jev-omni.test:8010",
    )
    app = create_app(
        settings,
        object_store=store,
        auto_annotation_adapter=auto_adapter,
        jev_omni_adapter=jev_omni,
    )
    app.state.auto_annotation_dispatcher = _InlineDispatcher(app.state.run_auto_annotation_job)
    return app


def _ingest_and_run(client: TestClient, app, event_id: str) -> str:
    event = _review_event(event_id)
    response = client.post(
        "/api/v1/ingest/review-events", json=event, headers={"X-Service-Key": "service-secret"}
    )
    assert response.status_code == 201, response.text
    sample_id = response.json()["sample_id"]
    created = client.post(
        "/api/v1/auto-annotations", json={"sample_ids": [sample_id]}, headers=_login(client)
    )
    assert created.status_code == 202, created.text
    return sample_id


def test_api_auto_annotation_uses_tier25_confirm_and_reject_without_touching_tier3(tmp_path):
    store = MemoryObjectStore()
    store.put_bytes(
        "review-images",
        "review/2026/09/02/session-001/original.jpg",
        _jpeg_bytes(_solid((110, 130, 150), (960, 720))),
        "image/jpeg",
    )
    # 两个候选都是「车」且分数 0.61 / 0.48：Tier1(0.85) 与 Tier2(0.12) 都不会命中。
    jev_omni = _RecordingJevOmni({0.61: 0.95, 0.48: 0.04})
    tier3 = _FixedTier3Adapter()
    app = _app(tmp_path, store=store, auto_adapter=tier3, jev_omni=jev_omni)
    client = TestClient(app)

    _ingest_and_run(client, app, "jev-omni-tier25-001")

    assert tier3.calls == []  # Tier 3 完全未被调用
    assert len(jev_omni.calls) == 2

    with app.state.session_factory() as session:  # type: ignore[attr-defined]
        labels = {
            annotation.candidate_id: annotation.label
            for annotation in session.scalars(select(Annotation)).all()
        }
        events = session.scalars(
            select(AuditEvent).where(AuditEvent.action == "auto_annotation.applied")
        ).all()
    reasons = {
        json.loads(event.payload_json)["cascade_routing"]["reason_code"] for event in events
    }

    assert sorted(labels.values()) == ["false", "true"]
    assert reasons == {TIER25_REASON_CONFIRM, TIER25_REASON_REJECT}


class _FixedTier3Adapter:
    """Tier 3 适配器：记录调用次数并返回固定决策。"""

    def __init__(self, decisions: list | None = None) -> None:
        self.decisions = decisions or []
        self.calls: list = []

    def infer(self, image, options):
        self.calls.append((image, options))
        return {"decisions": self.decisions}


def test_api_auto_annotation_falls_back_to_tier3_when_jev_omni_raises(tmp_path):
    store = MemoryObjectStore()
    store.put_bytes(
        "review-images",
        "review/2026/09/02/session-001/original.jpg",
        _jpeg_bytes(_solid((90, 90, 90), (960, 720))),
        "image/jpeg",
    )
    jev_omni = _ExplodingJevOmni()
    tier3 = _FixedTier3Adapter(
        [
            {
                "target_index": 0,
                "votes": [
                    {"role": "target_presence_fit", "label": "true", "confidence": 0.8, "reason": "car"},
                    {"role": "class_scene", "label": "true", "confidence": 0.8, "reason": "car"},
                    {"role": "adversarial", "label": "true", "confidence": 0.9, "reason": "clean"},
                ],
                "consensus": "true",
            },
            {
                "target_index": 1,
                "votes": [
                    {"role": "target_presence_fit", "label": "false", "confidence": 0.8, "reason": "background"},
                    {"role": "class_scene", "label": "false", "confidence": 0.8, "reason": "background"},
                    {"role": "adversarial", "label": "false", "confidence": 0.9, "reason": "empty"},
                ],
                "consensus": "false",
            },
        ]
    )
    app = _app(tmp_path, store=store, auto_adapter=tier3, jev_omni=jev_omni)
    client = TestClient(app)

    _ingest_and_run(client, app, "jev-omni-tier25-fallback-001")

    assert jev_omni.calls == 2  # 每个候选都尝试过 Jev-Omni
    assert len(tier3.calls) == 1  # 失败后整批回退到一次 Tier 3 调用

    with app.state.session_factory() as session:  # type: ignore[attr-defined]
        labels = sorted(
            annotation.label for annotation in session.scalars(select(Annotation)).all()
        )
    assert labels == ["false", "true"]
