from __future__ import annotations

from pathlib import Path
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "scripts"))

from qwen3vl_verify import parse_strict_decision, summarize_predictions


def test_strict_json_parser_fails_closed_to_uncertain():
    assert parse_strict_decision('{"decision":"CONFIRMED"}') == ("true", None)
    assert parse_strict_decision('{"decision":"REJECTED"}') == ("false", None)
    assert parse_strict_decision('{"decision":"UNCERTAIN"}') == (
        "uncertain",
        "unsupported_decision",
    )
    assert parse_strict_decision("CONFIRMED") == ("uncertain", "invalid_json")
    assert parse_strict_decision('{"decision":"CONFIRMED","reason":"x"}') == (
        "uncertain",
        "unexpected_json_fields",
    )
    assert parse_strict_decision('{"decision":"confirmed"}') == (
        "uncertain",
        "unsupported_decision",
    )
    assert parse_strict_decision(
        '{"decision":"REJECTED","decision":"CONFIRMED"}'
    ) == ("uncertain", "duplicate_json_key")


def test_summary_reports_exact_risk_coverage_and_per_class_metrics():
    rows = [
        {"class_name": "车", "gold": "true", "prediction": "true", "parse_error": None},
        {"class_name": "车", "gold": "false", "prediction": "true", "parse_error": None},
        {"class_name": "人员", "gold": "true", "prediction": "false", "parse_error": None},
        {"class_name": "人员", "gold": "false", "prediction": "false", "parse_error": None},
        {"class_name": "人员", "gold": "true", "prediction": "uncertain", "parse_error": "invalid_json"},
        {"class_name": "人员", "gold": "false", "prediction": "uncertain", "parse_error": None},
    ]

    summary = summarize_predictions(rows)

    assert summary["overall"] == {
        "n": 6,
        "correct": 2,
        "accuracy": 2 / 6,
        "coverage_n": 4,
        "coverage": 4 / 6,
        "actual_positive_n": 3,
        "actual_negative_n": 3,
        "selective_accuracy": 2 / 4,
        "far": 1 / 3,
        "frr": 1 / 3,
        "false_accept_rate_actual_negative": 1 / 3,
        "false_reject_rate_actual_positive": 1 / 3,
        "manual_review_rate": 2 / 6,
        "confusion": {
            "tp": 1,
            "tn": 1,
            "fp": 1,
            "fn": 1,
            "true_uncertain": 1,
            "false_uncertain": 1,
        },
    }
    assert summary["by_class"]["车"]["far"] == 1.0
    assert summary["by_class"]["人员"]["manual_review_rate"] == 0.5
    assert summary["parse_failures"] == 1


def test_zero_denominator_risk_rates_are_undefined_not_perfect():
    summary = summarize_predictions(
        [
            {
                "class_name": "车",
                "gold": "true",
                "prediction": "true",
                "parse_error": None,
            }
        ]
    )["overall"]

    assert summary["actual_negative_n"] == 0
    assert summary["far"] is None
    assert summary["false_accept_rate_actual_negative"] is None


def test_by_class_table_and_threshold_search_for_target_far():
    from qwen3vl_verify import (
        find_optimal_threshold_for_far,
        format_by_class_markdown_table,
    )

    rows = [
        {"class_name": "person", "gold": "true", "prediction": "true", "score": 0.95, "score_is_p_true": True, "parse_error": None},
        {"class_name": "person", "gold": "true", "prediction": "true", "score": 0.85, "score_is_p_true": True, "parse_error": None},
        {"class_name": "person", "gold": "false", "prediction": "true", "score": 0.60, "score_is_p_true": True, "parse_error": None},
        {"class_name": "person", "gold": "false", "prediction": "false", "score": 0.20, "score_is_p_true": True, "parse_error": None},
        {"class_name": "road_crack", "gold": "true", "prediction": "true", "score": 0.90, "score_is_p_true": True, "parse_error": None},
        {"class_name": "road_crack", "gold": "false", "prediction": "false", "score": 0.10, "score_is_p_true": True, "parse_error": None},
    ]
    summary = summarize_predictions(rows)
    assert len(summary["by_class_table"]) == 2
    md = format_by_class_markdown_table(summary)
    assert "| person |" in md
    assert "| road_crack |" in md

    search = find_optimal_threshold_for_far(rows, max_far=0.05)
    assert search["best_metrics"]["meets_far_constraint"] is True
    assert search["best_metrics"]["far"] == 0.0
    assert 0.60 < search["recommended_threshold"] <= 0.85


def test_high_confidence_rejection_is_never_flipped_into_an_acceptance():
    """回归：prediction="false", confidence=0.95 代表 P(true)=0.05，不能按 0.95 判真。"""
    from qwen3vl_verify import _resolve_positive_score, apply_confidence_threshold

    rejected = {"gold": "false", "prediction": "false", "confidence": 0.95}
    confirmed = {"gold": "true", "prediction": "true", "confidence": 0.95}

    assert _resolve_positive_score(rejected) == pytest.approx(0.05)
    assert _resolve_positive_score(confirmed) == pytest.approx(0.95)

    adjusted = apply_confidence_threshold([rejected, confirmed], 0.90)
    assert adjusted[0]["prediction"] == "false"
    assert adjusted[0]["correct"] is True
    assert adjusted[1]["prediction"] == "true"
    assert adjusted[1]["correct"] is True


def test_positive_score_explicit_fields_win_over_the_confidence_heuristic():
    from qwen3vl_verify import _resolve_positive_score

    assert _resolve_positive_score(
        {"prediction": "false", "confidence": 0.9, "positive_score": 0.31}
    ) == pytest.approx(0.31)
    assert _resolve_positive_score(
        {"prediction": "false", "confidence": 0.9, "p_true": 0.27}
    ) == pytest.approx(0.27)
    # An explicitly declared P(true) score must not be inverted again.
    assert _resolve_positive_score(
        {"prediction": "false", "score": 0.2, "score_is_p_true": True}
    ) == pytest.approx(0.2)
    assert _resolve_positive_score({"prediction": "true", "confidence": 0.7}) == pytest.approx(0.7)
    assert _resolve_positive_score({"prediction": "true"}) is None


def test_threshold_search_on_raw_records_recovers_a_usable_threshold():
    """回归：原始记录（含高置信 false）直传时不得退化成 0.99。"""
    from qwen3vl_verify import find_optimal_threshold_for_far

    raw_rows = [
        {"class_name": "车", "gold": "true", "prediction": "true", "confidence": 0.95},
        {"class_name": "车", "gold": "true", "prediction": "true", "confidence": 0.90},
        {"class_name": "车", "gold": "false", "prediction": "false", "confidence": 0.95},
        {"class_name": "车", "gold": "false", "prediction": "false", "confidence": 0.60},
        {"class_name": "人员", "gold": "false", "prediction": "true", "confidence": 0.80},
    ]

    search = find_optimal_threshold_for_far(raw_rows, max_far=0.05)

    assert search["recommended_threshold"] < 0.99
    # The search keeps the lowest threshold that still reaches the best F1.
    assert search["recommended_threshold"] == pytest.approx(0.85)
    assert search["best_metrics"]["f1"] == pytest.approx(1.0)
    assert search["best_metrics"]["far"] == 0.0
