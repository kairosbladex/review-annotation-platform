from __future__ import annotations

from pathlib import Path
import sys

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
        {"class_name": "person", "gold": "true", "prediction": "true", "score": 0.95, "parse_error": None},
        {"class_name": "person", "gold": "true", "prediction": "true", "score": 0.85, "parse_error": None},
        {"class_name": "person", "gold": "false", "prediction": "true", "score": 0.60, "parse_error": None},
        {"class_name": "person", "gold": "false", "prediction": "false", "score": 0.20, "parse_error": None},
        {"class_name": "road_crack", "gold": "true", "prediction": "true", "score": 0.90, "parse_error": None},
        {"class_name": "road_crack", "gold": "false", "prediction": "false", "score": 0.10, "parse_error": None},
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
