"""Tests for pattern detection logic."""

import pandas as pd
import numpy as np
import pytest
from metrics import compute_metrics
from data_loader import build_course_term_averages
from patterns import detect_patterns


def _make_section_rows(dfw_rates, enrollments=30, grades_used=28,
                       section="001", subject="MATH", catalog="101"):
    """Helper to create section-level rows with specific DFW rates."""
    rows = []
    for i, rate in enumerate(dfw_rates):
        dfw_count = int(rate * grades_used)
        rows.append(
            {
                "Term Description": f"Spring {2020 + i}",
                "Subject": subject,
                "Catalog Number": catalog,
                "Section Number": section,
                "Official Class Enrollments": enrollments,
                "Tot. # Grades used for DFW Rate Analysis": grades_used,
                "Ds,Fs,Ws used for DFW Rate Analysis": dfw_count,
                "Dropped": 1,
                "Repeats": 0,
                "Incomplete (I)": 0,
                "Extended Incomplete (EI)": 0,
                "Permanent Incomplete (PI)": 0,
                "Incomplete lapsed to F (@F)": 0,
            }
        )
    return rows


def _build_course_avg(rows):
    """From raw section rows, compute metrics and build course-term averages."""
    df = pd.DataFrame(rows)
    df = compute_metrics(df)
    return build_course_term_averages(df)


def test_persistent_high_dfw():
    rows = _make_section_rows([0.25, 0.30, 0.22])
    ct = _build_course_avg(rows)
    flagged = detect_patterns(ct, dfw_threshold=0.20, consecutive_terms=2,
                              min_enrollments=1)
    assert len(flagged) == 1
    reasons = flagged.iloc[0]["reasons"]
    assert any("above 20%" in r and "in a row" in r for r in reasons)


def test_not_persistent_if_below():
    rows = _make_section_rows([0.10, 0.25, 0.10])
    ct = _build_course_avg(rows)
    flagged = detect_patterns(ct, dfw_threshold=0.20, consecutive_terms=2,
                              min_enrollments=1)
    if not flagged.empty:
        reasons = flagged.iloc[0]["reasons"]
        assert not any("in a row" in r for r in reasons)


def test_min_enrollment_filter():
    rows = _make_section_rows([0.50, 0.50, 0.50], enrollments=5)
    ct = _build_course_avg(rows)
    flagged = detect_patterns(ct, min_enrollments=20)
    assert len(flagged) == 0


def test_spike_detection():
    rows = _make_section_rows([0.05, 0.07, 0.04, 0.06, 0.50], grades_used=100)
    ct = _build_course_avg(rows)
    flagged = detect_patterns(ct, dfw_threshold=0.20, min_enrollments=1)
    assert not flagged.empty
    reasons = flagged.iloc[0]["reasons"]
    assert any("spike" in r for r in reasons)


def test_worsening_trend():
    rows = _make_section_rows([0.05, 0.10, 0.15, 0.20, 0.25])
    ct = _build_course_avg(rows)
    flagged = detect_patterns(ct, dfw_threshold=0.20, trend_terms=4,
                              min_enrollments=1)
    if not flagged.empty:
        reasons = flagged.iloc[0]["reasons"]
        assert any("rising" in r for r in reasons)


def test_averaging_across_sections():
    """Two sections in the same term should be averaged, not summed."""
    rows_001 = _make_section_rows([0.21, 0.21, 0.21], section="001")
    rows_002 = _make_section_rows([0.03, 0.03, 0.03], section="002")
    ct = _build_course_avg(rows_001 + rows_002)
    flagged = detect_patterns(ct, dfw_threshold=0.20, consecutive_terms=2,
                              min_enrollments=1)
    assert len(flagged) == 0


def test_averaging_both_high():
    """If both sections are high, course should be flagged."""
    rows_001 = _make_section_rows([0.30, 0.30, 0.30], section="001")
    rows_002 = _make_section_rows([0.25, 0.25, 0.25], section="002")
    ct = _build_course_avg(rows_001 + rows_002)
    flagged = detect_patterns(ct, dfw_threshold=0.20, consecutive_terms=2,
                              min_enrollments=1)
    assert len(flagged) == 1
    reasons = flagged.iloc[0]["reasons"]
    assert any("in a row" in r for r in reasons)


def test_reason_includes_threshold_values():
    """Reason strings should include the actual threshold values."""
    rows = _make_section_rows([0.25, 0.30, 0.22])
    ct = _build_course_avg(rows)
    flagged = detect_patterns(ct, dfw_threshold=0.15, consecutive_terms=2,
                              min_enrollments=1)
    assert len(flagged) == 1
    reasons = flagged.iloc[0]["reasons"]
    # Should mention the 15% threshold, not a hardcoded 20%
    assert any("15%" in r for r in reasons)


def test_empty_input():
    df = pd.DataFrame(
        columns=[
            "Term Description", "Subject", "Catalog Number",
            "Official Class Enrollments", "dfw_rate", "drop_rate",
            "incomplete_rate", "repeat_rate", "lapsed_incomplete_rate",
            "dfw_count", "drop_count", "incomplete_count", "repeat_count",
            "lapsed_incomplete_count", "num_sections",
        ]
    )
    flagged = detect_patterns(df)
    assert len(flagged) == 0
