"""Tests for data_loader module."""

import pandas as pd
import numpy as np
import pytest
from data_loader import clean_dataframe, build_course_term_averages, COUNT_COLUMNS
from metrics import compute_metrics


def _make_section_row(
    term="Fall 2023",
    subject="MATH",
    catalog="101",
    section="001",
    enrollments=30,
    grades_used=28,
    dfw=5,
    dropped=2,
    repeats=1,
    inc_i=0,
    inc_ei=0,
    inc_pi=0,
    lapsed=0,
):
    return {
        "Term Description": term,
        "Subject": subject,
        "Catalog Number": catalog,
        "Section Number": section,
        "Official Class Enrollments": enrollments,
        "Tot. # Grades used for DFW Rate Analysis": grades_used,
        "Ds,Fs,Ws used for DFW Rate Analysis": dfw,
        "Dropped": dropped,
        "Repeats": repeats,
        "Incomplete (I)": inc_i,
        "Extended Incomplete (EI)": inc_ei,
        "Permanent Incomplete (PI)": inc_pi,
        "Incomplete lapsed to F (@F)": lapsed,
    }


def test_rollup_removal_subject_total():
    rows = [
        _make_section_row(subject="MATH"),
        _make_section_row(subject="Total"),
    ]
    df = pd.DataFrame(rows)
    cleaned, rollup_count, _ = clean_dataframe(df)
    assert rollup_count == 1
    assert len(cleaned) == 1
    assert cleaned.iloc[0]["Subject"] == "MATH"


def test_rollup_removal_catalog_total():
    rows = [
        _make_section_row(catalog="101"),
        _make_section_row(catalog="Total"),
    ]
    df = pd.DataFrame(rows)
    cleaned, rollup_count, _ = clean_dataframe(df)
    assert rollup_count == 1
    assert len(cleaned) == 1


def test_rollup_removal_section_nan():
    rows = [
        _make_section_row(section="001"),
        _make_section_row(section=np.nan),
    ]
    df = pd.DataFrame(rows)
    cleaned, rollup_count, _ = clean_dataframe(df)
    assert rollup_count == 1
    assert len(cleaned) == 1


def test_no_rollups():
    rows = [
        _make_section_row(section="001"),
        _make_section_row(section="002"),
    ]
    df = pd.DataFrame(rows)
    cleaned, rollup_count, _ = clean_dataframe(df)
    assert rollup_count == 0
    assert len(cleaned) == 2


def test_numeric_coercion():
    rows = [_make_section_row()]
    df = pd.DataFrame(rows)
    df.loc[0, "Official Class Enrollments"] = "not a number"
    cleaned, _, _ = clean_dataframe(df)
    assert cleaned.iloc[0]["Official Class Enrollments"] == 0.0


def test_course_term_averaging():
    rows = [
        _make_section_row(section="001", enrollments=30, dfw=6, grades_used=28),
        _make_section_row(section="002", enrollments=20, dfw=4, grades_used=18),
    ]
    df = pd.DataFrame(rows)
    cleaned, _, _ = clean_dataframe(df)
    cleaned = compute_metrics(cleaned)
    ct = build_course_term_averages(cleaned)
    assert len(ct) == 1
    # Enrollments should be averaged: (30 + 20) / 2 = 25
    assert ct.iloc[0]["Official Class Enrollments"] == pytest.approx(25.0)
    # DFW rates: 6/28 = 0.2143, 4/18 = 0.2222 -> avg ~0.2183
    assert ct.iloc[0]["dfw_rate"] == pytest.approx((6/28 + 4/18) / 2)
    assert ct.iloc[0]["num_sections"] == 2


def test_sum_matching_rollup_removal():
    """A row whose enrollment = sum of siblings is detected as a rollup,
    even if its Section Number isn't 'Total'."""
    rows = [
        _make_section_row(section="001", enrollments=25, dfw=3, grades_used=24),
        _make_section_row(section="002", enrollments=25, dfw=4, grades_used=23),
        # This is an embedded total row — enrollment 50 = 25 + 25
        _make_section_row(section="099", enrollments=50, dfw=7, grades_used=47),
    ]
    df = pd.DataFrame(rows)
    cleaned, rollup_count, removed = clean_dataframe(df)
    assert rollup_count == 1
    assert len(cleaned) == 2
    assert set(cleaned["Section Number"]) == {"001", "002"}


def test_sum_matching_does_not_remove_real_sections():
    """Two sections with equal enrollment should NOT trigger false removal."""
    rows = [
        _make_section_row(section="001", enrollments=25, dfw=3),
        _make_section_row(section="002", enrollments=25, dfw=4),
    ]
    df = pd.DataFrame(rows)
    cleaned, rollup_count, _ = clean_dataframe(df)
    assert rollup_count == 0
    assert len(cleaned) == 2
