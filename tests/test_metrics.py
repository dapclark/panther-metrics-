"""Tests for metrics module."""

import pandas as pd
import numpy as np
import pytest
from metrics import compute_metrics


def _make_df(**overrides):
    defaults = {
        "Official Class Enrollments": 30,
        "Tot. # Grades used for DFW Rate Analysis": 28,
        "Ds,Fs,Ws used for DFW Rate Analysis": 6,
        "Dropped": 2,
        "Repeats": 3,
        "Incomplete (I)": 1,
        "Extended Incomplete (EI)": 0,
        "Permanent Incomplete (PI)": 0,
        "Incomplete lapsed to F (@F)": 1,
    }
    defaults.update(overrides)
    return pd.DataFrame([defaults])


def test_dfw_rate():
    df = compute_metrics(_make_df())
    assert df.iloc[0]["dfw_rate"] == pytest.approx(6 / 28)


def test_drop_rate():
    df = compute_metrics(_make_df())
    assert df.iloc[0]["drop_rate"] == pytest.approx(2 / 30)


def test_incomplete_rate():
    df = compute_metrics(_make_df())
    # (1 + 0 + 0) / 28
    assert df.iloc[0]["incomplete_rate"] == pytest.approx(1 / 28)


def test_lapsed_incomplete_rate():
    df = compute_metrics(_make_df())
    assert df.iloc[0]["lapsed_incomplete_rate"] == pytest.approx(1 / 28)


def test_repeat_rate():
    df = compute_metrics(_make_df())
    assert df.iloc[0]["repeat_rate"] == pytest.approx(3 / 30)


def test_zero_grades_used_dfw_nan():
    df = compute_metrics(
        _make_df(**{"Tot. # Grades used for DFW Rate Analysis": 0})
    )
    assert pd.isna(df.iloc[0]["dfw_rate"])


def test_zero_enrollments_drop_nan():
    df = compute_metrics(_make_df(**{"Official Class Enrollments": 0}))
    assert pd.isna(df.iloc[0]["drop_rate"])


def test_incomplete_fallback_to_enrollments():
    df = compute_metrics(
        _make_df(
            **{
                "Tot. # Grades used for DFW Rate Analysis": 0,
                "Official Class Enrollments": 30,
                "Incomplete (I)": 3,
            }
        )
    )
    # Falls back to enrollments: 3/30
    assert df.iloc[0]["incomplete_rate"] == pytest.approx(3 / 30)
