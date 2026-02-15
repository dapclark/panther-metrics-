"""Metric computation for student success analysis."""

import pandas as pd
import numpy as np


def compute_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Compute all rate metrics on a dataframe with count columns.

    Works for both section-level and course-term aggregated tables.
    """
    df = df.copy()

    enrollments = df["Official Class Enrollments"]
    grades_used = df["Tot. # Grades used for DFW Rate Analysis"]

    # DFW
    df["dfw_count"] = df["Ds,Fs,Ws used for DFW Rate Analysis"]
    df["dfw_rate"] = np.where(grades_used > 0, df["dfw_count"] / grades_used, np.nan)

    # Drops
    df["drop_count"] = df["Dropped"]
    df["drop_rate"] = np.where(enrollments > 0, df["drop_count"] / enrollments, np.nan)

    # Incompletes
    df["incomplete_count"] = (
        df["Incomplete (I)"]
        + df["Extended Incomplete (EI)"]
        + df["Permanent Incomplete (PI)"]
    )
    # Use grades_used as denominator; fallback to enrollments if grades_used is 0
    denom_inc = np.where(grades_used > 0, grades_used, enrollments)
    df["incomplete_rate"] = np.where(
        denom_inc > 0, df["incomplete_count"] / denom_inc, np.nan
    )

    # Lapsed incompletes
    df["lapsed_incomplete_count"] = df["Incomplete lapsed to F (@F)"]
    df["lapsed_incomplete_rate"] = np.where(
        denom_inc > 0, df["lapsed_incomplete_count"] / denom_inc, np.nan
    )

    # Repeats
    df["repeat_count"] = df["Repeats"]
    df["repeat_rate"] = np.where(
        enrollments > 0, df["repeat_count"] / enrollments, np.nan
    )

    return df
