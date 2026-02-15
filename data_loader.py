"""Data loading and cleaning for Section Attrition & Grade Report Excel files."""

import pandas as pd
import numpy as np

# Required columns that must be present for the app to function
REQUIRED_COLUMNS = [
    "Term Description",
    "Subject",
    "Catalog Number",
    "Section Number",
    "Official Class Enrollments",
    "Tot. # Grades used for DFW Rate Analysis",
    "Ds,Fs,Ws used for DFW Rate Analysis",
    "Dropped",
]

# Columns that are optional but used if present
OPTIONAL_COLUMNS = [
    "Repeats",
    "Incomplete (I)",
    "Extended Incomplete (EI)",
    "Permanent Incomplete (PI)",
    "Incomplete lapsed to F (@F)",
]

# All count columns that should be cast to numeric
COUNT_COLUMNS = [
    "Official Class Enrollments",
    "Tot. # Grades used for DFW Rate Analysis",
    "Ds,Fs,Ws used for DFW Rate Analysis",
    "Dropped",
    "Repeats",
    "Incomplete (I)",
    "Extended Incomplete (EI)",
    "Permanent Incomplete (PI)",
    "Incomplete lapsed to F (@F)",
]


def load_excel(file) -> tuple[pd.DataFrame, dict]:
    """Load Excel file and return raw dataframe plus validation info.

    Returns:
        (df, info) where info has keys: total_rows, missing_columns, present_columns
    """
    df = pd.read_excel(file, sheet_name="Export")
    info = {
        "total_rows": len(df),
        "missing_required": [c for c in REQUIRED_COLUMNS if c not in df.columns],
        "missing_optional": [c for c in OPTIONAL_COLUMNS if c not in df.columns],
        "present_columns": list(df.columns),
    }
    return df, info


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, int, pd.DataFrame]:
    """Normalize types, detect and remove rollup rows.

    Returns:
        (cleaned_df, rollup_count, removed_rows)
    """
    df = df.copy()

    # Normalize identifier columns to string
    for col in ["Catalog Number", "Section Number"]:
        if col in df.columns:
            df[col] = df[col].astype(str).str.strip()

    if "Subject" in df.columns:
        df["Subject"] = df["Subject"].astype(str).str.strip()

    # Cast count fields to numeric, NaN -> 0
    for col in COUNT_COLUMNS:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0).astype(float)

    # Add missing optional columns as zeros
    for col in OPTIONAL_COLUMNS:
        if col not in df.columns:
            df[col] = 0.0

    total_before = len(df)

    # Identify rollup rows (case-insensitive "Total" check)
    def _is_rollup_value(series):
        lower = series.str.lower()
        return (
            lower.isin(["total", "nan", "none", ""])
            | series.isna()
        )

    rollup_mask = (
        _is_rollup_value(df["Subject"])
        | _is_rollup_value(df["Catalog Number"])
        | _is_rollup_value(df["Section Number"])
    )

    df_clean = df[~rollup_mask].copy().reset_index(drop=True)
    removed_text = df[rollup_mask].copy().reset_index(drop=True)

    # Second pass: detect sum-matching rollup rows.
    # For each term/subject/catalog group, if a row's enrollment equals the sum
    # of the other rows in the group, it's a rollup (regardless of Section Number).
    df_clean, removed_sum = _remove_sum_matching_rollups(df_clean)

    removed = pd.concat([removed_text, removed_sum], ignore_index=True)
    rollup_count = total_before - len(df_clean)

    return df_clean, rollup_count, removed


def _remove_sum_matching_rollups(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Remove rows whose enrollment equals the sum of sibling rows in the same group.

    In datasets with embedded totals, the rollup row for a course-term has
    enrollment == sum(other sections' enrollments).  We detect and remove these.
    """
    group_cols = ["Term Description", "Subject", "Catalog Number"]
    enroll_col = "Official Class Enrollments"

    # Only check groups with 3+ rows (need at least 2 real sections + 1 total)
    group_sizes = df.groupby(group_cols).size()
    multi_groups = group_sizes[group_sizes >= 3].index

    if len(multi_groups) == 0:
        return df, pd.DataFrame(columns=df.columns)

    rollup_indices = []

    for keys in multi_groups:
        mask = (
            (df["Term Description"] == keys[0])
            & (df["Subject"] == keys[1])
            & (df["Catalog Number"] == keys[2])
        )
        group = df[mask]

        if len(group) < 3:
            continue

        group_total = group[enroll_col].sum()
        for idx, row in group.iterrows():
            others_sum = group_total - row[enroll_col]
            # If this row's enrollment equals the sum of all others, it's a rollup.
            # Use a small tolerance for floating point.
            if row[enroll_col] > 0 and abs(row[enroll_col] - others_sum) < 0.5:
                rollup_indices.append(idx)

    if not rollup_indices:
        return df, pd.DataFrame(columns=df.columns)

    removed = df.loc[rollup_indices].copy()
    df_clean = df.drop(index=rollup_indices).reset_index(drop=True)
    return df_clean, removed


def build_course_term_averages(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate section-level data to course-term level by averaging across sections.

    For each (Term, Subject, Catalog Number), computes:
    - mean of each section's rates (dfw_rate, drop_rate, etc.)
    - mean of each section's counts (enrollments, dfw_count, etc.)
    - number of sections contributing to the average

    Expects df to already have metrics computed via compute_metrics().
    """
    group_cols = ["Term Description", "Subject", "Catalog Number"]

    rate_cols = ["dfw_rate", "drop_rate", "incomplete_rate",
                 "lapsed_incomplete_rate", "repeat_rate"]
    count_cols = ["Official Class Enrollments", "dfw_count", "drop_count",
                  "incomplete_count", "lapsed_incomplete_count", "repeat_count"]

    # Average rates and counts across sections
    agg_rates = df.groupby(group_cols, as_index=False)[rate_cols].mean()
    agg_counts = df.groupby(group_cols, as_index=False)[count_cols].mean()
    section_counts = df.groupby(group_cols, as_index=False).size()
    section_counts = section_counts.rename(columns={"size": "num_sections"})

    result = agg_rates.merge(agg_counts, on=group_cols)
    result = result.merge(section_counts, on=group_cols)
    return result
