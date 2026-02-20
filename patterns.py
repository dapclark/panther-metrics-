"""Pattern detection logic for identifying problematic course outcomes.

Operates on course-term averages (mean of section-level rates per term).
"""

import pandas as pd
import numpy as np
from scipy import stats


RESULT_COLUMNS = [
    "Subject",
    "Catalog Number",
    "latest_dfw_rate",
    "latest_drop_rate",
    "latest_incomplete_rate",
    "latest_repeat_rate",
    "latest_enrollments",
    "avg_enrollments",
    "latest_num_sections",
    "reasons",
    "latest_term",
]


def detect_patterns(
    course_term_df: pd.DataFrame,
    dfw_threshold: float = 0.20,
    drop_threshold: float = 0.10,
    incomplete_threshold: float = 0.05,
    lapsed_threshold: float = 0.02,
    repeat_threshold: float = 0.10,
    min_enrollments: int = 20,
    consecutive_terms: int = 2,
    lookback_window: int = 6,
    trend_terms: int = 4,
    trend_slope_cutoff: float = 0.01,
    recency_terms: int = 4,
) -> pd.DataFrame:
    """Detect problematic patterns at the course level using section averages.

    Args:
        course_term_df: Course-term table with averaged section metrics.
        Other args: user-adjustable thresholds and pattern parameters.

    Returns:
        DataFrame with one row per flagged course.
    """
    df = course_term_df.copy()

    df = _sort_terms(df)

    # Filter by minimum average enrollments per section
    df = df[df["Official Class Enrollments"] >= min_enrollments].copy()

    if df.empty:
        return _empty_result()

    # Group by course
    courses = df.groupby(["Subject", "Catalog Number"])

    results = []
    for (subject, catalog), group in courses:
        group = group.sort_values("_term_order")
        reasons = []

        recent = group.tail(lookback_window)
        if recent.empty:
            continue

        latest = recent.iloc[-1]

        # 1. High DFW persistent
        persistent = _check_persistent(
            recent, "dfw_rate", dfw_threshold, consecutive_terms
        )
        if persistent:
            reasons.append(persistent)

        # 2. Worsening trend
        trend = _check_worsening_trend(
            recent, "dfw_rate", dfw_threshold, trend_terms, trend_slope_cutoff
        )
        if trend:
            reasons.append(trend)

        # 3. Spike anomaly
        spike = _check_spike(recent, "dfw_rate")
        if spike:
            reasons.append(spike)

        # 4. Co-occurring issues
        co_issues = _check_co_occurring(
            latest, dfw_threshold, drop_threshold, incomplete_threshold
        )
        if co_issues:
            reasons.append(co_issues)

        if reasons:
            results.append(
                {
                    "Subject": subject,
                    "Catalog Number": catalog,
                    "latest_dfw_rate": latest["dfw_rate"],
                    "latest_drop_rate": latest["drop_rate"],
                    "latest_incomplete_rate": latest["incomplete_rate"],
                    "latest_repeat_rate": latest.get("repeat_rate", np.nan),
                    "latest_enrollments": latest["Official Class Enrollments"],
                    "avg_enrollments": group["Official Class Enrollments"].mean(),
                    "latest_num_sections": latest.get("num_sections", np.nan),
                    "reasons": reasons,
                    "latest_term": latest["Term Description"],
                }
            )

    if not results:
        return _empty_result()

    # Filter out stale courses not offered recently
    term_order = _build_term_order(course_term_df["Term Description"].unique())
    if term_order:
        max_order = max(term_order.values())
        results = [
            r for r in results
            if max_order - term_order.get(r["latest_term"], 0) < recency_terms
        ]

    if not results:
        return _empty_result()

    result_df = pd.DataFrame(results)
    result_df = result_df.sort_values("latest_dfw_rate", ascending=False).reset_index(
        drop=True
    )
    return result_df


def _sort_terms(df: pd.DataFrame) -> pd.DataFrame:
    """Add a _term_order column for chronological sorting."""
    df = df.copy()
    term_order = _build_term_order(df["Term Description"].unique())
    df["_term_order"] = df["Term Description"].map(term_order)
    df = df.sort_values("_term_order")
    return df


def _build_term_order(terms) -> dict:
    """Build a sort-order mapping for term descriptions."""
    season_order = {"spring": 0, "summer": 1, "fall": 2, "winter": 3}

    def term_sort_key(term):
        parts = str(term).strip().split()
        if len(parts) >= 2:
            season = parts[0].lower()
            try:
                year = int(parts[-1])
            except ValueError:
                return (0, 0, term)
            return (year, season_order.get(season, 5), term)
        return (0, 0, term)

    sorted_terms = sorted(terms, key=term_sort_key)
    return {t: i for i, t in enumerate(sorted_terms)}


def _check_persistent(
    df: pd.DataFrame, rate_col: str, threshold: float, n_consecutive: int
) -> str | None:
    """Check if rate_col is >= threshold for at least n_consecutive terms.

    Returns a descriptive string if flagged, None otherwise.
    """
    values = df[rate_col].dropna().values
    if len(values) < n_consecutive:
        return None

    # Find the longest streak and the terms it spans
    best_streak = 0
    streak = 0
    for v in values:
        if v >= threshold:
            streak += 1
            best_streak = max(best_streak, streak)
        else:
            streak = 0

    if best_streak >= n_consecutive:
        return (
            f"Avg DFW rate above {threshold:.0%} for "
            f"{best_streak} terms in a row"
        )
    return None


def _check_worsening_trend(
    df: pd.DataFrame,
    rate_col: str,
    threshold: float,
    trend_terms: int,
    slope_cutoff: float,
) -> str | None:
    """Check if there's a positive slope over the last K terms.

    Returns a descriptive string if flagged, None otherwise.
    """
    recent = df.tail(trend_terms)
    values = recent[rate_col].dropna().values
    if len(values) < 3:
        return None

    x = np.arange(len(values))
    slope, _, _, _, _ = stats.linregress(x, values)

    latest = values[-1]
    if slope > slope_cutoff and latest >= threshold * 0.8:
        terms_used = len(values)
        first_term = recent.iloc[0]["Term Description"]
        last_term = recent.iloc[-1]["Term Description"]
        return (
            f"Avg DFW rate rising (+{slope:.1%}/term) "
            f"over the last {terms_used} terms "
            f"({first_term} to {last_term}), "
            f"latest at {latest:.1%}"
        )
    return None


def _check_spike(df: pd.DataFrame, rate_col: str) -> str | None:
    """Flag if latest term is > mean + 2*std within the lookback window.

    Returns a descriptive string if flagged, None otherwise.
    """
    values = df[rate_col].dropna().values
    if len(values) < 3:
        return None

    prior = values[:-1]
    mean = prior.mean()
    std = prior.std()
    if std == 0:
        return None

    latest = values[-1]
    if latest > mean + 2 * std:
        return (
            f"Latest avg DFW rate ({latest:.1%}) is a spike — "
            f"prior average was {mean:.1%} "
            f"(more than 2 std deviations above normal)"
        )
    return None


def _check_co_occurring(
    latest: pd.Series,
    dfw_threshold: float,
    drop_threshold: float,
    incomplete_threshold: float,
) -> str | None:
    """Check for co-occurring high DFW + high drop or high incomplete.

    Returns a descriptive string if flagged, None otherwise.
    """
    high_dfw = (
        not pd.isna(latest["dfw_rate"]) and latest["dfw_rate"] >= dfw_threshold
    )
    high_drop = (
        not pd.isna(latest["drop_rate"]) and latest["drop_rate"] >= drop_threshold
    )
    high_inc = (
        not pd.isna(latest["incomplete_rate"])
        and latest["incomplete_rate"] >= incomplete_threshold
    )

    parts = []
    if high_dfw:
        parts.append(f"DFW rate ({latest['dfw_rate']:.1%}) above {dfw_threshold:.0%}")
    if high_drop:
        parts.append(f"drop rate ({latest['drop_rate']:.1%}) above {drop_threshold:.0%}")
    if high_inc:
        parts.append(f"incomplete rate ({latest['incomplete_rate']:.1%}) above {incomplete_threshold:.0%}")

    if high_dfw and (high_drop or high_inc):
        return "Multiple issues in latest term: " + "; ".join(parts)
    return None


def _empty_result() -> pd.DataFrame:
    return pd.DataFrame(columns=RESULT_COLUMNS)
