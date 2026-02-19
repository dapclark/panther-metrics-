"""Written analysis generation for flagged courses."""

import pandas as pd
import numpy as np
from patterns import _build_term_order


# Maps pattern keywords found in reason strings to recommended actions.
_PATTERN_ACTIONS = {
    "persistent": "Review prerequisites, pedagogy, and grading practices",
    "trend": "Investigate recent changes; engage early-alert systems",
    "spike": "Determine cause of spike; monitor next term",
    "co_occurring": "Holistic review with department and advising",
    "section_variation": "Compare section-level outcomes by instructor/modality",
}


def _reason_tags(reasons: list[str]) -> set[str]:
    """Return a set of pattern tags present in a course's reason strings."""
    tags = set()
    for r in reasons:
        if "in a row" in r:
            tags.add("persistent")
        if "rising" in r:
            tags.add("trend")
        if "spike" in r:
            tags.add("spike")
        if "Multiple issues" in r:
            tags.add("co_occurring")
    return tags


def _pick_action(tags: set[str]) -> str:
    """Choose the most urgent action given a set of pattern tags."""
    # Priority order: co-occurring > persistent > trend > spike > fallback
    priority = ["co_occurring", "persistent", "trend", "spike"]
    for tag in priority:
        if tag in tags:
            return _PATTERN_ACTIONS[tag]
    return "Review course data and discuss with department"


def _make_headline(row: pd.Series) -> str:
    """Generate a one-sentence headline from a flagged course row."""
    tags = _reason_tags(row["reasons"])
    rate = f"{row['latest_dfw_rate']:.0%}"
    term = row["latest_term"]

    if "persistent" in tags:
        # Extract streak length from reason text
        for r in row["reasons"]:
            if "in a row" in r:
                parts = r.split()
                for i, p in enumerate(parts):
                    if p == "terms":
                        return f"DFW rate at {rate} for {parts[i-1]} consecutive terms"
        return f"Persistently high DFW rate at {rate}"
    if "trend" in tags:
        return f"DFW rate rising, now at {rate} in {term}"
    if "spike" in tags:
        return f"DFW rate spiked to {rate} in {term}"
    if "co_occurring" in tags:
        return f"Multiple issues co-occurring at {rate} DFW in {term}"
    return f"DFW rate at {rate} in {term}"


def classify_severity(
    flagged_df: pd.DataFrame,
    course_term_df: pd.DataFrame,
    section_df: pd.DataFrame,
    settings: dict,
) -> dict:
    """Classify flagged courses into severity tiers.

    Returns a dict with keys 'immediate', 'moderate', 'watch', each containing
    a list of course dicts with severity info, headline, and recommended action.
    """
    if flagged_df.empty:
        return {"immediate": [], "moderate": [], "watch": []}

    threshold = settings["dfw_threshold"]
    results = {"immediate": [], "moderate": [], "watch": []}

    for _, row in flagged_df.iterrows():
        tags = _reason_tags(row["reasons"])
        n_reasons = len(row["reasons"])
        dfw = row["latest_dfw_rate"]

        # Check for section variation
        secs = section_df[
            (section_df["Subject"] == row["Subject"])
            & (section_df["Catalog Number"] == row["Catalog Number"])
            & (section_df["Term Description"] == row["latest_term"])
        ]
        has_section_var = False
        if len(secs) > 1:
            sec_rates = secs["dfw_rate"].dropna()
            if len(sec_rates) > 1 and sec_rates.std() > 0.05:
                has_section_var = True
                tags.add("section_variation")

        # Classify severity
        if n_reasons >= 2 or "persistent" in tags or dfw > threshold * 1.5:
            severity = "immediate"
        elif "trend" in tags or "spike" in tags:
            severity = "moderate"
        else:
            severity = "watch"

        entry = {
            "course": f"{row['Subject']} {row['Catalog Number']}",
            "severity": severity,
            "dfw_rate": dfw,
            "term": row["latest_term"],
            "headline": _make_headline(row),
            "action": _pick_action(tags),
            "reasons": row["reasons"],
            "has_section_variation": has_section_var,
        }
        results[severity].append(entry)

    return results


def generate_analysis(
    flagged_df: pd.DataFrame,
    course_term_df: pd.DataFrame,
    section_df: pd.DataFrame,
    settings: dict,
) -> str:
    """Generate a written analysis of flagged courses with recommendations.

    Returns a markdown string with actions first, then summary, then detail.
    """
    if flagged_df.empty:
        return (
            "## No Courses Flagged\n\n"
            "No courses met the criteria for flagging based on the current "
            "thresholds and selected data. Consider lowering thresholds or "
            "expanding the term/subject selection if you want a broader scan."
        )

    tiers = classify_severity(flagged_df, course_term_df, section_df, settings)

    sections = []
    sections.append(_write_actions(tiers))
    sections.append(_write_executive_summary(flagged_df, course_term_df, settings))
    sections.append(_write_supporting_detail(flagged_df, course_term_df, section_df))

    return "\n\n---\n\n".join(s for s in sections if s)


def _write_actions(tiers: dict) -> str:
    """Recommended actions section, organized by severity tier."""
    lines = ["## Recommended Actions", ""]

    tier_labels = [
        ("immediate", "Act Now"),
        ("moderate", "Monitor Closely"),
        ("watch", "Awareness"),
    ]

    for tier_key, label in tier_labels:
        courses = tiers[tier_key]
        if not courses:
            continue
        lines.append(f"### {label} ({len(courses)})")
        lines.append("")
        for c in courses:
            lines.append(f"- **{c['course']}**: {c['headline']}")
            lines.append(f"  - *Action:* {c['action']}")
            if c.get("has_section_variation"):
                lines.append(
                    f"  - *Note:* {_PATTERN_ACTIONS['section_variation']}"
                )
        lines.append("")

    return "\n".join(lines)


def _write_executive_summary(
    flagged_df: pd.DataFrame,
    course_term_df: pd.DataFrame,
    settings: dict,
) -> str:
    """High-level overview."""
    n_flagged = len(flagged_df)
    n_total = course_term_df[["Subject", "Catalog Number"]].drop_duplicates().shape[0]
    pct = n_flagged / n_total * 100 if n_total > 0 else 0

    # Count by reason type
    all_reasons = []
    for reasons in flagged_df["reasons"]:
        all_reasons.extend(reasons)

    persistent_count = sum(1 for r in all_reasons if "in a row" in r)
    trend_count = sum(1 for r in all_reasons if "rising" in r)
    spike_count = sum(1 for r in all_reasons if "spike" in r)
    multi_count = sum(1 for r in all_reasons if "Multiple issues" in r)

    lines = [
        "## Executive Summary",
        "",
        f"Out of **{n_total} courses** analyzed, **{n_flagged}** "
        f"({pct:.0f}%) were flagged for student success concerns.",
    ]

    if persistent_count or trend_count or spike_count or multi_count:
        lines.append("")
        if persistent_count:
            lines.append(
                f"- **{persistent_count}** with persistently high DFW rates"
            )
        if trend_count:
            lines.append(
                f"- **{trend_count}** with worsening DFW trends"
            )
        if spike_count:
            lines.append(
                f"- **{spike_count}** with unusual spikes"
            )
        if multi_count:
            lines.append(
                f"- **{multi_count}** with multiple co-occurring issues"
            )

    return "\n".join(lines)


def _write_supporting_detail(
    flagged_df: pd.DataFrame,
    course_term_df: pd.DataFrame,
    section_df: pd.DataFrame,
) -> str:
    """Merged priority concerns and course details as supporting reference."""
    lines = ["## Supporting Detail", ""]

    scored = flagged_df.copy()
    scored["n_reasons"] = scored["reasons"].apply(len)
    scored = scored.sort_values(
        ["n_reasons", "latest_dfw_rate"], ascending=[False, False]
    )

    for rank, (_, row) in enumerate(scored.iterrows(), 1):
        course = f"{row['Subject']} {row['Catalog Number']}"
        lines.append(f"### {rank}. {course}")
        lines.append("")
        lines.append(
            f"- **Latest avg DFW rate:** {row['latest_dfw_rate']:.1%} "
            f"({row['latest_term']})"
        )
        if not pd.isna(row.get("latest_num_sections")):
            lines.append(
                f"- **Sections:** {row['latest_num_sections']:.0f} "
                f"(avg {row['latest_enrollments']:.0f} students per section)"
            )

        ct = course_term_df[
            (course_term_df["Subject"] == row["Subject"])
            & (course_term_df["Catalog Number"] == row["Catalog Number"])
        ]
        if not ct.empty:
            term_order = _build_term_order(ct["Term Description"].unique())
            ct = ct.copy()
            ct["_order"] = ct["Term Description"].map(term_order)
            ct = ct.sort_values("_order")
            historical_mean = ct["dfw_rate"].mean()
            lines.append(
                f"- **Historical avg DFW rate:** {historical_mean:.1%} "
                f"(across {len(ct)} terms)"
            )

        lines.append("- **Findings:**")
        for reason in row["reasons"]:
            lines.append(f"  - {reason}")

        secs = section_df[
            (section_df["Subject"] == row["Subject"])
            & (section_df["Catalog Number"] == row["Catalog Number"])
            & (section_df["Term Description"] == row["latest_term"])
        ]
        if len(secs) > 1:
            sec_rates = secs["dfw_rate"].dropna()
            if len(sec_rates) > 1 and sec_rates.std() > 0.05:
                low = sec_rates.min()
                high = sec_rates.max()
                lines.append(
                    f"- **Section variation:** DFW rates range from "
                    f"{low:.1%} to {high:.1%} across sections, "
                    f"suggesting the issue may be concentrated in "
                    f"specific sections rather than course-wide."
                )

        lines.append("")

    return "\n".join(lines)


def _shorten_reason(reason: str) -> str:
    """Create a brief version of a reason for the summary list."""
    if "in a row" in reason:
        return "persistent high DFW"
    if "rising" in reason:
        return "worsening trend"
    if "spike" in reason:
        return "recent spike"
    if "Multiple issues" in reason:
        return "multiple co-occurring issues"
    return reason
