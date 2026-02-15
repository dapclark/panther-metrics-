"""Written analysis generation for flagged courses."""

import pandas as pd
import numpy as np
from patterns import _build_term_order


def generate_analysis(
    flagged_df: pd.DataFrame,
    course_term_df: pd.DataFrame,
    section_df: pd.DataFrame,
    settings: dict,
) -> str:
    """Generate a written analysis of flagged courses with recommendations.

    Returns a markdown string.
    """
    if flagged_df.empty:
        return (
            "## No Courses Flagged\n\n"
            "No courses met the criteria for flagging based on the current "
            "thresholds and selected data. Consider lowering thresholds or "
            "expanding the term/subject selection if you want a broader scan."
        )

    sections = []
    sections.append(_write_executive_summary(flagged_df, course_term_df, settings))
    sections.append(_write_priority_concerns(flagged_df, course_term_df, section_df))
    sections.append(_write_course_details(flagged_df, course_term_df, section_df))
    sections.append(_write_recommendations(flagged_df))

    return "\n\n---\n\n".join(sections)


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
        "",
    ]

    if persistent_count or trend_count or spike_count or multi_count:
        lines.append("**Breakdown by pattern type:**")
        lines.append("")
        if persistent_count:
            lines.append(
                f"- **{persistent_count}** with persistently high DFW rates "
                f"(above {settings['dfw_threshold']:.0%} for "
                f"{settings['consecutive_terms']}+ consecutive terms)"
            )
        if trend_count:
            lines.append(
                f"- **{trend_count}** with worsening DFW trends "
                f"(rates increasing over recent terms)"
            )
        if spike_count:
            lines.append(
                f"- **{spike_count}** with unusual spikes "
                f"(latest term significantly above the course's historical average)"
            )
        if multi_count:
            lines.append(
                f"- **{multi_count}** with multiple co-occurring issues "
                f"(e.g., high DFW combined with high drop or incomplete rates)"
            )

    # Worst average DFW
    worst = flagged_df.iloc[0]
    lines.extend([
        "",
        f"The highest average DFW rate among flagged courses is "
        f"**{worst['Subject']} {worst['Catalog Number']}** "
        f"at **{worst['latest_dfw_rate']:.1%}** in {worst['latest_term']}.",
    ])

    return "\n".join(lines)


def _write_priority_concerns(
    flagged_df: pd.DataFrame,
    course_term_df: pd.DataFrame,
    section_df: pd.DataFrame,
) -> str:
    """Identify the most urgent concerns."""
    lines = ["## Priority Concerns", ""]

    # Sort by severity: courses with multiple reasons first, then by DFW rate
    scored = flagged_df.copy()
    scored["n_reasons"] = scored["reasons"].apply(len)
    scored = scored.sort_values(
        ["n_reasons", "latest_dfw_rate"], ascending=[False, False]
    )

    top = scored.head(5)

    for rank, (_, row) in enumerate(top.iterrows(), 1):
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

        # Get the trend data for context
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

        # Check for section-level variation in latest term
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


def _write_course_details(
    flagged_df: pd.DataFrame,
    course_term_df: pd.DataFrame,
    section_df: pd.DataFrame,
) -> str:
    """Full listing of all flagged courses with context."""
    if len(flagged_df) <= 5:
        return ""  # Already covered in priority concerns

    remaining = flagged_df.iloc[5:]
    if remaining.empty:
        return ""

    lines = ["## Additional Flagged Courses", ""]

    for _, row in remaining.iterrows():
        course = f"{row['Subject']} {row['Catalog Number']}"
        reasons_short = "; ".join(
            _shorten_reason(r) for r in row["reasons"]
        )
        lines.append(
            f"- **{course}** — avg DFW {row['latest_dfw_rate']:.1%} "
            f"in {row['latest_term']}: {reasons_short}"
        )

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


def _write_recommendations(flagged_df: pd.DataFrame) -> str:
    """Generate actionable recommendations based on the patterns found."""
    lines = ["## Recommendations", ""]

    all_reasons = []
    for reasons in flagged_df["reasons"]:
        all_reasons.extend(reasons)

    has_persistent = any("in a row" in r for r in all_reasons)
    has_trend = any("rising" in r for r in all_reasons)
    has_spike = any("spike" in r for r in all_reasons)
    has_multi = any("Multiple issues" in r for r in all_reasons)

    # Check for section variation
    has_variation = any(
        "Section variation" in r for r in all_reasons
    )  # Not in reasons, but we'll add general advice

    rec_num = 1

    if has_persistent:
        lines.extend([
            f"**{rec_num}. Address courses with persistently high DFW rates**",
            "",
            "Courses that have maintained high DFW rates across multiple "
            "consecutive terms indicate a structural issue rather than a "
            "one-time problem. Consider:",
            "",
            "- Reviewing course prerequisites to ensure students are "
            "adequately prepared",
            "- Examining whether the course curriculum and assessments "
            "align with learning objectives",
            "- Consulting with instructors about pedagogy, grading "
            "practices, and student support strategies",
            "- Comparing section-level data to identify whether outcomes "
            "differ by instructor or modality",
            "- Exploring supplemental instruction, tutoring, or "
            "embedded support options",
            "",
        ])
        rec_num += 1

    if has_trend:
        lines.extend([
            f"**{rec_num}. Investigate courses with worsening trends**",
            "",
            "Rising DFW rates over recent terms may signal emerging "
            "problems that could worsen without intervention. Consider:",
            "",
            "- Determining whether changes to course content, instruction, "
            "or student demographics coincide with the trend",
            "- Checking whether recent terms reflect changes in "
            "course delivery (e.g., shift to online, new textbook, "
            "instructor turnover)",
            "- Engaging instructors in early-alert systems to identify "
            "struggling students sooner",
            "- Reviewing whether adequate academic support resources "
            "are available and promoted to students",
            "",
        ])
        rec_num += 1

    if has_spike:
        lines.extend([
            f"**{rec_num}. Follow up on courses with sudden spikes**",
            "",
            "A spike in DFW rate that is significantly above a course's "
            "historical average may reflect a one-time disruption or "
            "a new problem. Consider:",
            "",
            "- Determining whether the spike was caused by a specific event "
            "(instructor change, schedule change, pandemic-related disruption, "
            "unusual student cohort)",
            "- Monitoring the next term to see if rates return to normal "
            "or if the spike becomes a new baseline",
            "- If the spike persists, escalating to the interventions "
            "described above for persistent issues",
            "",
        ])
        rec_num += 1

    if has_multi:
        lines.extend([
            f"**{rec_num}. Prioritize courses with multiple co-occurring issues**",
            "",
            "When a course has high DFW rates alongside high drop rates "
            "or high incomplete rates, it suggests students are struggling "
            "in multiple ways. These courses may benefit from:",
            "",
            "- A holistic review involving the department, academic advising, "
            "and student success offices",
            "- Exploring whether students are dropping because the course "
            "is too difficult, poorly scheduled, or misaligned with "
            "their preparation",
            "- Reviewing incomplete policies — high incomplete rates "
            "may indicate students are disengaging rather than withdrawing",
            "",
        ])
        rec_num += 1

    lines.extend([
        f"**{rec_num}. General best practices**",
        "",
        "- Share this data with department chairs and program coordinators "
        "to foster data-informed conversations about student success",
        "- Use section-level drill-downs (available in the Flag Challenges tab) "
        "to distinguish between course-wide issues and section-specific ones",
        "- Track whether interventions lead to improvement by re-running "
        "this analysis in subsequent terms",
        "- Consider student voice — survey or focus-group data can "
        "provide context that quantitative data cannot",
    ])

    return "\n".join(lines)
