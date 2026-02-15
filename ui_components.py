"""Reusable UI components for the Streamlit app."""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from patterns import _build_term_order


def render_thresholds_sidebar() -> dict:
    """Render the sidebar with thresholds, catalog search, and pattern controls."""
    st.sidebar.header("Search")

    catalog_search = st.sidebar.text_input(
        "Catalog Number (search)", "", help="Filter by catalog number (contains)"
    )

    st.sidebar.header("Thresholds")

    min_enrollments = st.sidebar.number_input(
        "Min Avg Enrollments per Section", value=20, min_value=1, step=5
    )

    dfw_threshold = st.sidebar.slider(
        "DFW Rate Threshold", 0.0, 1.0, 0.20, 0.01, format="%.2f"
    )
    drop_threshold = st.sidebar.slider(
        "Drop Rate Threshold", 0.0, 1.0, 0.10, 0.01, format="%.2f"
    )
    incomplete_threshold = st.sidebar.slider(
        "Incomplete Rate Threshold", 0.0, 1.0, 0.05, 0.01, format="%.2f"
    )
    lapsed_threshold = st.sidebar.slider(
        "Lapsed Incomplete Threshold", 0.0, 1.0, 0.02, 0.01, format="%.2f"
    )
    repeat_threshold = st.sidebar.slider(
        "Repeat Rate Threshold", 0.0, 1.0, 0.10, 0.01, format="%.2f"
    )

    st.sidebar.header("Pattern Detection")

    consecutive_terms = st.sidebar.number_input(
        "Consecutive terms above threshold", value=2, min_value=1, max_value=20
    )
    lookback_window = st.sidebar.number_input(
        "Lookback window (terms)", value=6, min_value=2, max_value=50
    )
    trend_terms = st.sidebar.number_input(
        "Trend detection terms", value=4, min_value=2, max_value=20
    )

    return {
        "catalog_search": catalog_search,
        "min_enrollments": min_enrollments,
        "dfw_threshold": dfw_threshold,
        "drop_threshold": drop_threshold,
        "incomplete_threshold": incomplete_threshold,
        "lapsed_threshold": lapsed_threshold,
        "repeat_threshold": repeat_threshold,
        "consecutive_terms": consecutive_terms,
        "lookback_window": lookback_window,
        "trend_terms": trend_terms,
    }


def apply_filters(df: pd.DataFrame, filters: dict) -> pd.DataFrame:
    """Apply term, subject, and catalog filters to a dataframe."""
    filtered = df.copy()

    if filters["selected_terms"]:
        filtered = filtered[
            filtered["Term Description"].isin(filters["selected_terms"])
        ]

    if filters["selected_subjects"]:
        filtered = filtered[filtered["Subject"].isin(filters["selected_subjects"])]

    if filters.get("catalog_search"):
        filtered = filtered[
            filtered["Catalog Number"]
            .str.contains(filters["catalog_search"], case=False, na=False)
        ]

    return filtered


def render_flagged_table(flagged_df: pd.DataFrame) -> str | None:
    """Render the flagged courses table. Returns selected course key or None."""
    if flagged_df.empty:
        st.info("No courses flagged with current thresholds and filters.")
        return None

    st.subheader(f"Flagged Courses ({len(flagged_df)})")

    display = flagged_df.copy()
    display["reasons_str"] = display["reasons"].apply(lambda r: ", ".join(r))

    rate_cols = [
        "latest_dfw_rate",
        "latest_drop_rate",
        "latest_incomplete_rate",
        "latest_repeat_rate",
    ]
    for col in rate_cols:
        display[col] = display[col].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )
    display["avg_enrollments"] = display["avg_enrollments"].apply(
        lambda x: f"{x:.1f}"
    )
    display["latest_enrollments"] = display["latest_enrollments"].apply(
        lambda x: f"{x:.1f}"
    )
    display["latest_num_sections"] = display["latest_num_sections"].apply(
        lambda x: f"{x:.0f}" if pd.notna(x) else "N/A"
    )

    display_cols = {
        "Subject": "Subject",
        "Catalog Number": "Catalog #",
        "reasons_str": "Reasons Flagged",
        "latest_term": "Latest Term",
        "latest_num_sections": "Sections",
        "latest_dfw_rate": "Avg DFW Rate",
        "latest_drop_rate": "Avg Drop Rate",
        "latest_incomplete_rate": "Avg Incomplete Rate",
        "latest_repeat_rate": "Avg Repeat Rate",
        "latest_enrollments": "Avg Enrollment/Section",
        "avg_enrollments": "Avg Enrollment/Section (All Terms)",
    }

    st.dataframe(
        display[list(display_cols.keys())].rename(columns=display_cols),
        use_container_width=True,
        hide_index=True,
    )

    # Course selector for drill-down
    course_options = list(dict.fromkeys(
        f"{row['Subject']} {row['Catalog Number']}"
        for _, row in flagged_df.iterrows()
    ))
    if course_options:
        selected = st.selectbox(
            "Select a course for drill-down",
            [""] + course_options,
            index=0,
        )
        if selected:
            return selected

    return None


def render_course_drilldown(
    course_key: str,
    course_term_df: pd.DataFrame,
    section_df: pd.DataFrame,
    dfw_threshold: float,
    drop_threshold: float,
    incomplete_threshold: float,
):
    """Render the drill-down view for a selected course."""
    parts = course_key.split(" ", 1)
    subject, catalog = parts[0], parts[1]

    st.subheader(f"Course Detail: {subject} {catalog}")

    # Course-term averages for this course
    ct = course_term_df[
        (course_term_df["Subject"] == subject)
        & (course_term_df["Catalog Number"] == catalog)
    ].copy()

    if ct.empty:
        st.warning("No data found for this course in the filtered dataset.")
        return

    # Sort by term order
    term_order = _build_term_order(ct["Term Description"].unique())
    ct["_order"] = ct["Term Description"].map(term_order)
    ct = ct.sort_values("_order")

    # Trend chart — course averages over terms
    _render_trend_chart(ct, dfw_threshold, drop_threshold, incomplete_threshold)

    # Term-by-term averages table
    st.markdown("**Term-by-Term Section Averages**")
    term_display = ct[
        [
            "Term Description",
            "num_sections",
            "Official Class Enrollments",
            "dfw_count",
            "dfw_rate",
            "drop_count",
            "drop_rate",
            "incomplete_count",
            "incomplete_rate",
            "repeat_count",
            "repeat_rate",
        ]
    ].copy()
    term_display = term_display.rename(columns={
        "num_sections": "Sections",
        "Official Class Enrollments": "Avg Enrollment",
        "dfw_count": "Avg DFW Count",
        "drop_count": "Avg Drop Count",
        "incomplete_count": "Avg Incomplete Count",
        "repeat_count": "Avg Repeat Count",
    })
    for col in ["dfw_rate", "drop_rate", "incomplete_rate", "repeat_rate"]:
        term_display[col] = term_display[col].apply(
            lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
        )
    for col in ["Avg Enrollment", "Avg DFW Count", "Avg Drop Count",
                 "Avg Incomplete Count", "Avg Repeat Count"]:
        term_display[col] = term_display[col].apply(lambda x: f"{x:.1f}")
    st.dataframe(term_display, use_container_width=True, hide_index=True)

    # Section-level breakdown for a selected term
    st.divider()
    latest_term = ct.iloc[-1]["Term Description"]
    section_terms = sorted(
        section_df[
            (section_df["Subject"] == subject)
            & (section_df["Catalog Number"] == catalog)
        ]["Term Description"].unique(),
        key=lambda t: term_order.get(t, 0),
    )

    if section_terms:
        selected_term = st.selectbox(
            "View individual sections for term:",
            section_terms,
            index=section_terms.index(latest_term)
            if latest_term in section_terms
            else len(section_terms) - 1,
            key="section_term_select",
        )

        sections = section_df[
            (section_df["Subject"] == subject)
            & (section_df["Catalog Number"] == catalog)
            & (section_df["Term Description"] == selected_term)
        ].copy()

        st.markdown(
            f"**Individual Sections for {selected_term}** "
            f"({len(sections)} sections)"
        )

        section_display = sections[
            [
                "Section Number",
                "Official Class Enrollments",
                "dfw_count",
                "dfw_rate",
                "drop_count",
                "drop_rate",
                "incomplete_rate",
                "repeat_rate",
            ]
        ].copy()
        section_display = section_display.sort_values("dfw_rate", ascending=False)
        for col in ["dfw_rate", "drop_rate", "incomplete_rate", "repeat_rate"]:
            section_display[col] = section_display[col].apply(
                lambda x: f"{x:.1%}" if pd.notna(x) else "N/A"
            )
        st.dataframe(section_display, use_container_width=True, hide_index=True)


def _render_trend_chart(
    ct: pd.DataFrame,
    dfw_threshold: float,
    drop_threshold: float,
    incomplete_threshold: float,
):
    """Render a plotly trend chart for course-level averages over terms."""
    show_secondary = st.checkbox("Show drop & incomplete rates", value=False)

    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=ct["Term Description"],
            y=ct["dfw_rate"],
            mode="lines+markers",
            name="Avg DFW Rate",
            line=dict(color="#e74c3c", width=3),
        )
    )

    fig.add_hline(
        y=dfw_threshold,
        line_dash="dash",
        line_color="#e74c3c",
        opacity=0.5,
        annotation_text=f"DFW Threshold ({dfw_threshold:.0%})",
    )

    if show_secondary:
        fig.add_trace(
            go.Scatter(
                x=ct["Term Description"],
                y=ct["drop_rate"],
                mode="lines+markers",
                name="Avg Drop Rate",
                line=dict(color="#3498db", width=2),
            )
        )
        fig.add_trace(
            go.Scatter(
                x=ct["Term Description"],
                y=ct["incomplete_rate"],
                mode="lines+markers",
                name="Avg Incomplete Rate",
                line=dict(color="#f39c12", width=2),
            )
        )

    # Number of sections as bar on secondary axis
    fig.add_trace(
        go.Bar(
            x=ct["Term Description"],
            y=ct["num_sections"],
            name="Sections",
            yaxis="y2",
            opacity=0.2,
            marker_color="#95a5a6",
        )
    )

    fig.update_layout(
        title="Course Average Trend Over Terms",
        yaxis=dict(title="Avg Rate (across sections)", tickformat=".0%",
                    rangemode="tozero"),
        yaxis2=dict(
            title="Number of Sections",
            overlaying="y",
            side="right",
            rangemode="tozero",
        ),
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=400,
    )

    st.plotly_chart(fig, use_container_width=True)


def render_exports(
    flagged_df: pd.DataFrame,
    course_term_df: pd.DataFrame,
    section_df: pd.DataFrame,
    selected_course: str | None,
):
    """Render export buttons."""
    st.subheader("Exports")
    col1, col2, col3 = st.columns(3)

    with col1:
        if not flagged_df.empty:
            export_flagged = flagged_df.copy()
            export_flagged["reasons"] = export_flagged["reasons"].apply(
                lambda r: "; ".join(r)
            )
            st.download_button(
                "Download Flagged Courses (CSV)",
                export_flagged.to_csv(index=False),
                "flagged_courses.csv",
                "text/csv",
            )

    with col2:
        export_cols = [
            c for c in course_term_df.columns
            if c not in ("_term_order", "_order")
        ]
        st.download_button(
            "Download Course-Term Averages (CSV)",
            course_term_df[export_cols].to_csv(index=False),
            "course_term_averages.csv",
            "text/csv",
        )

    with col3:
        if selected_course:
            parts = selected_course.split(" ", 1)
            subject, catalog = parts[0], parts[1]
            sec_data = section_df[
                (section_df["Subject"] == subject)
                & (section_df["Catalog Number"] == catalog)
            ]
            export_cols_sec = [
                c for c in sec_data.columns if c not in ("_term_order", "_order")
            ]
            st.download_button(
                f"Download All Sections for {selected_course} (CSV)",
                sec_data[export_cols_sec].to_csv(index=False),
                f"sections_{subject}_{catalog}.csv",
                "text/csv",
            )


def render_methodology():
    """Render an expandable section explaining how metrics are computed."""
    with st.expander("How metrics are computed"):
        st.markdown(
            """
**Section-Level Metrics** (computed per section first):

- **DFW Rate** = (Ds + Fs + Ws) / Total Grades Used for DFW Rate Analysis
- **Drop Rate** = Dropped / Official Class Enrollments
- **Incomplete Rate** = (Incomplete + Extended Incomplete + Permanent Incomplete) / Grades Used (fallback: / Enrollments)
- **Lapsed Incomplete Rate** = Incomplete lapsed to F (@F) / Grades Used (same fallback)
- **Repeat Rate** = Repeats / Official Class Enrollments

**Course-Term Averages** (how courses are compared across terms):

For each course in each term, the section-level rates and counts are **averaged** across all sections offered that term. This gives the typical outcome for a section of that course.

**Pattern Detection** (applied to course-term averages):

- **Persistent High DFW**: Average DFW rate >= threshold for N consecutive terms
- **Worsening Trend**: Positive slope (> 0.01/term) over last K terms, with latest near/above threshold
- **Spike**: Latest term average DFW rate > mean + 2*std of prior terms
- **Co-occurring Issues**: High average DFW combined with high average drop or incomplete rate

**Rollup Row Removal**: Rows where Subject, Catalog Number, or Section Number is "Total", blank, or NaN are excluded. Only individual section rows are used.
"""
        )
