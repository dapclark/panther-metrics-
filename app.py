"""Panther Metrics — Student Success Patterns Analyzer.

A Streamlit app for analyzing multi-year, course-level student success outcomes
from Section Attrition & Grade Report Excel files.
"""

import streamlit as st
import pandas as pd

from data_loader import load_excel, clean_dataframe, build_course_term_averages
from metrics import compute_metrics
from patterns import detect_patterns, _build_term_order
from analysis import generate_analysis
from auth import is_authenticated, is_admin, render_login_page, render_logout_button
from ui_components import (
    _load_settings,
    render_thresholds_readonly,
    render_thresholds_admin,
    apply_filters,
    render_flagged_table,
    render_course_drilldown,
    render_exports,
    render_methodology,
)

st.set_page_config(
    page_title="Panther Metrics",
    page_icon="📊",
    layout="wide",
)

if not is_authenticated():
    render_login_page()
    st.stop()

render_logout_button()

st.title("Panther Metrics")
st.caption("Student Success Patterns Analyzer")


def _load_and_filter(settings):
    """Retrieve data from session state, apply filters, and detect patterns.

    Returns (filtered_ct, filtered_section, flagged, settings) or None
    if data is not ready.
    """
    if "course_term_df" not in st.session_state:
        return None

    section_df = st.session_state["section_df"]
    course_term_df = st.session_state["course_term_df"]
    selected_terms = st.session_state.get("selected_terms", [])
    selected_subjects = st.session_state.get("selected_subjects", [])

    filters = {
        "selected_terms": selected_terms,
        "selected_subjects": selected_subjects,
        "catalog_search": settings["catalog_search"],
    }

    filtered_ct = apply_filters(course_term_df, filters)
    filtered_section = apply_filters(section_df, filters)

    if filtered_ct.empty:
        return None

    flagged = detect_patterns(
        filtered_ct,
        dfw_threshold=settings["dfw_threshold"],
        drop_threshold=settings["drop_threshold"],
        incomplete_threshold=settings["incomplete_threshold"],
        lapsed_threshold=settings["lapsed_threshold"],
        repeat_threshold=settings["repeat_threshold"],
        min_enrollments=settings["min_enrollments"],
        consecutive_terms=settings["consecutive_terms"],
        lookback_window=settings["lookback_window"],
        trend_terms=settings["trend_terms"],
    )

    return filtered_ct, filtered_section, flagged


def main():
    if is_admin():
        tab_data, tab_flag, tab_analysis, tab_admin = st.tabs(
            ["Data", "Flagged Courses", "Analysis", "Admin"]
        )
    else:
        tab_data, tab_flag, tab_analysis = st.tabs(
            ["Data", "Flagged Courses", "Analysis"]
        )

    # ── Select Data Tab ──────────────────────────────────────────────────
    with tab_data:
        uploaded_file = st.file_uploader(
            "Upload Section Attrition & Grade Report (Excel)",
            type=["xlsx", "xls"],
            help="Upload the Excel file with an 'Export' sheet.",
        )

        if uploaded_file is None:
            st.info("Upload an Excel file to begin analysis.")
            render_methodology()
            return

        # Load and validate
        try:
            raw_df, info = load_excel(uploaded_file)
        except Exception as e:
            st.error(f"Error reading Excel file: {e}")
            return

        # Validation summary
        st.subheader("Upload Summary")
        st.write(f"**Rows loaded:** {info['total_rows']}")

        if info["missing_required"]:
            st.error(
                f"**Missing required columns:** {', '.join(info['missing_required'])}"
            )
            st.stop()

        if info["missing_optional"]:
            st.warning(
                f"**Missing optional columns (will default to 0):** "
                f"{', '.join(info['missing_optional'])}"
            )

        # Clean data
        section_df, rollup_count, _ = clean_dataframe(raw_df)
        st.write(f"**Rollup/total rows removed:** {rollup_count}")
        st.write(f"**Individual section rows retained:** {len(section_df)}")
        st.success("Data loaded successfully.")

        # Compute section-level metrics, then build course-term averages
        section_df = compute_metrics(section_df)
        course_term_df = build_course_term_averages(section_df)

        # ── Term and Subject selection via checkboxes ────────────────────
        st.divider()
        st.subheader("Select Terms")

        all_terms = sorted(
            section_df["Term Description"].unique(),
            key=lambda t: _build_term_order(
                section_df["Term Description"].unique()
            ).get(t, 0),
        )

        for term in all_terms:
            if f"term_{term}" not in st.session_state:
                st.session_state[f"term_{term}"] = True

        btn_col1, btn_col2, _ = st.columns([1, 1, 4])
        with btn_col1:
            if st.button("Select all terms"):
                for term in all_terms:
                    st.session_state[f"term_{term}"] = True
                st.rerun()
        with btn_col2:
            if st.button("Deselect all terms"):
                for term in all_terms:
                    st.session_state[f"term_{term}"] = False
                st.rerun()

        term_cols = st.columns(4)
        selected_terms = []
        for i, term in enumerate(all_terms):
            with term_cols[i % 4]:
                if st.checkbox(term, key=f"term_{term}"):
                    selected_terms.append(term)

        st.divider()
        st.subheader("Select Subjects")

        all_subjects = sorted(section_df["Subject"].unique())

        for subj in all_subjects:
            if f"subj_{subj}" not in st.session_state:
                st.session_state[f"subj_{subj}"] = True

        btn_col3, btn_col4, _ = st.columns([1, 1, 4])
        with btn_col3:
            if st.button("Select all subjects"):
                for subj in all_subjects:
                    st.session_state[f"subj_{subj}"] = True
                st.rerun()
        with btn_col4:
            if st.button("Deselect all subjects"):
                for subj in all_subjects:
                    st.session_state[f"subj_{subj}"] = False
                st.rerun()

        subj_cols = st.columns(4)
        selected_subjects = []
        for i, subj in enumerate(all_subjects):
            with subj_cols[i % 4]:
                if st.checkbox(subj, key=f"subj_{subj}"):
                    selected_subjects.append(subj)

        # Store in session state
        st.session_state["section_df"] = section_df
        st.session_state["course_term_df"] = course_term_df
        st.session_state["selected_terms"] = selected_terms
        st.session_state["selected_subjects"] = selected_subjects

        # Show current thresholds (read-only for all users)
        st.divider()
        render_thresholds_readonly()

    # ── Admin Tab (admin only) ────────────────────────────────────────────
    if is_admin():
        with tab_admin:
            render_thresholds_admin()

    # ── Load settings for downstream tabs ─────────────────────────────────
    settings = _load_settings()

    # ── Flag Challenges Tab ──────────────────────────────────────────────
    with tab_flag:
        if "course_term_df" not in st.session_state:
            st.info("Upload a file in the Select Data tab to begin.")
            return

        result = _load_and_filter(settings)
        if result is None:
            st.warning(
                "No data matches the current filters. "
                "Check your selections in the Select Data tab."
            )
            return

        filtered_ct, filtered_section, flagged = result

        # Summary metrics
        col1, col2, col3, col4 = st.columns(4)
        unique_courses = filtered_ct[
            ["Subject", "Catalog Number"]
        ].drop_duplicates()
        col1.metric("Courses Analyzed", len(unique_courses))
        col2.metric("Courses Flagged", len(flagged))
        col3.metric(
            "Terms in Dataset", filtered_ct["Term Description"].nunique()
        )
        col4.metric(
            "Avg DFW Rate (across courses)",
            f"{filtered_ct['dfw_rate'].mean():.1%}"
            if not filtered_ct["dfw_rate"].isna().all()
            else "N/A",
        )

        # Flagged courses table
        selected_course = render_flagged_table(flagged)

        # Drill-down
        if selected_course:
            render_course_drilldown(
                selected_course,
                filtered_ct,
                filtered_section,
                dfw_threshold=settings["dfw_threshold"],
                drop_threshold=settings["drop_threshold"],
                incomplete_threshold=settings["incomplete_threshold"],
            )

        # Exports
        render_exports(flagged, filtered_ct, filtered_section, selected_course)

        # Methodology
        render_methodology()

    # ── Analysis Tab ─────────────────────────────────────────────────────
    with tab_analysis:
        if "course_term_df" not in st.session_state:
            st.info("Upload a file in the Select Data tab to begin.")
            return

        result = _load_and_filter(settings)
        if result is None:
            st.warning(
                "No data matches the current filters. "
                "Check your selections in the Select Data tab."
            )
            return

        filtered_ct, filtered_section, flagged = result

        st.subheader("Written Analysis")
        st.caption(
            "This analysis is generated from the flagged courses on the "
            "Flag Challenges tab. Adjusting thresholds or data selections "
            "will update this report."
        )

        report = generate_analysis(flagged, filtered_ct, filtered_section, settings)
        st.markdown(report)

        # Download the report
        st.divider()
        st.download_button(
            "Download Analysis Report (Markdown)",
            report,
            "analysis_report.md",
            "text/markdown",
        )


if __name__ == "__main__":
    main()
