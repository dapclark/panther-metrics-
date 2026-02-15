# Panther Metrics — Student Success Patterns Analyzer

A Streamlit web app that helps academic departments analyze multi-year, course-level student success outcomes using standardized Section Attrition & Grade Report Excel files.

## Features

- **Upload & Validate** — Upload Excel files with automatic rollup-row detection and removal
- **Flagged Courses Dashboard** — Instantly see courses with problematic DFW, drop, incomplete, and repeat patterns
- **Pattern Detection** — Persistent high DFW, worsening trends, spikes, and co-occurring issues
- **Drill-Down** — Click any flagged course to see term-by-term trends and worst sections
- **Adjustable Thresholds** — All thresholds and pattern parameters are live-editable in the sidebar
- **Exports** — Download CSVs of flagged courses, course-term metrics, and section details

## Setup

```bash
# Create a virtual environment (recommended)
python -m venv venv
source venv/bin/activate  # macOS/Linux
# venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## Run

```bash
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

## Run Tests

```bash
pytest tests/ -v
```

## Data Format

Upload an Excel file (`.xlsx`) with a sheet named **Export** containing these columns:

| Column | Required |
|--------|----------|
| Term Description | Yes |
| Subject | Yes |
| Catalog Number | Yes |
| Section Number | Yes |
| Official Class Enrollments | Yes |
| Tot. # Grades used for DFW Rate Analysis | Yes |
| Ds,Fs,Ws used for DFW Rate Analysis | Yes |
| Dropped | Yes |
| Repeats | Optional |
| Incomplete (I) | Optional |
| Extended Incomplete (EI) | Optional |
| Permanent Incomplete (PI) | Optional |
| Incomplete lapsed to F (@F) | Optional |

Rollup rows (where Subject, Catalog Number, or Section Number is "Total" or blank) are automatically detected and excluded.

## Project Structure

```
app.py              — Streamlit entry point
data_loader.py      — Excel loading, cleaning, rollup removal, aggregation
metrics.py          — Rate computations (DFW, drop, incomplete, repeat)
patterns.py         — Pattern detection (persistent, trend, spike, co-occurring)
ui_components.py    — Reusable UI: filters, tables, charts, exports
tests/              — Test suite
requirements.txt    — Python dependencies
```
