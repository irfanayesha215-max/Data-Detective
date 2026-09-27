# Data Detective 🔍

An AI-powered data cleaning tool that profiles messy datasets, diagnoses issues in plain English, and lets you review — or auto-apply — the fixes.

## Why this exists

Real-world data is messy: missing values, duplicate rows, inconsistent categories ("USA" vs "usa" vs "U.S.A"), wrong data types. Most "auto-clean" tools silently change your data without explaining what they did or why. Data Detective is built the opposite way — it explains every issue it finds and every fix it proposes, and (in Review Mode) asks before touching anything.

## Status: 🚧 Work in progress

This project is being built incrementally. Current progress:

- [x] Missing values detection
- [x] Duplicate row detection
- [x] Inconsistent category detection
- [ ] Data type mismatch detection
- [ ] Outlier detection
- [ ] Formatting issue detection (whitespace, casing)
- [ ] AI-powered diagnosis layer (Claude API)
- [ ] Confidence scoring for auto-apply decisions
- [ ] Review Mode UI
- [ ] Auto Mode
- [ ] Cleaning report generation

## Tech stack

- **Backend:** Python, FastAPI, pandas
- **AI:** Claude API (diagnosis + fix suggestions)
- **Frontend:** Next.js (coming soon)

## Running the profiler locally

```bash
cd backend
python3 -m venv venv
source venv/bin/activate    # Windows: venv\Scripts\activate
pip install -r requirements.txt
python3 app/profiler.py
```

## Project structure

```
data-detective/
├── backend/
│   ├── app/
│   │   └── profiler.py     # Core data profiling checks
│   ├── tests/
│   └── requirements.txt
├── frontend/                # Coming soon
└── sample_data/              # Example messy datasets for testing
```