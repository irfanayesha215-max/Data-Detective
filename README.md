# Data Detective 🔍

An AI-powered data cleaning tool that profiles messy datasets, diagnoses issues in plain English, and lets you review — or auto-apply — the fixes.

## Why this exists

Real-world data is messy: missing values, duplicate rows, inconsistent categories ("USA" vs "usa" vs "U.S.A"), wrong data types. Most "auto-clean" tools silently change your data without explaining what they did or why. Data Detective is built the opposite way — it explains every issue it finds and every fix it proposes, and (in Review Mode) asks before touching anything.

## Status: 🚧 Work in progress

This project is being built incrementally. Current progress:

- [x] Missing values detection
- [x] Duplicate row detection
- [x] Inconsistent category detection
- [x] Data type mismatch detection
- [x] Outlier detection
- [x] Formatting issue detection (whitespace, casing)
- [x] AI-powered diagnosis layer (GROQ API)
- [x] Confidence scoring for auto-apply decisions
- [x] Review Mode UI
- [x] Auto Mode
- [x] Cleaning report generation

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

## Running the web app

\`\`\`bash
cd backend
source venv/bin/activate
uvicorn app.main:app --reload
\`\`\`
Then open http://127.0.0.1:8000 in your browser.


## Project structure

 data-detective/
├── README.md
├── .gitignore
├── backend/
│   ├── app/
│   │   ├── __init__.py
│   │   ├── profiler.py      # 6 data profiling checks
│   │   ├── diagnosis.py     # Sends the report to the AI (Groq) for plain-English diagnosis
│   │   ├── confidence.py    # Caps the AI's confidence scores per issue type
│   │   ├── fixer.py         # Applies approved fixes, saves cleaned file + report
│   │   ├── pipeline.py      # Connects profiler -> diagnosis -> confidence -> fixer
│   │   └── main.py          # FastAPI server (upload, review, download endpoints)
│   ├── tests/
│   └── requirements.txt
├── frontend/
│   └── index.html           # Upload page with Review Mode (approve fixes individually)
├── sample_data/
│   └── messy_sample.csv     # Example messy dataset for testing
└── output/                  # Generated cleaned files + reports (gitignored, not tracked)