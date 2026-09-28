"""
The web server. Wraps the existing pipeline (profiler -> AI -> confidence
caps -> fixer) behind a single upload endpoint, so it can be used from a
browser instead of the terminal.

Run from the backend/ folder:
    uvicorn app.main:app --reload
Then open http://127.0.0.1:8000 in a browser.
"""
import uuid
import json
import shutil
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from app.pipeline import build_report
from app.diagnosis import diagnose_report
from app.confidence import apply_confidence_caps
from app.fixer import apply_fixes, write_report

app = FastAPI(title="Data Detective")

# Allows a frontend served from a different port/file to call this API.
# Fine for a personal/portfolio project; a real production app would
# restrict this to a specific domain instead of "*".
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# every upload gets its own folder here, named with a random id,
# so two people using the tool at once never overwrite each other's files
JOBS_DIR = Path(__file__).resolve().parents[2] / "output" / "jobs"
JOBS_DIR.mkdir(parents=True, exist_ok=True)


@app.get("/", response_class=HTMLResponse)
def home():
    """Serve the upload page itself, so this one server does everything."""
    index_file = Path(__file__).resolve().parents[2] / "frontend" / "index.html"
    return index_file.read_text(encoding="utf-8")


class ApplyRequest(BaseModel):
    approved_ids: list[int]


def _run_and_save(df: pd.DataFrame, results: list, job_dir: Path, original_filename: str) -> dict:
    """Shared by both endpoints: apply fixes, save the outputs, build the response."""
    cleaned_df, applied, skipped = apply_fixes(df, results)

    cleaned_path = job_dir / "cleaned.csv"
    report_path = job_dir / "report.md"
    cleaned_df.convert_dtypes().to_csv(cleaned_path, index=False)
    write_report(report_path, original_filename, len(df), len(cleaned_df), applied, skipped)

    # save the current state of every result (including which are now
    # applied) so a later approval request can pick up from here
    (job_dir / "results.json").write_text(json.dumps(results), encoding="utf-8")

    return {
        "job_id": job_dir.name,
        "rows_before": len(df),
        "rows_after": len(cleaned_df),
        "results": results,
        "download_cleaned_csv": f"/api/download/{job_dir.name}/cleaned.csv",
        "download_report": f"/api/download/{job_dir.name}/report.md",
    }


@app.post("/api/clean")
async def clean_file(file: UploadFile = File(...)):
    """
    Accept a CSV upload, run it through the full pipeline, apply the safe
    fixes, and return the results as JSON with links to download the
    cleaned file and the report. Fixes below the confidence threshold are
    explained but left for the person to approve via /api/apply.
    """
    if not file.filename.lower().endswith(".csv"):
        raise HTTPException(status_code=400, detail="Please upload a .csv file")

    job_id = uuid.uuid4().hex[:8]
    job_dir = JOBS_DIR / job_id
    job_dir.mkdir(parents=True)

    upload_path = job_dir / "original.csv"
    with upload_path.open("wb") as f:
        shutil.copyfileobj(file.file, f)
    (job_dir / "original_filename.txt").write_text(file.filename, encoding="utf-8")

    try:
        df = pd.read_csv(upload_path)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Could not read that CSV: {e}")

    report = build_report(df)
    if not report:
        return {"job_id": job_id, "message": "No issues found - this dataset looks clean!", "results": []}

    diagnosis = diagnose_report(report)
    results = apply_confidence_caps(diagnosis)
    # give every diagnosis a stable id so the frontend can say
    # "the person approved fix #3" instead of matching by text
    for i, r in enumerate(results):
        r["id"] = i

    return _run_and_save(df, results, job_dir, file.filename)


@app.post("/api/apply/{job_id}")
def apply_selected(job_id: str, body: ApplyRequest):
    """
    Take the ids of fixes the person approved from the "needs review" list,
    mark them as approved, and re-run the fixer starting from the ORIGINAL
    file - so approvals are never stacked on top of a half-fixed file.
    """
    job_dir = JOBS_DIR / job_id
    original_path = job_dir / "original.csv"
    results_path = job_dir / "results.json"
    if not original_path.exists() or not results_path.exists():
        raise HTTPException(status_code=404, detail="Job not found")

    df = pd.read_csv(original_path)
    results = json.loads(results_path.read_text(encoding="utf-8"))

    for r in results:
        if r["id"] in body.approved_ids:
            r["auto_apply"] = True

    original_filename = (job_dir / "original_filename.txt").read_text(encoding="utf-8")
    return _run_and_save(df, results, job_dir, original_filename)


@app.get("/api/download/{job_id}/{filename}")
def download(job_id: str, filename: str):
    """Serve a cleaned file or report back for download."""
    # only allow the two known filenames - stops someone from requesting
    # an arbitrary path like ../../.env
    if filename not in ("cleaned.csv", "report.md"):
        raise HTTPException(status_code=404, detail="Not found")

    file_path = JOBS_DIR / job_id / filename
    if not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")

    return FileResponse(file_path, filename=filename)