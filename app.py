"""
FastAPI backend for PDF sanitization.
POST /sanitize  -> returns ZIP with masked.pdf + report.pdf
GET  /health    -> health check
"""
import os
import shutil
import tempfile
import time
import zipfile

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse

from sanitization.detector import detect_entities, load_config
from sanitization.extractor import extract_pages
from sanitization.masker import mask_pdf
from sanitization.reporter import generate_report

app = FastAPI(title="PDF Sanitization API", version="2.0.0")
CONFIG = load_config("configs/entities.yaml")


@app.get("/health")
def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/sanitize")
async def sanitize(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    work_dir = tempfile.mkdtemp()
    try:
        start = time.time()
        print("[EXTRACT] Saving uploaded file")

        input_path = os.path.join(work_dir, "input.pdf")
        with open(input_path, "wb") as f:
            f.write(await file.read())

        pages = extract_pages(input_path)
        total_pages = len(pages)
        mode_summary = {}
        for p in pages:
            mode_summary[p["method"]] = mode_summary.get(p["method"], 0) + 1

        entities, review_log = detect_entities(pages, CONFIG)

        masked_path = os.path.join(work_dir, "masked.pdf")
        mask_pdf(input_path, masked_path, entities)

        report_path = os.path.join(work_dir, "report.pdf")
        elapsed = round(time.time() - start, 3)
        generate_report(
            output_path=report_path,
            source_file=file.filename,
            total_pages=total_pages,
            extraction_mode_summary=mode_summary,
            entities=entities,
            review_log=review_log,
            processing_seconds=elapsed,
        )

        print("[REPORT] Bundling output PDFs")
        zip_path = os.path.join(work_dir, "sanitized_bundle.zip")
        with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED) as zf:
            zf.write(masked_path, "masked.pdf")
            zf.write(report_path, "report.pdf")

        return FileResponse(path=zip_path, media_type="application/zip", filename="sanitized_bundle.zip")
    except Exception as e:
        print(f"[REPORT] Pipeline error: {e}")
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
