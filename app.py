import io
import os
import tempfile
import zipfile
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from sanitization.config import load_config
from sanitization.pipeline import sanitize_document
from sanitization.report import write_report_pdf

app = FastAPI(title="Document Sanitization Backend", version="1.0.0")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/sanitize")
async def sanitize(file: UploadFile = File(...)):
    filename = file.filename or "uploaded.pdf"
    if not filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    cfg = load_config("configs/pipeline.yaml")
    cfg["pipeline"]["mode"] = "redact"

    with tempfile.TemporaryDirectory(prefix="sanitize_") as tmp:
        input_pdf = os.path.join(tmp, "input.pdf")
        masked_pdf = os.path.join(tmp, "masked.pdf")
        report_json = os.path.join(tmp, "report.json")
        report_pdf = os.path.join(tmp, "report.pdf")

        content = await file.read()
        Path(input_pdf).write_bytes(content)

        try:
            summary = sanitize_document(
                input_pdf=input_pdf,
                output_pdf=masked_pdf,
                report_file=report_json,
                config=cfg,
            )
            write_report_pdf(summary, report_pdf)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Sanitization failed: {e}") from e

        bundle = io.BytesIO()
        with zipfile.ZipFile(bundle, "w", compression=zipfile.ZIP_DEFLATED) as zf:
            zf.write(masked_pdf, arcname="masked.pdf")
            zf.write(report_pdf, arcname="report.pdf")
        bundle.seek(0)

        return StreamingResponse(
            bundle,
            media_type="application/zip",
            headers={"Content-Disposition": f'attachment; filename="{Path(filename).stem}_sanitized_bundle.zip"'},
        )


@app.get("/")
def root():
    return JSONResponse(
        {
            "message": "Upload PDF to POST /sanitize",
            "output": "ZIP with masked.pdf and report.pdf",
        }
    )

