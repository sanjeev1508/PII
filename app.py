"""
FastAPI backend for PDF sanitization + chat UI.

Endpoints
---------
POST /sanitize            -> starts pipeline, returns session_id
GET  /pdf/{session_id}/{name}  -> serves masked.pdf or report.pdf
POST /chat                -> conversational Q&A about the report
GET  /health
GET  /                    -> serves the UI
"""
import os
import shutil
import tempfile
import time
import uuid
import zipfile
from pathlib import Path

import uvicorn
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from sanitization.chat_agent import chat_with_report
from sanitization.detector import load_config
from sanitization.orchestrator import run_sanitization_pipeline

app = FastAPI(title="PDF Sanitization API", version="3.0.0")
CONFIG = load_config("configs/entities.yaml")

# In-memory session store: session_id -> {work_dir, source_file, report_text}
_SESSIONS: dict[str, dict] = {}

# Serve static assets
STATIC_DIR = Path(__file__).parent / "static"
STATIC_DIR.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


# --------------------------------------------------------------------------- #
#  UI
# --------------------------------------------------------------------------- #

@app.get("/", response_class=HTMLResponse)
def ui():
    html_path = STATIC_DIR / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text(encoding="utf-8"))
    return HTMLResponse("<h1>UI not found — place index.html in static/</h1>", status_code=404)


# --------------------------------------------------------------------------- #
#  Health
# --------------------------------------------------------------------------- #

@app.get("/health")
def health():
    return {"status": "ok", "version": "3.0.0"}


# --------------------------------------------------------------------------- #
#  Sanitize
# --------------------------------------------------------------------------- #

@app.post("/sanitize")
async def sanitize(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are accepted.")

    session_id = str(uuid.uuid4())
    work_dir = tempfile.mkdtemp(prefix=f"pii_{session_id}_")

    try:
        print(f"[EXTRACT] Saving uploaded file  session={session_id}")
        input_path = os.path.join(work_dir, "input.pdf")
        with open(input_path, "wb") as f:
            f.write(await file.read())

        pipeline_result = run_sanitization_pipeline(
            input_path=input_path,
            source_file=file.filename,
            config=CONFIG,
            work_dir=work_dir,
        )

        masked_path = pipeline_result["masked_path"]
        report_path = pipeline_result["report_path"]

        # Extract plain text from report PDF for chat context
        report_text = _extract_pdf_text(report_path)

        _SESSIONS[session_id] = {
            "work_dir": work_dir,
            "source_file": file.filename,
            "masked_path": masked_path,
            "report_path": report_path,
            "report_text": report_text,
        }

        return JSONResponse({
            "session_id": session_id,
            "source_file": file.filename,
            "pdfs": {
                "masked": f"/pdf/{session_id}/masked",
                "report": f"/pdf/{session_id}/report",
            },
        })

    except Exception as e:
        print(f"[SANITIZE] Pipeline error: {e}")
        shutil.rmtree(work_dir, ignore_errors=True)
        raise HTTPException(status_code=500, detail=str(e))


def _extract_pdf_text(pdf_path: str) -> str:
    try:
        import fitz
        doc = fitz.open(pdf_path)
        text = "\n".join(page.get_text("text") for page in doc)
        doc.close()
        return text
    except Exception:
        return ""


# --------------------------------------------------------------------------- #
#  PDF serving
# --------------------------------------------------------------------------- #

@app.get("/pdf/{session_id}/{name}")
def serve_pdf(session_id: str, name: str):
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if name == "masked":
        path = session["masked_path"]
        fname = "masked.pdf"
    elif name == "report":
        path = session["report_path"]
        fname = "report.pdf"
    else:
        raise HTTPException(status_code=400, detail="name must be 'masked' or 'report'.")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="PDF not found.")
    return FileResponse(path=path, media_type="application/pdf", filename=fname,
                        headers={"Content-Disposition": "inline"})


@app.get("/download/{session_id}/{name}")
def download_pdf(session_id: str, name: str):
    """Force-download endpoint — serves with Content-Disposition: attachment."""
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found.")
    if name == "masked":
        path = session["masked_path"]
        fname = "masked.pdf"
    elif name == "report":
        path = session["report_path"]
        fname = "report.pdf"
    else:
        raise HTTPException(status_code=400, detail="name must be 'masked' or 'report'.")
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="PDF not found.")
    return FileResponse(
        path=path,
        media_type="application/pdf",
        filename=fname,
        headers={"Content-Disposition": f'attachment; filename="{fname}"'},
    )


# --------------------------------------------------------------------------- #
#  Chat
# --------------------------------------------------------------------------- #

class ChatRequest(BaseModel):
    session_id: str
    question: str


@app.post("/chat")
def chat(req: ChatRequest):
    session = _SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found. Process a PDF first.")
    answer = chat_with_report(
        session_id=req.session_id,
        question=req.question,
        report_text=session.get("report_text", ""),
        source_file=session.get("source_file", ""),
    )
    return {"answer": answer}


if __name__ == "__main__":
    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True)
