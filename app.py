"""
FastAPI application — PII Sanitizer Platform
Endpoints:
  POST /sanitize             → start pipeline (async), return session_id immediately
  GET  /stream/{session_id} → SSE live agent progress
  GET  /pdf/{session_id}/{name}     → inline PDF viewer
  GET  /pdf/{session_id}/original  → original PDF (for side-by-side)
  GET  /download/{session_id}/{filename}.pdf → force-download
  GET  /entities/{session_id}      → entity list + bbox manifest for UI overlay
  POST /annotate/{session_id}      → re-mask with user-drawn annotation boxes
  POST /chat                       → conversational Q&A about report
  GET  /sessions                   → list active sessions
  GET  /health
"""
import asyncio
import json
import os
import tempfile
import uuid

import fitz
from dotenv import load_dotenv
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

load_dotenv()

from sanitization.detector import load_config
from sanitization.orchestrator import run_pipeline_async
from sanitization.extractor import prewarm as prewarm_ocr

app = FastAPI(title="PII Sanitizer")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ── Model readiness flag ─────────────────────────────────────────────────────
_MODELS_READY = False


@app.on_event("startup")
async def startup():
    """
    Blocking parallel prewarm — loads ALL models before server accepts requests.
    doctr OCR + Presidio spaCy + transformer NER load concurrently in threads.
    Total startup time ≈ max(doctr_load, presidio_load) ≈ 10–25s once.
    After this, every request starts instantly with zero cold-start overhead.
    """
    global _MODELS_READY
    import asyncio
    from sanitization.detector import prewarm_analyzer
    print("[STARTUP] Loading all models in parallel…")
    await asyncio.gather(
        asyncio.to_thread(prewarm_ocr),        # blocks until doctr is loaded
        asyncio.to_thread(prewarm_analyzer),   # blocks until Presidio + NER loaded
    )
    _MODELS_READY = True
    print("[STARTUP] All models ready — server accepting requests.")


_SESSIONS: dict[str, dict] = {}
_CONFIG = load_config()


# ──────────────── helpers ───────────────────────────────────────────────────

def _extract_pdf_text(path: str) -> str:
    try:
        doc = fitz.open(path)
        text = "\n".join(p.get_text("text") for p in doc)
        doc.close()
        return text
    except Exception:
        return ""


# ──────────────── routes ────────────────────────────────────────────────────

@app.get("/")
def root():
    from fastapi.responses import FileResponse
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok", "models_ready": _MODELS_READY}


@app.get("/ready")
def ready():
    """Returns 200 when all models are loaded, 503 while warming up."""
    if _MODELS_READY:
        return {"ready": True, "message": "All models loaded"}
    from fastapi.responses import JSONResponse
    return JSONResponse(
        status_code=503,
        content={"ready": False, "message": "Models warming up, please wait…"}
    )



@app.get("/sessions")
def list_sessions():
    return {
        sid: {
            "source_file": s.get("source_file"),
            "status": s.get("status", "unknown"),
            "entity_count": s.get("entity_count", 0),
        }
        for sid, s in _SESSIONS.items()
    }


@app.post("/sanitize")
async def sanitize(file: UploadFile, background_tasks: BackgroundTasks):
    session_id = str(uuid.uuid4())
    work_dir = os.path.join(tempfile.mkdtemp(), session_id)
    os.makedirs(work_dir, exist_ok=True)

    content = await file.read()
    input_path = os.path.join(work_dir, file.filename)
    with open(input_path, "wb") as f:
        f.write(content)

    event_queue: asyncio.Queue = asyncio.Queue()
    _SESSIONS[session_id] = {
        "work_dir": work_dir,
        "source_file": file.filename,
        "input_path": input_path,
        "masked_path": None,
        "report_path": None,
        "report_text": "",
        "status": "running",
        "event_queue": event_queue,
        "entity_count": 0,
        "redaction_manifest": {},
    }

    async def _run():
        try:
            ctx = await run_pipeline_async(
                session_id=session_id,
                input_path=input_path,
                source_file=file.filename,
                config=_CONFIG,
                work_dir=work_dir,
                event_queue=event_queue,
            )
            _SESSIONS[session_id].update({
                "masked_path": ctx.masked_path,
                "report_path": ctx.report_path,
                "report_text": _extract_pdf_text(ctx.report_path),
                "status": "complete",
                "entity_count": len(ctx.final_entities),
                "redaction_manifest": ctx.redaction_manifest,
            })
        except Exception as e:
            _SESSIONS[session_id]["status"] = "error"
            await event_queue.put({"type": "pipeline_error", "error": str(e)})
            await event_queue.put(None)

    background_tasks.add_task(_run)

    return JSONResponse({
        "session_id": session_id,
        "stream_url": f"/stream/{session_id}",
    })


@app.get("/stream/{session_id}")
async def stream_progress(session_id: str, request: Request):
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")

    queue: asyncio.Queue = session["event_queue"]

    async def generator():
        while True:
            if await request.is_disconnected():
                break
            try:
                event = await asyncio.wait_for(queue.get(), timeout=60.0)
                if event is None:
                    yield f"data: {json.dumps({'type': 'done'})}\n\n"
                    break
                yield f"data: {json.dumps(event)}\n\n"
            except asyncio.TimeoutError:
                yield 'data: {"type":"ping"}\n\n'

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@app.get("/pdf/{session_id}/{name}")
def serve_pdf(session_id: str, name: str):
    from fastapi.responses import FileResponse
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    paths = {
        "masked": session.get("masked_path"),
        "report": session.get("report_path"),
        "original": session.get("input_path"),
    }
    path = paths.get(name)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(path=path, media_type="application/pdf",
                        headers={"Content-Disposition": "inline"})


@app.get("/download/{session_id}/{filename}")
def download_pdf(session_id: str, filename: str):
    from fastapi.responses import FileResponse
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    name_map = {
        "masked.pdf": session.get("masked_path"),
        "report.pdf": session.get("report_path"),
        "original.pdf": session.get("input_path"),
    }
    path = name_map.get(filename)
    if not path or not os.path.exists(path):
        raise HTTPException(status_code=404, detail="PDF not found")
    return FileResponse(path=path, media_type="application/pdf", filename=filename,
                        headers={"Content-Disposition": f'attachment; filename="{filename}"'})


@app.get("/entities/{session_id}")
def get_entities(session_id: str):
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    return {"manifest": session.get("redaction_manifest", {})}


class AnnotationRequest(BaseModel):
    boxes: list[dict]  # [{page, x0, y0, x1, y1}]


@app.post("/annotate/{session_id}")
async def annotate(session_id: str, req: AnnotationRequest, background_tasks: BackgroundTasks):
    session = _SESSIONS.get(session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    if not session.get("masked_path"):
        raise HTTPException(status_code=400, detail="Pipeline not complete yet")

    # Re-apply redactions including user boxes onto the masked PDF
    def _apply_annotations():
        try:
            doc = fitz.open(session["masked_path"])
            for box in req.boxes:
                pg = int(box.get("page", 0))
                if pg < len(doc):
                    rect = fitz.Rect(box["x0"], box["y0"], box["x1"], box["y1"])
                    doc[pg].add_redact_annot(rect, fill=(0, 0, 0))
                    doc[pg].apply_redactions()
            doc.save(session["masked_path"], garbage=4, deflate=True)
            doc.close()
        except Exception as e:
            print(f"[ANNOTATE] Error: {e}")

    background_tasks.add_task(_apply_annotations)
    return {"status": "annotation_applied", "boxes": len(req.boxes)}


class ChatRequest(BaseModel):
    session_id: str
    question: str


@app.post("/chat")
async def chat(req: ChatRequest):
    session = _SESSIONS.get(req.session_id)
    if not session:
        raise HTTPException(status_code=404, detail="Session not found")
    report_text = session.get("report_text", "")
    try:
        from sanitization.chat_agent import chat_with_report
        answer = await asyncio.to_thread(
            chat_with_report,
            session_id=req.session_id,
            question=req.question,
            report_text=report_text,
        )
        return {"answer": answer}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
