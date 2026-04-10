# PII Sanitizer — Agentic Architecture (V4.0)

A high-performance FastAPI and Agentic-AI backend for document sanitization. This system features native PDF extraction, chunked OCR fallback, an ensemble of Named Entity Recognition (NER), LLM validation, strict O(N) deduplication, and single-pass PyMuPDF redaction.

V4.0 introduces an entirely new **Agentic Orchestration Pipeline** connected to a premium, glassmorphic UI via Server-Sent Events (SSE) for real-time progress tracking.

## 🚀 Extreme Performance Optimizations

This system is built to handle massive, 100+ page financial and legal documents (e.g., 10-K filings, court records) in **under 30 seconds**.

- **PyMuPDF native tables**: Replaced `pdfplumber` with PyMuPDF's C-compiled `find_tables()`, bringing table extraction time from minutes down to milliseconds.
- **Embedded Image Limits**: Automatically filters out invisible 1x1 structural pixel spacers prevalent in SEC filings from hitting the deep learning doctr engine.
- **Chunked OCR Batching**: Uses adaptive threading and dynamically batched OCR to avoid RAM saturation without sacrificing CPU throughput.
- **Single-Pass Redaction**: Uses a 1-pass zero-deflate PyMuPDF operation to mask entities and reconstruct PDFs in 1–2 seconds, entirely dropping `garbage=4` compressions.
- **Lightweight NLP Ensemble**: Relies on `en_core_web_lg` combined with Presidio and custom Regex engines to provide ultra-fast heuristic detection. Heavy CPU-bound Transformer logic (`dslim/bert-base-NER`) is disabled by default for maximum throughput.

## 🧠 Pipeline Agents

The process runs sequentially across highly optimized autonomous agents:

1. `ExtractionAgent`: Parallel chunked native text, C-speed table extraction, and image OCR.
2. `DetectionAgent`: ThreadPool chunked execution of Presidio (spaCy `en_core_web_lg`) and custom validators. Passes high-confidence signals and captures low-confidence boundary edge cases.
3. `ResolutionAgent`: O(N log N) Exact-match dictionary deduping combined with fuzzy clustering for cross-page entity normalization.
4. `ReviewAgent`: Defers ambiguous or conflicting entities to a Batch LLM (OpenAI/Anthropic) using Qdrant vector-store memory of past programmatic decisions.
5. `MaskingAgent`: Receives the resolved manifest and does a single-pass `fitz` traversal to draw bounding boxes and apply true destruction to underlying PDF strings.
6. `ReportingAgent`: Generates a high-quality summary PDF detailing mask counts and LLM categorization decisions.

## 💻 The Dashboard

A modern, glassmorphic, vanilla HTML/JS interactive dashboard is served automatically at `/`.
- **Drag-and-drop** PDF ingestion.
- **Live SSE Progress**: Watch agents execute your document in real-time.
- **Entity Highlighting**: Draw yellow, translucent bounding boxes over exactly what the ML pipeline detected.
- **Side-by-Side Compare**: Compare your pre-masked document with the masked output securely in the browser using `pdf.js`.
- **LLM Chat**: Post-process, you can chat with the reporting agent explicitly about the generated entity data.

## ⚙️ Build and Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Download the core spaCy engine
python -m spacy download en_core_web_lg

# 3. Start the server (models pre-warm immediately at boot)
uvicorn app:app --host 0.0.0.0 --port 8000
```

### Environment Overrides

Create a `.env` file to customize agent parameters:

```env
OPENAI_API_KEY=your_openai_key
EXTRACTOR_WORKERS=6
DETECT_WORKERS=4
OCR_MIN_NATIVE_CHARS=80
ENABLE_HF_NER=0         # Set to 1 to enable BERT baseline (WARNING: Extremely slow on CPU)
```

## 🌐 API Endpoints

- `GET /` — Loads the interactive Dashboard.
- `GET /health` — Returns status and `models_ready`.
- `GET /ready` — 503 while model booting, 200 when ready.
- `POST /sanitize` — Streaming endpoint that accepts a PDF file.
- `GET /stream/{session_id}` — SSE pipeline progress hook.
- `GET /pdf/{session_id}/{type}` — Fetch outputs (`original`, `masked`, `report`).
- `GET /entities/{session_id}` — Resolves the JSON manifest built after masking.
