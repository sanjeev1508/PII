# PII Sanitizer — Agentic Architecture (V4.0)

A fast, secure, production-tier PII sanitization pipeline orchestrated by Agentic modules, zero-shot ML ensembles, and Vision LLM mapping natively handling unstructured 100+ page datasets.

V4.0 introduces an entirely new **Agentic Orchestration Pipeline** connected to a premium, glassmorphic UI via Server-Sent Events (SSE) for real-time progress tracking.

## 🚀 Extreme Performance Optimizations

This system is built to handle massive, 100+ page financial and legal documents (e.g., 10-K filings, court records) in **under 30 seconds**.

The `ExtractionAgent` processes incoming binary streams, parsing the physical document layout into dual-layer structures:
- **Native Context**: PyMuPDF extraction handles purely selectable texts and layout tables immediately.
- **Vision Layer**: For complex graphical banners and embedded images, the pipeline skips archaic, heavy-weight OCR binaries (like `doctr`) array-by-array, opting to actively capture visual base64 data to proxy asynchronously to an interconnected **OpenAI Vision Engine**. Thus, proprietary logos or stylized graph representations are safely decoded over secure channels, dropping CPU pressure and latency by ~70%.
- **Embedded Image Limits**: Automatically filters out invisible 1x1 structural pixel spacers prevalent in SEC filings from hitting the deep learning doctr engine.
- **Chunked OCR Batching**: Uses adaptive threading and dynamically batched OCR to avoid RAM saturation without sacrificing CPU throughput.
- **Single-Pass Redaction**: Uses a 1-pass zero-deflate PyMuPDF operation to mask entities and reconstruct PDFs in 1–2 seconds, entirely dropping `garbage=4` compressions.
- **Lightweight NLP Ensemble**: Relies on `en_core_web_lg` combined with Presidio and custom Regex engines to provide ultra-fast heuristic detection. Heavy CPU-bound Transformer logic (`dslim/bert-base-NER`) is disabled by default for maximum throughput.

## 🧠 Pipeline Agents

### Agentic Core Pipeline V5.0
1. **Extraction Pipeline (`ExtractionAgent`)**: Parallelizes native PDF text extraction while routing graphical raster assets physically through asynchronous LLM Vision endpoints to seamlessly spot localized logo/brand boundaries.  
2. **Entity Detection (`DetectionAgent`)**: Integrates structured Presidio, SpaCy (Named Entity Recognition), Regex models, and the returned Vision AI masks into one multi-threaded candidate queue.
3. **Redaction Resolution (`ResolutionAgent`)**: Auto-corrects overlapping detection boundaries via algorithmic interval merging (`O(N log N)`) mapped accurately up to the exact pixel layer of the native PDF `FitZ` coordinate grid.
4. **Review & Reasoning (`ReviewAgent`)**: Offloads deeply contextual borderline candidates to LLM chains, storing decisions actively in Qdrant persistent storage.
5. **PDF Assembly (`MaskingAgent`)**: Instantly overlays unified blackout rectangles inside the native document array structure (`page.draw_rect`), bypassing CPU-bound deflate re-compression delays to provide completely destructed output PDFs dynamically in under a minute flat.
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
