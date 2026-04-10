# PDF Sanitization API

FastAPI backend for document sanitization with native PDF extraction, OCR fallback, configurable entity detection, validation, masking, and report generation.

## Pipeline Stages

- `[EXTRACT]` Native text extraction by page with OCR fallback for low-text pages.
- `[DETECT]` Presidio + spaCy + custom recognizers with validation filters.
- `[MASK]` PDF redaction using PyMuPDF text search and redact annotations.
- `[REPORT]` PDF report generation and final ZIP bundling.

## Architecture

See the full architecture diagram in `docs/architecture.md`.

## Configuration

Entity behavior is controlled in `configs/entities.yaml`:

- Enable/disable each entity type.
- Confidence threshold per type.
- Optional validators (Luhn, address formatting).
- Mask templates.
- Optional LLM review for borderline scores (`llm_review` block).

`FINANCIAL_AMOUNT` is the canonical amount entity type.

## Run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

### Environment

Create a `.env` file in project root (`OCR/`) when using LLM review:

```env
OPENAI_API_KEY=your_openai_key
```

Optional MCP masking backend:

```env
MASK_WITH_MCP=true
MCP_SERVER_URL=http://localhost:3003
MCP_MASK_TOOL=mask_pdf
USE_MCP_LLM=true
MCP_LLM_TOOL=openai_sensitive_review
```

To run the MCP masking and OpenAI review server on port 3003:

```bash
pip install -r requirements.txt
python -m sanitization.mcp_server
```

This server exposes:
- `POST /mask_pdf`
- `POST /openai_sensitive_review`
- `POST /call_tool`

The local sanitizer will use `fast_mcp.Client` first when `MASK_WITH_MCP=true`, and can also call the OpenAI LLM review through `mcp_use` when `USE_MCP_LLM=true`.

The generated PDF report now includes separate LLM review annotations for borderline entities that required classification.

## API

- `GET /health`
- `POST /sanitize` (multipart form-data, file field name: `file`)

Example:

```bash
curl -X POST "http://localhost:8000/sanitize" ^
  -H "accept: application/zip" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@your.pdf" ^
  --output sanitized_bundle.zip
```

Response ZIP contains:

- `masked.pdf`
- `report.pdf`

## Threshold + LLM Review Behavior

For each entity candidate:

- score `< llm_review.min_confidence`: reject.
- score `>= confidence_threshold`: mask directly.
- score between min and max threshold: ask LLM classifier (`SENSITIVE` vs `NOT_SENSITIVE`), and mask only when classified as `SENSITIVE`.

## Output Directory

Keep `outputs/pdfs/` in the repository for sanitized PDF outputs/workflow compatibility.
