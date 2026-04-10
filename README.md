# PDF Sanitization API

FastAPI backend for document sanitization with native PDF extraction, OCR fallback, configurable entity detection, validation, masking, and report generation.

The system now runs with an agentic orchestration layer that coordinates hybrid spaCy/Presidio detection, memory lookup in Qdrant, and LLM judging.

Hybrid routing rule:
- candidates with detector confidence `> 0.90` are masked directly (spaCy/Presidio path)
- candidates with confidence `> 0.50` and `<= 0.90` are routed to the async LLM agent
- all candidates with confidence `> 0.50` are stored in Qdrant

OCR behavior:
- native extraction is used when page text is available
- OCR uses Doctr only (pages rendered with PDFium before recognition)
- OCR pages are processed in parallel worker threads (set `OCR_WORKERS` env var, default `4`)

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
- score between min and max threshold: send candidates to batch LLM fallback (`MASK` vs `KEEP`) and mask only `MASK`.

The fallback also reads:

- `SKILLS.md` (description, role, references)
- `outputs/llm/reference_scores.json` (local confidence summary generated after detection)

LLM JSON decisions are added to the generated `report.pdf`.

## Qdrant Agent (Unreported High-Confidence Candidates)

The pipeline also stores candidates in local Qdrant when they:

- have confidence above `qdrant.min_confidence_percent` (default `50`)
- are `REJECTED` in review flow
- are not part of the LLM-reviewed section in `report.pdf`

Agent pipeline behavior:

- sends all candidates above confidence threshold (default `> 50%`) asynchronously to OpenAI for PII acceptance
- writes acceptance fields per payload:
  - `accepted_as_pii` (boolean)
  - `pii_decision` (`accepted_pii` or `not_accepted`)
  - `pii_reason` (short model reason)
- stores enriched payloads into local Qdrant
- masks only `accepted_pii` rows
- adds all reviewed rows (`confidence > 50`) and accepted rows into final `report.pdf`

Configure in `configs/entities.yaml`:

- `orchestration.direct_spacy_confidence`
- `qdrant.enabled`
- `qdrant.url` (default `http://localhost:6333`)
- `qdrant.collection`
- `qdrant.min_confidence_percent`
- `qdrant.agent_model`
- `qdrant.agent_batch_size`

Optional env overrides:

- `QDRANT_URL`
- `QDRANT_COLLECTION`
- `QDRANT_MIN_CONFIDENCE_PERCENT`

### Performance Notes (Large PDFs)

Hybrid mode can become slow if every borderline candidate triggers a separate remote call. To keep runtime efficient:

- Use `llm_review.review_entity_types` to review only ambiguous entities (for example `PARTY_NAME`, `ADDRESS`, `JURISDICTION`).
- Keep deterministic entities like `FINANCIAL_AMOUNT` on direct threshold masking (adjust threshold accordingly).
- Use `llm_review.batch_size` to control batch request size.
- Use `llm_review.max_reviews_per_document` to cap review calls per document.
- Repeated values are cached per document, so identical spans are reviewed once.

## Output Directory

Keep `outputs/pdfs/` in the repository for sanitized PDF outputs/workflow compatibility.
