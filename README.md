# PDF Sanitization API

FastAPI backend for document sanitization with native PDF extraction, OCR fallback, configurable entity detection, validation, masking, and report generation.

## Pipeline Stages

- `[EXTRACT]` Native text extraction by page with OCR fallback for low-text pages.
- `[DETECT]` Presidio + spaCy + custom recognizers with validation filters.
- `[MASK]` PDF redaction using PyMuPDF text search and redact annotations.
- `[REPORT]` PDF report generation and final ZIP bundling.

## Configuration

Entity behavior is controlled in `configs/entities.yaml`:

- Enable/disable each entity type.
- Confidence threshold per type.
- Optional validators (Luhn, address formatting).
- Mask templates.

`FINANCIAL_AMOUNT` is the canonical amount entity type.

## Run

```bash
pip install -r requirements.txt
python -m spacy download en_core_web_lg
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

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

## Output Directory

Keep `outputs/pdfs/` in the repository for sanitized PDF outputs/workflow compatibility.
