# FastAPI Document Sanitization Backend

Upload one PDF and receive one ZIP containing:

- `masked.pdf` (securely redacted PDF)
- `report.pdf` (human-readable PDF report of detections)

## Run Backend

```bash
pip install -r requirements.txt
uvicorn app:app --host 0.0.0.0 --port 8000 --reload
```

## API

- `GET /health`
- `POST /sanitize` (multipart form-data with file field name: `file`)

Example:

```bash
curl -X POST "http://localhost:8000/sanitize" ^
  -H "accept: application/zip" ^
  -H "Content-Type: multipart/form-data" ^
  -F "file=@AES_2022_10K.pdf" ^
  --output sanitized_bundle.zip
```

Extract `sanitized_bundle.zip` to get `masked.pdf` and `report.pdf`.

