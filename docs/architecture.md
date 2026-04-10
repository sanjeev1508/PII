# Architecture Diagram

```mermaid
flowchart TD
    A[Client Uploads PDF] --> B[FastAPI /sanitize]
    B --> C[Extract Pages]
    C --> D[Detect Entities]
    D --> E{Score Bucket}

    E -->|score < min_confidence| F[Reject Candidate]
    E -->|score >= confidence_threshold| G[Mask Directly]
    E -->|min <= score < max| H[LLM Batch Review]

    H -->|SENSITIVE| G
    H -->|NOT_SENSITIVE| F
    H -->|Review Error + fail_open_on_error=true| G

    G --> I[Masking Stage]
    I --> K[Local PyMuPDF Redaction]
    K --> L[Generate Report PDF]

    F --> L
    L --> M[Bundle masked.pdf + report.pdf]
    M --> N[ZIP Response]
```

## Notes

- NLP detection backend: Presidio + spaCy `en_core_web_lg`
- Borderline candidates are reviewed using direct OpenAI batch classification
- Report includes direct-threshold and LLM review outcomes
