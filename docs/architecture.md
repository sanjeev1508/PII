# Architecture Diagram

```mermaid
flowchart TD
    A[Client Uploads PDF] --> B[FastAPI /sanitize]
    B --> C[Extract Pages]
    C --> D[Detect Entities]
    D --> E{Score Bucket}

    E -->|score < min_confidence| F[Reject Candidate]
    E -->|score >= confidence_threshold| G[Mask Directly]
    E -->|min <= score < max| H[LLM/MCP Review]

    H -->|SENSITIVE| G
    H -->|NOT_SENSITIVE| F
    H -->|Review Error + fail_open_on_error=true| G

    G --> I[Masking Stage]
    I -->|MASK_WITH_MCP=true| J[MCP mask_pdf Tool]
    I -->|fallback| K[Local PyMuPDF Redaction]

    J --> L[Generate Report PDF]
    K --> L

    F --> L
    L --> M[Bundle masked.pdf + report.pdf]
    M --> N[ZIP Response]
```

## Notes

- NLP detection backend: Presidio + spaCy `en_core_web_lg`
- Borderline candidates are reviewed using OpenAI (direct or via MCP)
- Report includes direct-threshold and LLM/MCP review outcomes
