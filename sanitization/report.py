import json
import time
from collections import Counter
from pathlib import Path
from typing import List

import fitz

from .schemas import EntityDetection, FileSummary


def build_summary(
    doc_id: str,
    source_file: str,
    output_file: str,
    pages: int,
    extraction_mode: str,
    started_at: float,
    entities: List[EntityDetection],
    errors: List[str] | None = None,
) -> FileSummary:
    counts = Counter([e.entity_type for e in entities])
    return FileSummary(
        doc_id=doc_id,
        source_file=source_file,
        output_file=output_file,
        pages=pages,
        extraction_mode=extraction_mode,
        processing_seconds=round(time.time() - started_at, 3),
        entity_counts=dict(counts),
        entities=entities,
        errors=errors or [],
    )


def write_report(summary: FileSummary, report_path: str) -> None:
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(summary.to_dict(), f, indent=2)


def append_jsonl(payload: dict, path: str) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=True) + "\n")


def write_report_pdf(summary: dict, output_pdf_path: str) -> None:
    Path(output_pdf_path).parent.mkdir(parents=True, exist_ok=True)
    doc = fitz.open()
    page = doc.new_page()

    lines = [
        "Document Sanitization Report",
        "",
        f"Document ID: {summary.get('doc_id', '')}",
        f"Source File: {summary.get('source_file', '')}",
        f"Pages: {summary.get('pages', 0)}",
        f"Extraction Mode: {summary.get('extraction_mode', '')}",
        f"Processing Seconds: {summary.get('processing_seconds', 0)}",
        "",
        "Entity Counts:",
    ]

    for k, v in (summary.get("entity_counts") or {}).items():
        lines.append(f"- {k}: {v}")

    lines.append("")
    lines.append("Detected Entities:")
    for idx, ent in enumerate(summary.get("entities") or [], start=1):
        line = (
            f"{idx}. page={ent.get('page')} "
            f"type={ent.get('entity_type')} "
            f"value={ent.get('value')} "
            f"formatted={ent.get('formatted')}"
        )
        lines.append(line[:1800])

    y = 50
    for ln in lines:
        if y > 800:
            page = doc.new_page()
            y = 50
        page.insert_text((50, y), ln, fontsize=10)
        y += 14

    doc.save(output_pdf_path)
    doc.close()

