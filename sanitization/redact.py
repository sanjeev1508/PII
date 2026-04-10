import os
from datetime import datetime
from pathlib import Path
from typing import List

import fitz

from .schemas import EntityDetection


def safe_save_pdf(doc: fitz.Document, output_pdf: str) -> str:
    try:
        doc.save(output_pdf)
        return output_pdf
    except Exception as e:
        if "permission denied" not in str(e).lower():
            raise
    base, ext = os.path.splitext(output_pdf)
    fallback = f"{base}_{datetime.now().strftime('%Y%m%d_%H%M%S')}{ext or '.pdf'}"
    doc.save(fallback)
    return fallback


def apply_masking(doc: fitz.Document, entities: List[EntityDetection], mode: str = "redact") -> None:
    for ent in entities:
        page = doc[ent.page]
        rect = fitz.Rect(ent.bbox)
        if mode == "highlight":
            annot = page.add_highlight_annot(rect)
            annot.set_colors(stroke=(1, 1, 0))
            annot.update()
        else:
            page.add_redact_annot(rect, fill=(0, 0, 0))

    if mode != "highlight":
        for page in doc:
            page.apply_redactions()


def ensure_output_dirs(pdf_dir: str, report_dir: str, log_dir: str) -> None:
    Path(pdf_dir).mkdir(parents=True, exist_ok=True)
    Path(report_dir).mkdir(parents=True, exist_ok=True)
    Path(log_dir).mkdir(parents=True, exist_ok=True)

