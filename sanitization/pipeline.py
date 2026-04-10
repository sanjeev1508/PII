import time
from pathlib import Path
from typing import Dict

import fitz

from .detect import detect_entities
from .extract import extract_words
from .redact import apply_masking, safe_save_pdf
from .report import build_summary, write_report


def sanitize_document(input_pdf: str, output_pdf: str, report_file: str, config: Dict) -> Dict:
    started = time.time()
    doc_id = Path(input_pdf).stem
    errors = []
    entities = []
    extraction_mode = "unknown"
    pages = 0

    doc = fitz.open(input_pdf)
    pages = len(doc)
    try:
        words_by_page, extraction_mode = extract_words(
            doc,
            input_pdf,
            native_text_min_words=int(config["extraction"]["native_text_min_words"]),
            use_ocr_fallback=bool(config["extraction"]["use_ocr_fallback"]),
            ocr_scale=int(config["extraction"]["ocr_scale"]),
        )
        entities = detect_entities(
            words_by_page,
            entity_types=config["detection"]["entity_types"],
        )
        apply_masking(doc, entities, mode=str(config["pipeline"]["mode"]).lower())
        saved_output = safe_save_pdf(doc, output_pdf)
    except Exception as e:
        errors.append(str(e))
        raise
    finally:
        doc.close()

    summary = build_summary(
        doc_id=doc_id,
        source_file=input_pdf,
        output_file=saved_output,
        pages=pages,
        extraction_mode=extraction_mode,
        started_at=started,
        entities=entities,
        errors=errors,
    )
    write_report(summary, report_file)

    return summary.to_dict()

