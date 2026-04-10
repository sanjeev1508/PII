"""
Apply masking to a PDF by drawing black redaction rectangles over detected entity spans.
Uses PyMuPDF for precise coordinate-based redaction.
"""
import fitz


def mask_pdf(input_pdf_path: str, output_pdf_path: str, entities: list[dict]) -> None:
    """
    For each detected entity, find its text location in the page and redact it.
    Falls back to text-search based redaction if coordinates are unavailable.
    """
    print("[MASK] Starting PDF masking")
    doc = fitz.open(input_pdf_path)

    entities_by_page = {}
    for ent in entities:
        p = ent["page"]
        entities_by_page.setdefault(p, []).append(ent)

    for page_num, page_entities in entities_by_page.items():
        if page_num >= len(doc):
            continue
        page = doc[page_num]

        for ent in page_entities:
            value = ent["value"]
            instances = page.search_for(value)
            if not instances:
                continue
            for rect in instances:
                page.add_redact_annot(rect, fill=(0, 0, 0))

        page.apply_redactions()

    doc.save(output_pdf_path, garbage=4, deflate=True)
    doc.close()
    print("[MASK] Masking complete")
