"""
Apply masking to a PDF — OPTIMIZED: single-pass that builds manifest + applies
redactions simultaneously, eliminating two extra fitz.open() traversals.
"""
import fitz


def mask_pdf(
    input_pdf_path: str,
    output_pdf_path: str,
    entities: list[dict],
) -> dict:
    """
    Single-pass redaction:
    - Searches for each entity value on its page
    - Collects bbox rects into a manifest dict
    - Applies redaction annotations
    - Saves once
    Returns the redaction manifest (page -> list of {type, value, rects}).
    """
    print("[MASK] Starting single-pass PDF masking")
    doc = fitz.open(input_pdf_path)
    manifest: dict[str, list] = {}

    # Group entities by page for efficient traversal
    entities_by_page: dict[int, list] = {}
    for ent in entities:
        entities_by_page.setdefault(ent.get("page", 0), []).append(ent)

    for page_num, page_entities in entities_by_page.items():
        if page_num >= len(doc):
            continue
        page = doc[page_num]
        page_key = str(page_num)

        for ent in page_entities:
            # ── 1. Image-based redaction (Full bounding box mapping)
            if "xref" in ent:
                try:
                    rects = page.get_image_rects(ent["xref"])
                    for r in rects:
                        page.add_redact_annot(r, fill=(0, 0, 0))
                    manifest.setdefault(page_key, []).append({
                        "type": ent.get("type"),
                        "value": ent.get("value"),
                        "confidence": ent.get("confidence", 0),
                        "rects": [[r.x0, r.y0, r.x1, r.y1] for r in rects],
                    })
                except Exception as e:
                    print(f"[MASK] Failed to redact image xref {ent['xref']}: {e}")
                continue

            # ── 2. Standard Text redaction
            value = (ent.get("value") or "").strip()
            if not value:
                continue

            instances = page.search_for(value)
            if not instances:
                continue

            # Build manifest entry
            manifest.setdefault(page_key, []).append({
                "type": ent.get("type"),
                "value": value,
                "confidence": ent.get("confidence", 0),
                "rects": [[r.x0, r.y0, r.x1, r.y1] for r in instances],
            })

            # Apply redaction annotations
            for rect in instances:
                page.add_redact_annot(rect, fill=(0, 0, 0))

        page.apply_redactions()

    # Removed garbage=4 and deflate=True. These operations aggressively recompress
    # and deduplicate the entire PDF stream natively in C, which takes 80-90s on 
    # 100+ page documents. Dropping them trades a few MB of file size for a 50x speedup.
    doc.save(output_pdf_path)
    doc.close()
    print("[MASK] Masking complete")
    return manifest
