"""
PDF extraction — native text and/or full-page docTR OCR.

EXTRACT_OCR_MODE:
  full_page (default) — Render every page at OCR_DPI, run docTR in batches (like a standalone
    fitz→PIL→numpy script). Text visible only inside images/logos on the page is included in OCR.
  native_first — Use PyMuPDF text when >= MIN_NATIVE_CHARS; docTR only for “empty” pages (faster
    on text-born PDFs but misses bitmap-only text on otherwise dense pages).

Embedded rasters are still OCR’d separately in extract_embedded_images (xref + masking).

Returns: [{page: int, text: str, method: str}]
"""
import concurrent.futures
import io
import os

import fitz
import numpy as np
from PIL import Image

# ── Tuning (all overridable via env vars) ─────────────────────────────────────
_EXTRACT_MODE_RAW = os.getenv("EXTRACT_OCR_MODE", "full_page").strip().lower()
if _EXTRACT_MODE_RAW not in ("full_page", "native_first"):
    print(f"[EXTRACT] Unknown EXTRACT_OCR_MODE={_EXTRACT_MODE_RAW!r}, using native_first")
    EXTRACT_OCR_MODE = "native_first"
else:
    EXTRACT_OCR_MODE = _EXTRACT_MODE_RAW
MIN_NATIVE_CHARS = int(os.getenv("MIN_NATIVE_CHARS", "80"))
OCR_DPI          = int(os.getenv("OCR_DPI", "150"))
# 0 = no resize (full pixmap like standalone script; uses more RAM)
OCR_MAX_SIDE     = int(os.getenv("OCR_MAX_SIDE", "0"))
CPU_COUNT        = os.cpu_count() or 4
MAX_WORKERS      = int(os.getenv("EXTRACTOR_WORKERS", str(min(8, CPU_COUNT))))
OCR_CHUNK_SIZE   = int(os.getenv("OCR_CHUNK_SIZE", "8"))
# ─────────────────────────────────────────────────────────────────────────────



# ── Helpers ───────────────────────────────────────────────────────────────────



def _read_native_page(args):
    """Worker: extract native text from one page (thread-safe fitz read)."""
    page_idx, pdf_path = args
    try:
        doc = fitz.open(pdf_path)
        text = doc[page_idx].get_text("text").strip()
        doc.close()
        return page_idx, text
    except Exception:
        return page_idx, ""




# ── Main entry point ──────────────────────────────────────────────────────────

def _run_doctr_on_page_indices(
    pdf_path: str,
    page_indices: list[int],
    progress_cb,
    total_for_progress: int,
) -> list[tuple[int, str]]:
    """Render selected pages and return [(page_idx, ocr_text), ...] in same order as indices."""
    from sanitization.doctr_ocr import run_batch_ocr

    out: list[tuple[int, str]] = []
    doc = fitz.open(pdf_path)
    try:
        done = 0
        for chunk_start in range(0, len(page_indices), OCR_CHUNK_SIZE):
            chunk_indices = page_indices[chunk_start : chunk_start + OCR_CHUNK_SIZE]
            arrs: list[np.ndarray] = []
            for pi in chunk_indices:
                page = doc[pi]
                pix = page.get_pixmap(dpi=OCR_DPI)
                pil_img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
                if OCR_MAX_SIDE > 0:
                    pil_img = _resize(pil_img, OCR_MAX_SIDE)
                arrs.append(np.asarray(pil_img, dtype=np.uint8))
                del pil_img
            texts = run_batch_ocr(arrs)
            for pi, txt in zip(chunk_indices, texts):
                out.append((pi, txt))
            done += len(chunk_indices)
            if progress_cb:
                progress_cb(done, total_for_progress)
    finally:
        doc.close()
    return out


def extract_pages(
    pdf_path: str,
    progress_cb=None,   # optional callback(done_pages, total_ocr_pages)
) -> list[dict]:
    """
    See EXTRACT_OCR_MODE at module top.

    If a page contains embedded images, we run docTR OCR on that page.
    Pages with no images are extracted via PyMuPDF text only, which is faster and preserves text quality.
    """
    import time
    t0 = time.perf_counter()
    print(f"[EXTRACT] Starting extraction (mode={EXTRACT_OCR_MODE!r})")

    doc = fitz.open(pdf_path)
    try:
        n_pages = len(doc)

        workers = min(MAX_WORKERS, max(2, n_pages // 10))
        results: list[dict] = [None] * n_pages
        ocr_needed: list[int] = []

        has_images = [bool(page.get_images(full=True)) for page in doc]

        if EXTRACT_OCR_MODE == "full_page":
            for i in range(n_pages):
                if has_images[i]:
                    results[i] = {"page": i, "text": "", "method": "ocr"}
                    ocr_needed.append(i)
                else:
                    results[i] = {"page": i, "text": doc[i].get_text("text").strip(), "method": "native"}
            print(
                f"[EXTRACT] full_page image-aware: {len(ocr_needed)} pages with images will use OCR, "
                f"{n_pages - len(ocr_needed)} pages use native text"
            )
        else:
            # native_first
            args = []
            for i in range(n_pages):
                if has_images[i]:
                    results[i] = {"page": i, "text": "", "method": "ocr"}
                    ocr_needed.append(i)
                else:
                    args.append((i, pdf_path))

            if args:
                with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
                    for page_idx, text in pool.map(_read_native_page, args):
                        if len(text) >= MIN_NATIVE_CHARS:
                            results[page_idx] = {"page": page_idx, "text": text, "method": "native"}
                        else:
                            results[page_idx] = {"page": page_idx, "text": "", "method": "ocr"}
                            ocr_needed.append(page_idx)

            print(
                f"[EXTRACT] native_first: {len(args)} image-free pages native-read, "
                f"{len(ocr_needed)} pages need OCR [{time.perf_counter()-t0:.2f}s]"
            )

        if ocr_needed:
            total_ocr = len(ocr_needed)
            pairs = _run_doctr_on_page_indices(pdf_path, ocr_needed, progress_cb, total_ocr)
            for pi, txt in pairs:
                results[pi]["text"] = txt
            print(f"[EXTRACT] docTR OCR {total_ocr} pages [{time.perf_counter()-t0:.2f}s elapsed]")

        print(f"[EXTRACT] Done in {time.perf_counter()-t0:.2f}s")
        return results
    finally:
        doc.close()


# ── Table extraction (Ultra-fast via PyMuPDF) ─────────────────────────────────

def extract_tables(pdf_path: str) -> list[dict]:
    """Extract tables using PyMuPDF (C-speed, ~100x faster than pdfplumber)."""
    out = []
    try:
        doc = fitz.open(pdf_path)
        for i, page in enumerate(doc):
            tabs = page.find_tables()
            if tabs and tabs.tables:
                for j, tbl in enumerate(tabs.tables):
                    rows = tbl.extract()
                    if rows:
                        out.append({"page": i, "table_index": j, "rows": rows})
        doc.close()
    except Exception as e:
        print(f"[EXTRACT] Table extraction error: {e}")
    return out


# ── Embedded image OCR ────────────────────────────────────────────────────────

def extract_embedded_images(pdf_path: str) -> list[dict]:
    """OCR embedded raster images with docTR (text only; no vision / base64 to LLM)."""
    metas: list[dict] = []
    arrs: list[np.ndarray] = []
    _emb_default = max(OCR_MAX_SIDE, 512) if OCR_MAX_SIDE > 0 else 1024
    max_side = int(os.getenv("EMBEDDED_OCR_MAX_SIDE", str(_emb_default)))
    try:
        doc = fitz.open(pdf_path)
        for pg in range(len(doc)):
            for idx, info in enumerate(doc[pg].get_images(full=True)):
                xref = info[0]
                img_data = doc.extract_image(xref)
                pil_img = Image.open(io.BytesIO(img_data["image"])).convert("RGB")
                if (pil_img.width < 30 and pil_img.height < 30) or pil_img.width <= 2 or pil_img.height <= 2:
                    del pil_img
                    continue
                if max_side > 0:
                    pil_img = _resize(pil_img, max_side)
                metas.append({
                    "page": pg,
                    "img_idx": idx,
                    "xref": xref,
                    "w": pil_img.width,
                    "h": pil_img.height,
                })
                arrs.append(np.asarray(pil_img, dtype=np.uint8))
                del pil_img
        doc.close()
    except Exception as e:
        print(f"[EXTRACT] Embedded image OCR prep error: {e}")
        return [{"page": 0, "xref": "CRASH_LOG", "w": 0, "h": 0, "text": f"SYSTEM_ERROR: {str(e)}"}]

    if not arrs:
        return []

    from sanitization.doctr_ocr import run_batch_ocr

    out: list[dict] = []
    for i in range(0, len(arrs), OCR_CHUNK_SIZE):
        chunk_metas = metas[i : i + OCR_CHUNK_SIZE]
        texts = run_batch_ocr(arrs[i : i + OCR_CHUNK_SIZE])
        for m, t in zip(chunk_metas, texts):
            row = dict(m)
            row["text"] = (t or "").strip()
            out.append(row)
    return out


def _resize(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    ratio = max_side / max(w, h)
    return img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.Resampling.LANCZOS)
