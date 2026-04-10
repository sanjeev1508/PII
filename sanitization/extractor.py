"""
PDF extraction — OPTIMIZED for large PDFs.

Key fixes for large-page-count PDFs:
  1. Chunked OCR batching (OCR_CHUNK_SIZE pages at a time) — prevents RAM exhaustion
     that caused 94s+ hangs on large scanned docs
  2. JPEG temp files instead of PNG — 5x smaller, faster disk I/O during OCR
  3. Adaptive worker count — scales down for large PDFs to avoid memory pressure
  4. Progress callback — ExtractionAgent can stream per-chunk progress to SSE
  5. Explicit PIL image del after save — keeps peak RAM low
  6. Page-content hash cache — skip re-OCR for repeated pages within a session

Returns: [{page: int, text: str, method: str}]
"""
import concurrent.futures
import hashlib
import io
import math
import os
import threading
from functools import lru_cache
from pathlib import Path
from tempfile import TemporaryDirectory

import fitz
from PIL import Image

# ── Tuning (all overridable via env vars) ─────────────────────────────────────
MIN_NATIVE_CHARS = int(os.getenv("MIN_NATIVE_CHARS", "80"))
OCR_DPI          = int(os.getenv("OCR_DPI", "96"))   # Reduced from 150 -> 96 for speed
OCR_MAX_SIDE     = int(os.getenv("OCR_MAX_SIDE", "1024")) # Reduced from 1400 -> 1024
MAX_WORKERS      = int(os.getenv("EXTRACTOR_WORKERS", "6"))

# How many OCR pages to send to doctr in one batch.
# Larger = faster for small docs. For 100+ pages keep ≤15 to avoid RAM spikes.
OCR_CHUNK_SIZE   = int(os.getenv("OCR_CHUNK_SIZE", "12"))
# ─────────────────────────────────────────────────────────────────────────────

_predictor_ready = threading.Event()

# Global page-hash → text cache (lives for process lifetime; avoids re-OCR)
_page_cache: dict[str, str] = {}
_cache_lock = threading.Lock()


@lru_cache(maxsize=1)
def _get_predictor():
    """Load lightweight doctr model. Cached — loaded ONCE per process."""
    from doctr.models import ocr_predictor
    p = ocr_predictor(
        det_arch="db_mobilenet_v3_large",
        reco_arch="crnn_mobilenet_v3_small",
        pretrained=True,
        assume_straight_pages=True,   # skip rectification — big speedup
    )
    print("[EXTRACT] Doctr predictor ready (db_mobilenet_v3_large + crnn_mobilenet_v3_small)")
    _predictor_ready.set()
    return p


def prewarm():
    """
    BLOCKING prewarm — loads doctr fully before returning.
    Called by app startup so the first request has zero cold-start delay.
    """
    print("[EXTRACT] Loading doctr model (blocking startup prewarm)…")
    try:
        _get_predictor()   # lru_cache: loads once, fast on subsequent calls
        print("[EXTRACT] Doctr model ready.")
    except Exception as e:
        print(f"[EXTRACT] Prewarm failed (non-fatal): {e}")


# ── Helpers ───────────────────────────────────────────────────────────────────

def _page_hash(pix: fitz.Pixmap) -> str:
    return hashlib.md5(pix.samples).hexdigest()


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


def _render_page_to_jpeg(args):
    """
    Worker: render one fitz page → JPEG bytes in memory.
    JPEG is ~5x smaller than PNG → less disk I/O during batch OCR.
    Returns (page_idx, jpeg_bytes_or_None, hash, was_cached).
    """
    page_idx, pdf_path, dpi, max_side = args
    try:
        doc = fitz.open(pdf_path)
        mat = fitz.Matrix(dpi / 72.0, dpi / 72.0)
        pix = doc[page_idx].get_pixmap(matrix=mat, alpha=False)
        h = _page_hash(pix)

        with _cache_lock:
            if h in _page_cache:
                doc.close()
                return page_idx, _page_cache[h], h, True  # cache hit

        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        img = _resize(img, max_side)
        doc.close()

        # Save to JPEG bytes in memory — no disk write yet
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=85, optimize=False)
        jpeg_bytes = buf.getvalue()
        del img, buf  # release RAM immediately

        return page_idx, jpeg_bytes, h, False
    except Exception as e:
        print(f"[EXTRACT] Render error page {page_idx}: {e}")
        return page_idx, None, "", False


# ── Main entry point ──────────────────────────────────────────────────────────

def extract_pages(
    pdf_path: str,
    progress_cb=None,   # optional callback(done_pages, total_ocr_pages)
) -> list[dict]:
    """
    Fast-path: parallel native PyMuPDF text per page.
    Fallback: parallel JPEG render → chunked doctr batches for OCR pages.
    progress_cb(done, total) is called after each OCR chunk.
    """
    import time
    t0 = time.perf_counter()
    print("[EXTRACT] Starting parallel extraction")

    doc = fitz.open(pdf_path)
    n_pages = len(doc)
    doc.close()

    # Adapt worker count for very large PDFs (avoid memory pressure)
    workers = min(MAX_WORKERS, max(2, n_pages // 10))

    # ── STEP 1: Read all pages in parallel ───────────────────────────────────
    results: list[dict] = [None] * n_pages
    ocr_needed: list[int] = []

    args = [(i, pdf_path) for i in range(n_pages)]
    with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
        for page_idx, text in pool.map(_read_native_page, args):
            if len(text) >= MIN_NATIVE_CHARS:
                results[page_idx] = {"page": page_idx, "text": text, "method": "native"}
            else:
                results[page_idx] = {"page": page_idx, "text": "", "method": "ocr"}
                ocr_needed.append(page_idx)

    native_count = n_pages - len(ocr_needed)
    print(f"[EXTRACT] {native_count} native, {len(ocr_needed)} OCR  "
          f"[{time.perf_counter()-t0:.2f}s]")

    # ── STEP 2: Chunked OCR (prevents RAM spikes on large docs) ──────────────
    if ocr_needed:
        ocr_results = _chunked_ocr(pdf_path, ocr_needed, workers, progress_cb)
        for pg, text in ocr_results.items():
            results[pg]["text"] = text

    print(f"[EXTRACT] Done in {time.perf_counter()-t0:.2f}s")
    return results


# ── Chunked OCR pipeline ──────────────────────────────────────────────────────

def _chunked_ocr(
    pdf_path: str,
    page_nums: list[int],
    workers: int,
    progress_cb=None,
) -> dict[int, str]:
    """
    Process OCR pages in chunks of OCR_CHUNK_SIZE.
    - Each chunk: parallel JPEG render → doctr inference → free RAM
    - Prevents the OOM / 90s hangs seen on large scanned PDFs
    """
    predictor = _get_predictor()
    results: dict[int, str] = {}
    total = len(page_nums)
    done = 0

    # Split into chunks
    chunks = [page_nums[i:i + OCR_CHUNK_SIZE]
              for i in range(0, total, OCR_CHUNK_SIZE)]

    print(f"[EXTRACT] OCR: {total} pages in {len(chunks)} chunks of ≤{OCR_CHUNK_SIZE}")

    for chunk_idx, chunk in enumerate(chunks):
        chunk_t = __import__("time").perf_counter()

        render_args = [(pg, pdf_path, OCR_DPI, OCR_MAX_SIDE) for pg in chunk]

        # Parallel JPEG render for this chunk
        cache_hits: dict[int, str] = {}
        to_infer: list[tuple[int, bytes, str]] = []  # (page_idx, jpeg_bytes, hash)

        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            for page_idx, data, h, was_cached in pool.map(_render_page_to_jpeg, render_args):
                if was_cached:
                    cache_hits[page_idx] = data
                elif data is not None:
                    to_infer.append((page_idx, data, h))
                else:
                    cache_hits[page_idx] = ""

        results.update(cache_hits)

        # Doctr inference on this chunk (write JPEG files, infer, clean up)
        if to_infer:
            with TemporaryDirectory() as tmp:
                paths = []
                meta = []
                for i, (page_idx, jpeg_bytes, h) in enumerate(to_infer):
                    p = Path(tmp) / f"p{i}.jpg"
                    p.write_bytes(jpeg_bytes)
                    paths.append(str(p))
                    meta.append((page_idx, h))
                    del jpeg_bytes  # release memory immediately

                try:
                    from doctr.io import DocumentFile
                    ocr_out = predictor(DocumentFile.from_images(paths))
                    for i, ocr_page in enumerate(ocr_out.pages):
                        lines = [
                            " ".join(w.value for w in ln.words if w.value)
                            for blk in ocr_page.blocks
                            for ln in blk.lines
                        ]
                        text = "\n".join(lines).strip()
                        page_idx, h = meta[i]
                        results[page_idx] = text
                        if h:
                            with _cache_lock:
                                _page_cache[h] = text
                except Exception as e:
                    print(f"[EXTRACT] OCR chunk {chunk_idx} failed: {e}")
                    for page_idx, _ in meta:
                        results[page_idx] = ""

        done += len(chunk)
        elapsed = __import__("time").perf_counter() - chunk_t
        print(f"[EXTRACT] Chunk {chunk_idx+1}/{len(chunks)} done  "
              f"({done}/{total} pages)  [{elapsed:.1f}s]")

        if progress_cb:
            try:
                progress_cb(done, total)
            except Exception:
                pass

    return results


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
    """OCR raster images embedded inside the PDF."""
    out = []
    try:
        predictor = _get_predictor()
        doc = fitz.open(pdf_path)
        paths, meta = [], []

        with TemporaryDirectory() as tmp:
            for pg in range(len(doc)):
                for idx, info in enumerate(doc[pg].get_images(full=True)):
                    xref = info[0]
                    img_data = doc.extract_image(xref)
                    pil_img = Image.open(io.BytesIO(img_data["image"])).convert("RGB")
                    
                    # 10-K and financial PDFs contain hundreds of 1x1 or tiny pixel spacers/lines.
                    # OCR'ing these creates massive bottlenecks. Skip images smaller than 60x60.
                    if pil_img.width < 60 or pil_img.height < 60:
                        del pil_img
                        continue

                    pil_img = _resize(pil_img, OCR_MAX_SIDE)
                    p = Path(tmp) / f"img_{pg}_{idx}.jpg"
                    pil_img.save(p, format="JPEG", quality=85)
                    del pil_img
                    paths.append(str(p))
                    meta.append({"page": pg, "img_idx": idx})

            if paths:
                from doctr.io import DocumentFile
                ocr_out = predictor(DocumentFile.from_images(paths))
                for i, ocr_page in enumerate(ocr_out.pages):
                    lines = [
                        " ".join(w.value for w in ln.words if w.value)
                        for blk in ocr_page.blocks for ln in blk.lines
                    ]
                    text = "\n".join(lines).strip()
                    if text:
                        out.append({**meta[i], "text": text})
        doc.close()
    except Exception as e:
        print(f"[EXTRACT] Embedded image OCR error: {e}")
    return out


def _resize(img: Image.Image, max_side: int) -> Image.Image:
    w, h = img.size
    if max(w, h) <= max_side:
        return img
    ratio = max_side / max(w, h)
    return img.resize((max(1, int(w * ratio)), max(1, int(h * ratio))), Image.Resampling.LANCZOS)
