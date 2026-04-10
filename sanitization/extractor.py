"""
PDF text extraction with native mode first, doctr OCR fallback for scanned pages.
Returns a list of dicts: [{page: int, text: str, method: str}]
"""
import concurrent.futures
from functools import lru_cache
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import threading

import fitz  # pymupdf
import pypdfium2 as pdfium
from PIL import Image


MIN_NATIVE_TEXT_CHARS = 5000000000   # chars below which we fall back to OCR
OCR_DPI = 120
OCR_MAX_SIDE = 2200
OCR_WORKERS = max(1, int(os.getenv("OCR_WORKERS", os.cpu_count() or 4)))
_THREAD_LOCAL = threading.local()


def extract_pages(pdf_path: str) -> list[dict]:
    results: list[dict] = []
    print("[EXTRACT] Starting extraction")
    doc = fitz.open(pdf_path)
    ocr_pages: list[int] = []

    for page_num in range(len(doc)):
        page = doc[page_num]
        native_text = page.get_text("text").strip()

        if len(native_text) >= MIN_NATIVE_TEXT_CHARS:
            results.append({"page": page_num, "text": native_text, "method": "native"})
        else:
            ocr_pages.append(page_num)
            results.append({"page": page_num, "text": "", "method": "ocr"})

    doc.close()
    if ocr_pages:
        ocr_results = _ocr_pages_parallel(pdf_path, ocr_pages)
        by_page = {row["page"]: row["text"] for row in ocr_results}
        for row in results:
            if row["method"] == "ocr":
                row["text"] = by_page.get(row["page"], "")

    print("[EXTRACT] Extraction complete")
    return results


def _ocr_pages_parallel(pdf_path: str, page_numbers: list[int]) -> list[dict]:
    out: list[dict] = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=OCR_WORKERS) as executor:
        futures = {executor.submit(_ocr_page, pdf_path, page_num): page_num for page_num in page_numbers}
        for fut in concurrent.futures.as_completed(futures):
            page_num = futures[fut]
            try:
                text = fut.result()
            except Exception as e:
                print(f"[EXTRACT] OCR worker failed page {page_num}: {e}")
                text = ""
            out.append({"page": page_num, "text": text})
    return sorted(out, key=lambda x: x["page"])


def _get_thread_pdf_document(pdf_path: str):
    pdf = getattr(_THREAD_LOCAL, "pdf", None)
    if pdf is not None and getattr(_THREAD_LOCAL, "pdf_path", None) == pdf_path:
        return pdf
    try:
        if pdf is not None:
            pdf.close()
    except Exception:
        pass
    pdf = pdfium.PdfDocument(str(pdf_path))
    _THREAD_LOCAL.pdf = pdf
    _THREAD_LOCAL.pdf_path = pdf_path
    return pdf


def _ocr_page(pdf_path: str, page_num: int) -> str:
    predictor = _get_thread_doctr_predictor()
    if predictor is None:
        return ""
    try:
        pdf = _get_thread_pdf_document(pdf_path)
        scale = max(OCR_DPI / 72.0, 1.0)
        page_img = pdf[page_num].render(scale=scale).to_pil().convert("RGB")
        ocr_img = _resize_for_ocr(page_img, OCR_MAX_SIDE)
        return _doctr_extract_text_from_image(ocr_img, predictor, page_num)
    except Exception as e:
        print(f"[EXTRACT] OCR ERROR page {page_num}: {e}")
    return ""


@lru_cache(maxsize=1)
def _get_ocr_predictor_factory():
    from doctr.models import ocr_predictor
    return ocr_predictor


def _get_thread_doctr_predictor():
    try:
        predictor = getattr(_THREAD_LOCAL, "predictor", None)
        if predictor is not None:
            return predictor
        predictor_factory = _get_ocr_predictor_factory()
        predictor = predictor_factory(pretrained=True)
        _THREAD_LOCAL.predictor = predictor
        return predictor
    except Exception as e:
        print(f"[EXTRACT] Doctr unavailable: {e}")
        return None


def _resize_for_ocr(img: Image.Image, max_side: int) -> Image.Image:
    if max_side <= 0:
        return img
    width, height = img.size
    largest = max(width, height)
    if largest <= max_side:
        return img
    ratio = max_side / float(largest)
    new_w = max(1, int(width * ratio))
    new_h = max(1, int(height * ratio))
    return img.resize((new_w, new_h), Image.Resampling.LANCZOS)


def _doctr_extract_text_from_image(image: Image.Image, predictor, page_num: int) -> str:
    try:
        from doctr.io import DocumentFile

        with TemporaryDirectory() as tmp:
            page_path = Path(tmp) / f"page_{page_num}.png"
            image.save(page_path)
            doc = DocumentFile.from_images([str(page_path)])
            result = predictor(doc)

        lines = []
        ocr_page = result.pages[0] if result.pages else None
        if ocr_page is None:
            return ""
        for block in ocr_page.blocks:
            for line in block.lines:
                words = [word.value for word in line.words if word.value]
                if words:
                    lines.append(" ".join(words))
        text = "\n".join(lines).strip()
        if text:
            print(f"[EXTRACT] Doctr OCR used on page {page_num}")
        return text
    except Exception as e:
        print(f"[EXTRACT] Doctr OCR ERROR page {page_num}: {e}")
        return ""
