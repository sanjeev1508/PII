"""
PDF text extraction with native mode first, OCR fallback for scanned pages.
Returns a list of dicts: [{page: int, text: str, method: str}]
"""
import fitz  # pymupdf
import pdfplumber  # noqa: F401
from pdf2image import convert_from_path
import pytesseract


MIN_CHARS_NATIVE = 50  # If a page has fewer chars natively, use OCR


def extract_pages(pdf_path: str) -> list[dict]:
    results = []
    print("[EXTRACT] Starting extraction")
    doc = fitz.open(pdf_path)

    for page_num in range(len(doc)):
        page = doc[page_num]
        native_text = page.get_text("text").strip()

        if len(native_text) >= MIN_CHARS_NATIVE:
            results.append({"page": page_num, "text": native_text, "method": "native"})
        else:
            ocr_text = _ocr_page(pdf_path, page_num)
            results.append({"page": page_num, "text": ocr_text, "method": "ocr"})

    doc.close()
    print("[EXTRACT] Extraction complete")
    return results


def _ocr_page(pdf_path: str, page_num: int) -> str:
    try:
        images = convert_from_path(pdf_path, dpi=300, first_page=page_num + 1, last_page=page_num + 1)
        if images:
            return pytesseract.image_to_string(images[0])
    except Exception as e:
        print(f"[EXTRACT] OCR ERROR page {page_num}: {e}")
    return ""
