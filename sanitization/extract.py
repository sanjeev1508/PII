import os
import re
from collections import defaultdict
from typing import Dict, List, Tuple

import fitz

from .schemas import WordBox

os.environ["USE_TORCH"] = "1"

_DOCTR_MODEL = None


def normalize_token(token: str) -> str:
    return re.sub(r"[^\w@.+-]", "", token).strip().lower()


def extract_words_native(doc: fitz.Document) -> Tuple[Dict[int, List[WordBox]], str]:
    words_by_page: Dict[int, List[WordBox]] = defaultdict(list)
    total = 0
    for page_idx, page in enumerate(doc):
        for w in page.get_text("words", sort=True):
            x0, y0, x1, y1, token, block_no, line_no, word_no = w
            cleaned = normalize_token(token)
            if not cleaned:
                continue
            words_by_page[page_idx].append(
                WordBox(
                    raw_text=token,
                    text=cleaned,
                    bbox=(float(x0), float(y0), float(x1), float(y1)),
                    block=int(block_no),
                    line=int(line_no),
                    word_no=int(word_no),
                )
            )
            total += 1
    mode = "native" if total > 0 else "none"
    return words_by_page, mode


def _get_doctr_model():
    global _DOCTR_MODEL
    if _DOCTR_MODEL is None:
        from doctr.models import ocr_predictor

        _DOCTR_MODEL = ocr_predictor(
            det_arch="db_mobilenet_v3_large",
            reco_arch="crnn_mobilenet_v3_small",
            pretrained=True,
        )
    return _DOCTR_MODEL


def extract_words_ocr(input_pdf: str, scale: int = 1) -> Dict[int, List[WordBox]]:
    from doctr.io import DocumentFile

    model = _get_doctr_model()
    doc_doctr = DocumentFile.from_pdf(input_pdf, scale=scale)
    result = model(doc_doctr)
    words_by_page: Dict[int, List[WordBox]] = defaultdict(list)
    for page_idx, page in enumerate(result.pages):
        h, w = page.dimensions
        for block_idx, block in enumerate(page.blocks):
            for line_idx, line in enumerate(block.lines):
                for word_idx, word in enumerate(line.words):
                    x_min, y_min = word.geometry[0]
                    x_max, y_max = word.geometry[1]
                    token = word.value
                    cleaned = normalize_token(token)
                    if not cleaned:
                        continue
                    words_by_page[page_idx].append(
                        WordBox(
                            raw_text=token,
                            text=cleaned,
                            bbox=(
                                float(x_min * w),
                                float(y_min * h),
                                float(x_max * w),
                                float(y_max * h),
                            ),
                            block=block_idx,
                            line=line_idx,
                            word_no=word_idx,
                        )
                    )
    return words_by_page


def extract_words(
    doc: fitz.Document, input_pdf: str, native_text_min_words: int = 30, use_ocr_fallback: bool = True, ocr_scale: int = 1
) -> Tuple[Dict[int, List[WordBox]], str]:
    words_by_page, mode = extract_words_native(doc)
    total_native = sum(len(v) for v in words_by_page.values())
    if total_native >= native_text_min_words:
        return words_by_page, "native"
    if use_ocr_fallback:
        return extract_words_ocr(input_pdf, scale=ocr_scale), "ocr_fallback"
    return words_by_page, mode

