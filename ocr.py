import fitz  # PyMuPDF
import numpy as np
from PIL import Image
from doctr.models import ocr_predictor
import multiprocessing as mp

# -------------------------------
# CONFIG
# -------------------------------
PDF_PATH = r"F:\Downloads\pdfs-20260410T075826Z-3-001\pdfs\ACTIVISIONBLIZZARD_2015_10K.pdf"
OUTPUT_FILE = "ocr_output.txt"
NUM_WORKERS = max(1, mp.cpu_count() - 1)


# -------------------------------
# STEP 1: Convert PDF → Images
# -------------------------------
def pdf_to_images(pdf_path):
    doc = fitz.open(pdf_path)
    images = []

    for page in doc:
        pix = page.get_pixmap(dpi=300)  # high quality
        img = Image.frombytes("RGB", [pix.width, pix.height], pix.samples)
        images.append(np.array(img))

    return images


# -------------------------------
# STEP 2: Worker function
# Each process loads its own model
# -------------------------------
def process_chunk(image_chunk):
    model = ocr_predictor(
        det_arch='db_resnet50',
        reco_arch='crnn_mobilenet_v3_large',
        pretrained=True
    )

    result = model(image_chunk)
    return result.render()


# -------------------------------
# STEP 3: Split into chunks
# -------------------------------
def chunkify(lst, n):
    return [lst[i::n] for i in range(n)]


# -------------------------------
# STEP 4: Parallel OCR
# -------------------------------
def parallel_ocr(images):
    chunks = chunkify(images, NUM_WORKERS)

    with mp.Pool(NUM_WORKERS) as pool:
        results = pool.map(process_chunk, chunks)

    return "\n".join(results)


# -------------------------------
# MAIN PIPELINE
# -------------------------------
if __name__ == "__main__":
    print("🔹 Converting PDF to images...")
    images = pdf_to_images(PDF_PATH)

    print(f"🔹 Total pages: {len(images)}")
    print(f"🔹 Using {NUM_WORKERS} workers")

    print("🔹 Running OCR in parallel...")
    ocr_text = parallel_ocr(images)

    print("🔹 Saving output...")
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(ocr_text)

    print("✅ OCR completed! Output saved to:", OUTPUT_FILE)