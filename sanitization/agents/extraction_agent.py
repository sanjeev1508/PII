"""Extraction agent — chunked OCR with per-chunk SSE progress streaming."""
import asyncio
from sanitization.agents.base_agent import BaseAgent
from sanitization.pipeline_context import PipelineContext
from sanitization.extractor import extract_pages, extract_tables, extract_embedded_images


class ExtractionAgent(BaseAgent):
    name = "ExtractionAgent"
    max_retries = 1

    async def _execute(self, ctx: PipelineContext) -> PipelineContext:
        await ctx.events.put({"type": "progress", "agent": self.name,
                               "msg": "Extracting text from all pages…"})

        # Progress callback → pushes per-OCR-chunk events to SSE stream
        loop = asyncio.get_event_loop()

        def _ocr_progress(done: int, total: int):
            pct = int(done / total * 100)
            msg = f"OCR: {done}/{total} pages ({pct}%)"
            loop.call_soon_threadsafe(
                ctx.events.put_nowait,
                {"type": "progress", "agent": self.name, "msg": msg}
            )

        # ── Step 1: Fire all extractors concurrently ──────────────────────────
        try:
            pages, tables, image_texts = await asyncio.gather(
                asyncio.to_thread(extract_pages, ctx.input_path, _ocr_progress),
                asyncio.to_thread(extract_tables, ctx.input_path),
                asyncio.to_thread(extract_embedded_images, ctx.input_path),
                return_exceptions=True,
            )
        except Exception as e:
            print(f"[ExtractionAgent] Concurrent gather error: {e}")
            pages, tables, image_texts = [], [], []

        if isinstance(pages, Exception):
            print(f"[ExtractionAgent] Page extraction failed: {pages}")
            pages = []
        if isinstance(tables, Exception):
            print(f"[ExtractionAgent] Table extraction failed: {tables}")
            tables = []
        if isinstance(image_texts, Exception):
            print(f"[ExtractionAgent] Image OCR failed: {image_texts}")
            image_texts = []

        ctx.pages = pages
        ctx.total_pages = len(pages)
        native_ct = sum(1 for p in pages if p.get("method") == "native")
        ocr_ct = len(pages) - native_ct
        ctx.extraction_mode_summary = {"native": native_ct, "ocr": ocr_ct}
        
        await ctx.events.put({
            "type": "progress", "agent": self.name,
            "msg": f"{len(pages)} pages extracted ({native_ct} native, {ocr_ct} OCR)"
        })

        ctx.tables = tables
        ctx.image_texts = image_texts

        if tables:
            await ctx.events.put({"type": "progress", "agent": self.name,
                                   "msg": f"{len(tables)} tables extracted"})
        if image_texts:
            await ctx.events.put({"type": "progress", "agent": self.name,
                                   "msg": f"{len(image_texts)} embedded images OCR'd"})

        # ── Step 2: Augment page text ──────────────────────────────────────────
        for tbl in ctx.tables:
            pg = tbl.get("page", 0)
            table_text = "\n".join(
                " | ".join(str(c or "") for c in row) for row in tbl.get("rows", [])
            )
            if pg < len(ctx.pages):
                ctx.pages[pg]["text"] += f"\n\n[TABLE]\n{table_text}"

        # Step 1.5: Async Vision LLM Classification
        from sanitization.llm_classifier import vision_classify_image

        async def _classify(img):
            if "base64" in img:
                ans = await asyncio.to_thread(vision_classify_image, img["base64"])
                img["decision"] = ans
                img.pop("base64", None)  # instantly free base64 payload to preserve RAM
                img["text"] = f"<VISION ML DECISION: {ans}>"
            return img

        if image_texts:
            await ctx.events.put({"type": "progress", "agent": self.name, "msg": f"Running Vision LLM on {len(image_texts)} image(s)..."})
            classified_images = await asyncio.gather(*[_classify(img) for img in image_texts])
        else:
            classified_images = []

        for img in classified_images:
            pg = img.get("page", 0)
            if pg < len(ctx.pages):
                if "images" not in ctx.pages[pg]:
                    ctx.pages[pg]["images"] = []
                ctx.pages[pg]["images"].append(img)

        return ctx
