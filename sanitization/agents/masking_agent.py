"""Masking agent — single-pass redact+manifest, lightweight in-memory verify."""
import asyncio
import os
import fitz
from sanitization.agents.base_agent import BaseAgent
from sanitization.pipeline_context import PipelineContext
from sanitization.masker import mask_pdf


class MaskingAgent(BaseAgent):
    name = "MaskingAgent"
    max_retries = 1

    async def _execute(self, ctx: PipelineContext) -> PipelineContext:
        await ctx.events.put({"type": "progress", "agent": self.name,
                               "msg": "Applying redactions (single-pass)…"})
        masked_path = os.path.join(ctx.work_dir, "masked.pdf")

        # Single pass: build manifest + apply redactions + save
        manifest = await asyncio.to_thread(
            mask_pdf, ctx.input_path, masked_path, ctx.final_entities
        )
        ctx.redaction_manifest = manifest
        ctx.masked_path = masked_path

        # Lightweight in-memory verify — spot-check up to 20 entities, no extra fitz.open
        missed = await asyncio.to_thread(self._verify_sample, masked_path, ctx.final_entities)
        if missed:
            print(f"[MaskingAgent] Re-redacting {len(missed)} missed entities")
            await asyncio.to_thread(self._reredact, masked_path, missed)

        await ctx.events.put({
            "type": "progress",
            "agent": self.name,
            "msg": f"{len(ctx.final_entities)} entities redacted",
        })
        return ctx

    # ── Verification (sample-based, not full scan) ────────────────────────────

    def _verify_sample(self, masked_path: str, entities: list, sample_size: int = 20) -> list:
        """
        Spot-check a sample of entities instead of scanning all 100 pages.
        If a sampled entity is still visible → mark for re-redaction.
        Returns list of missed entities (from full scan only if sample has misses).
        """
        if not entities:
            return []
        missed = []
        try:
            doc = fitz.open(masked_path)
            # Sample up to `sample_size` entities evenly
            step = max(1, len(entities) // sample_size)
            sample = entities[::step][:sample_size]
            sample_missed = any(
                (val := (e.get("value") or "").strip())
                and e.get("page", 0) < len(doc)
                and doc[e["page"]].search_for(val)
                for e in sample
            )
            if sample_missed:
                # Full scan only if sample found a problem
                for ent in entities:
                    val = (ent.get("value") or "").strip()
                    pg = ent.get("page", 0)
                    if val and pg < len(doc) and doc[pg].search_for(val):
                        missed.append(ent)
            doc.close()
        except Exception as e:
            print(f"[MaskingAgent] Verify error: {e}")
        return missed

    def _reredact(self, masked_path: str, entities: list) -> None:
        doc = fitz.open(masked_path)
        by_page: dict = {}
        for ent in entities:
            by_page.setdefault(ent.get("page", 0), []).append(ent)
        for pg, ents in by_page.items():
            if pg >= len(doc):
                continue
            page = doc[pg]
            for ent in ents:
                for rect in page.search_for(ent.get("value", "")):
                    page.add_redact_annot(rect, fill=(0, 0, 0))
            page.apply_redactions()
        doc.save(masked_path, garbage=4, deflate=True)
        doc.close()
