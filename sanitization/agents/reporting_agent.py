"""Reporting agent — generate PDF report."""
import asyncio
import os
from sanitization.agents.base_agent import BaseAgent
from sanitization.pipeline_context import PipelineContext
from sanitization.reporter import generate_report


class ReportingAgent(BaseAgent):
    name = "ReportingAgent"
    max_retries = 1

    async def _execute(self, ctx: PipelineContext) -> PipelineContext:
        await ctx.events.put({"type": "progress", "agent": self.name, "msg": "Generating PDF report…"})
        report_path = os.path.join(ctx.work_dir, "report.pdf")

        review_log = []
        for row in ctx.agent_review_rows:
            is_masked = str(row.get("pii_decision", "")) == "accepted_pii"
            route = str(row.get("route", "llm"))
            review_log.append({
                "page": int(row.get("page", -1)),
                "type": str(row.get("type", "")),
                "value": str(row.get("value", "")),
                "confidence": float(row.get("confidence", 0.0)),
                "decision": "MASKED" if is_masked else "REJECTED",
                "reason": "above_direct_spacy_threshold" if route == "direct_spacy" else ("llm_pii" if is_masked else "llm_not_pii"),
                "review_method": "THRESHOLD" if route == "direct_spacy" else "LLM",
                "llm_classification": None if route == "direct_spacy" else ("PII" if is_masked else "NOT_PII"),
            })

        llm_json_results = [
            {
                "id": str(r.get("id", "")),
                "entity_type": str(r.get("type", "")),
                "value": str(r.get("value", "")),
                "decision": "MASK" if str(r.get("pii_decision", "")) == "accepted_pii" else "KEEP",
                "reason": str(r.get("pii_reason", "")),
                "confidence": r.get("confidence", ""),
            }
            for r in ctx.agent_review_rows
        ]

        # Generate a page-wise extraction log for user verification
        extraction_path = os.path.join(ctx.work_dir, "extraction.txt")
        with open(extraction_path, "w", encoding="utf-8") as f:
            f.write("=== PAGE-WISE EXTRACTION VERIFICATION LOG ===\n\n")
            for page in ctx.pages:
                pg_idx = page.get("page", 0)
                method = page.get("method", "unknown")
                text = page.get("text", "")
                f.write(f"--- PAGE {pg_idx + 1} [{method.upper()}] ---\n")
                f.write(text.strip() + "\n\n")
                # Include explicit image metadata if available
                for img in page.get("images", []):
                    img_text = img.get("text", "").strip()
                    f.write(f"[EMBEDDED IMAGE - xref {img.get('xref')} | dimensions {img.get('w', '?')}x{img.get('h', '?')}]\n")
                    f.write(f"{img_text if img_text else '<EMPTY_OCR>'}\n\n")

        await asyncio.to_thread(
            generate_report,
            output_path=report_path,
            source_file=ctx.source_file,
            total_pages=ctx.total_pages,
            extraction_mode_summary=ctx.extraction_mode_summary,
            entities=ctx.final_entities,
            review_log=review_log,
            llm_json_results=llm_json_results,
            processing_seconds=sum(ctx.timings.values()),
            accepted_agent_rows=ctx.agent_review_rows,
            agent_review_rows=ctx.agent_review_rows,
        )
        ctx.report_path = report_path
        ctx.extraction_path = extraction_path
        await ctx.events.put({"type": "progress", "agent": self.name, "msg": "Report ready"})
        return ctx
