"""MCP server for OCR masking and OpenAI sensitivity review tools."""
import os
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

try:
    import fast_mcp  # type: ignore
except ImportError:
    fast_mcp = None

from sanitization.masker import _mask_pdf_local
from sanitization.llm_classifier import _llm_is_sensitive_local

app = FastAPI(title="OCR MCP Server", version="1.0.0")


def _register_fast_mcp_tools() -> None:
    if fast_mcp is None:
        return
    if not hasattr(fast_mcp, "tool"):
        return

    try:
        @fast_mcp.tool(name="mask_pdf")
        def _fast_mcp_mask_pdf(input_pdf_path: str, output_pdf_path: str, entities: list[dict]) -> dict[str, Any]:
            _mask_pdf_local(input_pdf_path, output_pdf_path, entities)
            return {"status": "ok", "output_pdf_path": output_pdf_path}

        @fast_mcp.tool(name="openai_sensitive_review")
        def _fast_mcp_openai_review(
            entity_type: str,
            value: str,
            confidence: float,
            context: str,
            model: str,
        ) -> dict[str, Any]:
            result = _llm_is_sensitive_local(
                entity_type=entity_type,
                value=value,
                confidence=confidence,
                context=context,
                model=model,
            )
            return {"sensitive": result}

    except Exception as exc:
        print(f"[MCP] fast_mcp tool registration failed: {exc}")


_register_fast_mcp_tools()


class MaskPayload(BaseModel):
    input_pdf_path: str
    output_pdf_path: str
    entities: list[dict]


class LLMReviewPayload(BaseModel):
    entity_type: str
    value: str
    confidence: float
    context: str
    model: str


class ToolRequest(BaseModel):
    tool_name: str
    arguments: dict[str, Any]


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "server": "ocr-mcp",
        "fast_mcp_available": fast_mcp is not None,
    }


@app.post("/mask_pdf")
def mask_pdf_tool(payload: MaskPayload) -> dict[str, Any]:
    try:
        _mask_pdf_local(payload.input_pdf_path, payload.output_pdf_path, payload.entities)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "ok", "output_pdf_path": payload.output_pdf_path}


@app.post("/openai_sensitive_review")
def openai_sensitive_review_tool(payload: LLMReviewPayload) -> dict[str, Any]:
    try:
        sensitive = _llm_is_sensitive_local(
            entity_type=payload.entity_type,
            value=payload.value,
            confidence=payload.confidence,
            context=payload.context,
            model=payload.model,
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"sensitive": sensitive}


@app.post("/call_tool")
def call_tool(request: ToolRequest) -> dict[str, Any]:
    if request.tool_name == "mask_pdf":
        payload = MaskPayload(**request.arguments)
        return mask_pdf_tool(payload)
    if request.tool_name in {"openai_sensitive_review", "llm_sensitive_review"}:
        payload = LLMReviewPayload(**request.arguments)
        return openai_sensitive_review_tool(payload)

    raise HTTPException(status_code=404, detail=f"Unknown MCP tool: {request.tool_name}")


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("MCP_SERVER_PORT", "3003"))
    uvicorn.run(app, host="0.0.0.0", port=port)
