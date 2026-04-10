"""
Apply masking to a PDF by drawing black redaction rectangles over detected entity spans.
Uses PyMuPDF for precise coordinate-based redaction.
"""
import os

import fitz


def mask_pdf(input_pdf_path: str, output_pdf_path: str, entities: list[dict]) -> None:
    """
    For each detected entity, find its text location in the page and redact it.
    Falls back to text-search based redaction if coordinates are unavailable.
    """
    if _try_mcp_masking(input_pdf_path, output_pdf_path, entities):
        print("[MASK] MCP masking complete")
        return

    print("[MASK] Starting PDF masking")
    _mask_pdf_local(input_pdf_path, output_pdf_path, entities)
    print("[MASK] Masking complete")


def _mask_pdf_local(input_pdf_path: str, output_pdf_path: str, entities: list[dict]) -> None:
    doc = fitz.open(input_pdf_path)

    entities_by_page = {}
    for ent in entities:
        p = ent["page"]
        entities_by_page.setdefault(p, []).append(ent)

    for page_num, page_entities in entities_by_page.items():
        if page_num >= len(doc):
            continue
        page = doc[page_num]

        for ent in page_entities:
            value = ent["value"]
            instances = page.search_for(value)
            if not instances:
                continue
            for rect in instances:
                page.add_redact_annot(rect, fill=(0, 0, 0))

        page.apply_redactions()

    doc.save(output_pdf_path, garbage=4, deflate=True)
    doc.close()


def _try_mcp_masking(input_pdf_path: str, output_pdf_path: str, entities: list[dict]) -> bool:
    """
    Optionally call an external MCP masking tool.
    Enabled via MASK_WITH_MCP=true. Falls back to local masking on any failure.
    """
    if os.getenv("MASK_WITH_MCP", "false").lower() != "true":
        return False

    server_url = os.getenv("MCP_SERVER_URL")
    tool_name = os.getenv("MCP_MASK_TOOL", "mask_pdf")
    if not server_url:
        print("[MASK] MCP enabled but MCP_SERVER_URL is missing")
        return False

    payload = {
        "input_pdf_path": input_pdf_path,
        "output_pdf_path": output_pdf_path,
        "entities": entities,
    }

    try:
        import fast_mcp  # type: ignore
    except Exception:
        fast_mcp = None

    try:
        import mcp_use  # type: ignore
    except Exception:
        mcp_use = None

    try:
        if fast_mcp is not None and hasattr(fast_mcp, "Client"):
            client = fast_mcp.Client(server_url)
            if hasattr(client, "call_tool"):
                client.call_tool(tool_name, payload)
                return True
        if mcp_use is not None and hasattr(mcp_use, "call_tool"):
            mcp_use.call_tool(server_url=server_url, tool_name=tool_name, arguments=payload)
            return True
    except Exception as e:
        print(f"[MASK] MCP masking failed: {e}")
        return False

    print("[MASK] MCP packages found, but no compatible call interface")
    return False
