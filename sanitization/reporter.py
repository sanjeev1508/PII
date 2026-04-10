"""
Generate a human-readable PDF report of all detected and masked entities.
"""
import datetime
from collections import Counter

from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle


def generate_report(
    output_path: str,
    source_file: str,
    total_pages: int,
    extraction_mode_summary: dict,
    entities: list[dict],
    processing_seconds: float,
) -> None:
    print("[REPORT] Generating report PDF")
    doc = SimpleDocTemplate(output_path, pagesize=A4)
    styles = getSampleStyleSheet()
    story = []

    title_style = ParagraphStyle("Title", parent=styles["Title"], fontSize=18, spaceAfter=12)
    story.append(Paragraph("Document Sanitization Report", title_style))
    story.append(Spacer(1, 0.3 * cm))

    meta = [
        f"<b>Source File:</b> {source_file}",
        f"<b>Date:</b> {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"<b>Total Pages:</b> {total_pages}",
        f"<b>Processing Time:</b> {processing_seconds:.2f}s",
        (
            f"<b>Native Pages:</b> {extraction_mode_summary.get('native', 0)} | "
            f"<b>OCR Pages:</b> {extraction_mode_summary.get('ocr', 0)}"
        ),
    ]
    for line in meta:
        story.append(Paragraph(line, styles["Normal"]))
    story.append(Spacer(1, 0.5 * cm))

    type_counts = Counter(e["type"] for e in entities)
    story.append(Paragraph("<b>Entity Counts by Type:</b>", styles["Heading2"]))
    count_data = [["Entity Type", "Count"]] + [[k, str(v)] for k, v in sorted(type_counts.items())]
    count_table = Table(count_data, colWidths=[10 * cm, 5 * cm])
    count_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("GRID", (0, 0), (-1, -1), 0.5, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
            ]
        )
    )
    story.append(count_table)
    story.append(Spacer(1, 0.5 * cm))

    story.append(Paragraph("<b>Detected Entities (Detail):</b>", styles["Heading2"]))
    detail_data = [["#", "Page", "Type", "Original Value", "Masked As", "Confidence"]]
    for i, ent in enumerate(entities, 1):
        detail_data.append(
            [
                str(i),
                str(ent["page"] + 1),
                ent["type"],
                ent["value"][:60] + ("..." if len(ent["value"]) > 60 else ""),
                ent["mask"],
                str(ent.get("confidence", "N/A")),
            ]
        )

    detail_table = Table(detail_data, colWidths=[1 * cm, 1.5 * cm, 3.5 * cm, 5 * cm, 3 * cm, 2 * cm])
    detail_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.darkblue),
                ("TEXTCOLOR", (0, 0), (-1, 0), colors.white),
                ("FONTSIZE", (0, 0), (-1, -1), 7),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("ROWBACKGROUNDS", (0, 1), (-1, -1), [colors.whitesmoke, colors.white]),
                ("WORDWRAP", (0, 0), (-1, -1), True),
            ]
        )
    )
    story.append(detail_table)
    doc.build(story)
    print("[REPORT] Report generation complete")
