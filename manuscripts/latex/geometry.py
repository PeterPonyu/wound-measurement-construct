"""Measure appendix clearances in rendered PDFs, not TeX setting strings."""
from __future__ import annotations

import re

import fitz


CAPTION = re.compile(r"^Table\s+(S\d+)\s*[.(]")


def appendix_geometry(document):
    records = []
    for page in document:
        lines = []
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                text = "".join(span["text"] for span in line["spans"]).strip()
                lines.append((fitz.Rect(line["bbox"]), text))
        rules = sorted({round((item[1].y + item[2].y) / 2, 4)
                        for drawing in page.get_drawings()
                        for item in drawing["items"]
                        if item[0] == "l" and abs(item[1].y - item[2].y) < .2
                        and abs(item[1].x - item[2].x) > 100
                        and 40 < item[1].y < page.rect.height - 40})
        captions = [(box, match[1], "continued" in text)
                    for box, text in lines if (match := CAPTION.match(text))]
        for box, table, continued in captions:
            after = [y for y in rules if y > box.y1]
            if not after:
                raise RuntimeError(f"Page {page.number + 1}: {table} has no rendered top rule")
            top_rule = min(after)
            # Captions can wrap across several PDF lines/blocks. End at the
            # first table rule, never at the first line containing its number.
            caption_end = max(b.y1 for b, _ in lines
                              if b.y0 >= box.y0 - .1 and b.y1 < top_rule)
            before = [y for y in rules if y < box.y0]
            records.append({"page": page.number + 1, "table": table,
                "continued": continued,
                "caption_to_top_rule_pt": round(top_rule - caption_end, 3),
                "previous_rule_to_caption_pt": round(box.y0 - max(before), 3) if before else None})
    return records


def require_readable_appendix(records, *, caption_min=7, boundary_min=10):
    if not records:
        raise RuntimeError("No rendered appendix captions were measured")
    for row in records:
        if row["caption_to_top_rule_pt"] < caption_min:
            raise RuntimeError(f"Page {row['page']}, {row['table']}: caption-to-rule clearance "
                               f"{row['caption_to_top_rule_pt']} pt < {caption_min} pt")
        gap = row["previous_rule_to_caption_pt"]
        if gap is not None and gap < boundary_min:
            raise RuntimeError(f"Page {row['page']}, {row['table']}: inter-table clearance "
                               f"{gap} pt < {boundary_min} pt")


def inspect_appendix(document):
    records = appendix_geometry(document)
    require_readable_appendix(records)
    boundaries = [r["previous_rule_to_caption_pt"] for r in records
                  if r["previous_rule_to_caption_pt"] is not None]
    return {"caption_minimum_pt": min(r["caption_to_top_rule_pt"] for r in records),
            "boundary_minimum_pt": min(boundaries) if boundaries else None,
            "numbered_continuation_headings": sum(r["continued"] for r in records),
            "thresholds_pt": {"caption_to_rule": 7, "between_tables": 10},
            "measurements": records}
