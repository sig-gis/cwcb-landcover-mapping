#!/usr/bin/env python3
"""Render the authoritative technical handoff Markdown as standalone HTML."""
from pathlib import Path
import markdown

ROOT = Path(__file__).resolve().parents[1]


def main():
    source = ROOT / "reports" / "TECHNICAL_METHODS_AND_FIVE_YEAR_HANDOFF.md"
    body = markdown.markdown(
        source.read_text(encoding="utf-8"),
        extensions=["tables", "fenced_code", "toc"],
    )
    style = """body{font-family:system-ui,-apple-system,Segoe UI,sans-serif;max-width:1050px;margin:auto;padding:36px;color:#222;line-height:1.55}h1{font-size:2.2rem}h2{border-top:1px solid #ccc;padding-top:24px;margin-top:42px}h3{margin-top:28px}table{border-collapse:collapse;width:100%;font-size:.94rem}th,td{border-bottom:1px solid #ddd;padding:8px;text-align:right}th:first-child,td:first-child{text-align:left}code{background:#f3f3f3;padding:2px 4px}pre{background:#f5f5f5;padding:14px;overflow:auto}blockquote{border-left:4px solid #777;margin-left:0;padding-left:15px;color:#444}li{margin:4px 0}"""
    document = f"<!doctype html><html><head><meta charset='utf-8'><meta name='viewport' content='width=device-width,initial-scale=1'><title>Technical methods and five-year handoff</title><style>{style}</style></head><body>{body}</body></html>"
    target = ROOT / "reports" / "TECHNICAL_METHODS_AND_FIVE_YEAR_HANDOFF.html"
    target.write_text(document, encoding="utf-8")
    print(target)


if __name__ == "__main__":
    main()
