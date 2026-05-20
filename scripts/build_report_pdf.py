"""Convert report/CricQuery_Final_Report.md -> styled HTML -> PDF (via Chrome headless).

No external installs needed beyond `markdown` (pip) and Google Chrome (you have it).

Run:
    python scripts/build_report_pdf.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

# Lazy-install markdown if missing
try:
    import markdown  # noqa: F401
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "markdown"])
    import markdown  # noqa: F401

ROOT = Path(__file__).resolve().parent.parent
REPORT_MD = ROOT / "report" / "CricQuery_Final_Report.md"
REPORT_HTML = ROOT / "report" / "CricQuery_Final_Report.html"
REPORT_PDF = ROOT / "report" / "CricQuery_Final_Report.pdf"

CHROME_PATHS = [
    "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome",
    "/Applications/Chromium.app/Contents/MacOS/Chromium",
    "/Applications/Brave Browser.app/Contents/MacOS/Brave Browser",
    "/Applications/Microsoft Edge.app/Contents/MacOS/Microsoft Edge",
]


CSS = """
@page {
    size: A4;
    margin: 18mm 16mm 18mm 16mm;
}
* { box-sizing: border-box; }
body {
    font-family: 'Helvetica Neue', 'Arial', sans-serif;
    color: #1a1a1a;
    line-height: 1.55;
    font-size: 11pt;
    max-width: 100%;
}
h1 {
    font-size: 22pt;
    color: #0b3d91;
    margin: 0 0 0.4em 0;
}
h2 {
    font-size: 15pt;
    color: #0b3d91;
    border-bottom: 1px solid #d0d7e2;
    padding-bottom: 4px;
    margin-top: 1.4em;
    margin-bottom: 0.6em;
}
h3 {
    font-size: 12.5pt;
    color: #1a1a1a;
    margin-top: 1.1em;
}
p { margin: 0.4em 0 0.7em 0; text-align: justify; }
strong { color: #0b3d91; }
em { color: #444; }
table {
    border-collapse: collapse;
    width: 100%;
    margin: 0.6em 0 1em 0;
    font-size: 10.5pt;
}
th, td {
    border: 1px solid #c5cdd9;
    padding: 6px 9px;
    text-align: left;
    vertical-align: top;
}
th {
    background: #eef2f8;
    color: #0b3d91;
    font-weight: 600;
}
tr:nth-child(even) td { background: #fafbfd; }
code {
    background: #f3f5f9;
    padding: 1px 5px;
    border-radius: 3px;
    font-size: 10pt;
    font-family: Menlo, Consolas, monospace;
}
pre {
    background: #f3f5f9;
    padding: 10px 12px;
    border-radius: 5px;
    overflow-x: auto;
    font-size: 9.5pt;
    line-height: 1.35;
}
pre code { background: transparent; padding: 0; }
img { max-width: 100%; height: auto; display: block; margin: 0.4em auto; }
hr { border: none; border-top: 1px solid #d0d7e2; margin: 1.2em 0; }
.cover {
    text-align: center;
    padding-top: 30mm;
}
ul, ol { margin: 0.3em 0 0.7em 1.4em; padding: 0; }
li { margin: 0.15em 0; }
.page-break { page-break-after: always; height: 0; }
"""


def find_chrome() -> str:
    for p in CHROME_PATHS:
        if Path(p).exists():
            return p
    # PATH fallback
    for name in ["google-chrome", "chromium", "chrome"]:
        if shutil.which(name):
            return shutil.which(name)
    raise RuntimeError("Could not find Chrome/Chromium/Edge. Install Google Chrome from https://www.google.com/chrome/")


def md_to_html(md_text: str) -> str:
    """Convert Markdown to HTML, preserving inline HTML (for page-break divs and img tags)."""
    html_body = markdown.markdown(
        md_text,
        extensions=["extra", "tables", "sane_lists", "toc"],
    )
    # Replace the manual page-break divs with our class
    html_body = html_body.replace(
        '<div style="page-break-after: always;"></div>',
        '<div class="page-break"></div>',
    )
    return (
        f"<!DOCTYPE html>\n<html><head><meta charset='utf-8'>"
        f"<title>CricQuery — Final Report</title>"
        f"<style>{CSS}</style></head><body>{html_body}</body></html>"
    )


def main() -> int:
    if not REPORT_MD.exists():
        print(f"✗ {REPORT_MD} not found.")
        return 1

    print(f"→ Reading {REPORT_MD.relative_to(ROOT)}")
    md_text = REPORT_MD.read_text(encoding="utf-8")

    print(f"→ Converting to styled HTML")
    REPORT_HTML.write_text(md_to_html(md_text), encoding="utf-8")
    print(f"   wrote {REPORT_HTML.relative_to(ROOT)}")

    chrome = find_chrome()
    print(f"→ Using {chrome}")
    cmd = [
        chrome,
        "--headless=new",
        "--disable-gpu",
        "--no-pdf-header-footer",
        f"--print-to-pdf={REPORT_PDF}",
        REPORT_HTML.as_uri(),
    ]
    print(f"→ Rendering PDF (this takes ~5 seconds)…")
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0 or not REPORT_PDF.exists():
        # Fallback: try legacy --headless without =new
        cmd[1] = "--headless"
        result = subprocess.run(cmd, capture_output=True, text=True)

    if not REPORT_PDF.exists():
        print("✗ Chrome did not produce a PDF.")
        print("  stdout:", result.stdout[:500])
        print("  stderr:", result.stderr[:500])
        return 2

    size_kb = REPORT_PDF.stat().st_size / 1024
    print(f"✓ {REPORT_PDF.relative_to(ROOT)}  ({size_kb:.0f} KB)")
    print("  Open it with:  open " + str(REPORT_PDF))
    return 0


if __name__ == "__main__":
    sys.exit(main())
