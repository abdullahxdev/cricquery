"""Optional helper — bulk-downloads demo images from a list of URLs.

Use this if you've right-click-copied image URLs from Google Images (or
anywhere) and want to dump them into the right examples/ subfolder without
clicking 18 times.

Fill in the IMAGES dict with (url, target_filename), then:

    python scripts/download_demo_images.py

Images are saved into ../examples/<category>/<filename>.
Auto-resizes to max-side 1024 px to keep the demo folder small.
"""
from __future__ import annotations

import sys
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import urlopen, Request

try:
    from PIL import Image
except ImportError:
    print("Pillow not installed. Run: pip install Pillow")
    sys.exit(1)

ROOT = Path(__file__).resolve().parent.parent
EXAMPLES = ROOT / "examples"
MAX_SIDE = 1024

# Edit this dict with your URLs. The key is the subfolder name; the value is a
# list of (url, output_filename) tuples.
IMAGES: dict[str, list[tuple[str, str]]] = {
    "batsman": [
        # ("https://example.com/batsman1.jpg", "batsman_1.jpg"),
    ],
    "bowler": [
        # ("https://example.com/bowler1.jpg", "bowler_1.jpg"),
    ],
    "lefty": [],
    "righty": [],
    "front_foot": [],
    "back_foot": [],
    "freeform": [],
}


def download_one(url: str, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    req = Request(url, headers={"User-Agent": "Mozilla/5.0 (download_demo_images)"})
    with urlopen(req, timeout=20) as r:
        data = r.read()
    tmp = out_path.with_suffix(out_path.suffix + ".tmp")
    tmp.write_bytes(data)
    # resize if larger than MAX_SIDE on the long edge
    img = Image.open(tmp).convert("RGB")
    if max(img.size) > MAX_SIDE:
        img.thumbnail((MAX_SIDE, MAX_SIDE))
    img.save(out_path, "JPEG", quality=90)
    tmp.unlink(missing_ok=True)
    print(f"  ✓ {out_path.relative_to(ROOT)}  ({img.size[0]}x{img.size[1]})")


def main() -> int:
    total, ok, bad = 0, 0, 0
    for category, items in IMAGES.items():
        if not items:
            continue
        print(f"\n[{category}]")
        for url, filename in items:
            total += 1
            target = EXAMPLES / category / filename
            try:
                download_one(url, target)
                ok += 1
            except Exception as e:
                bad += 1
                print(f"  ✗ {filename}  ({type(e).__name__}: {e})")
    if total == 0:
        print("No URLs in IMAGES dict. Edit the script and add them.")
        return 1
    print(f"\nDone. {ok}/{total} downloaded ({bad} failed).")
    return 0 if bad == 0 else 2


if __name__ == "__main__":
    sys.exit(main())
