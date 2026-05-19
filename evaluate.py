"""Run the full CricQuery pipeline against test_vqa_pairs.json.

Prints:
  - per-question-type accuracy
  - overall accuracy
  - which models fired (router_via, model via)
  - a confusion matrix for Q1 (shot type) if any shot pairs are present
"""
from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path

from PIL import Image

from src.pipeline import CricQueryPipeline


def normalize(s: str) -> str:
    return s.strip().lower().replace("-", " ").replace("_", " ")


def main(test_json: Path, project_root: Path):
    data = json.loads(test_json.read_text())
    pairs = data["pairs"]
    pipe = CricQueryPipeline()

    per_type = defaultdict(lambda: {"correct": 0, "total": 0})
    shot_pairs = []
    rows = []
    skipped = 0

    for pair in pairs:
        img_path = project_root / pair["image"]
        if "REPLACE" in str(img_path) or not img_path.exists():
            skipped += 1
            continue
        img = Image.open(img_path).convert("RGB")
        res = pipe.answer(img, pair["question"])
        pred = normalize(res.answer)
        gold = normalize(pair["answer"])
        correct = (pred == gold) or (gold in pred) or (pred in gold)
        per_type[pair["qtype"]]["total"] += 1
        per_type[pair["qtype"]]["correct"] += int(correct)
        if pair["qtype"] == "shot":
            shot_pairs.append((gold, pred))
        rows.append((pair["qtype"], pair["question"][:40], gold, pred, res.via, correct))

    # ---- print summary ----
    print()
    print(f"{'qtype':<14} {'question':<42} {'gold':<14} {'pred':<14} {'via':<12} ok")
    print("-" * 100)
    for r in rows:
        print(f"{r[0]:<14} {r[1]:<42} {r[2]:<14} {r[3]:<14} {r[4]:<12} {'✓' if r[5] else '✗'}")

    print()
    print("=" * 60)
    print(f"{'Question type':<14} {'Correct':>8} {'Total':>8} {'Accuracy':>12}")
    print("-" * 60)
    total_c = total_t = 0
    for qt, s in per_type.items():
        acc = s["correct"] / s["total"] if s["total"] else 0.0
        total_c += s["correct"]; total_t += s["total"]
        print(f"{qt:<14} {s['correct']:>8} {s['total']:>8} {acc:>11.1%}")
    print("-" * 60)
    overall = total_c / total_t if total_t else 0.0
    print(f"{'OVERALL':<14} {total_c:>8} {total_t:>8} {overall:>11.1%}")
    print(f"\nSkipped (REPLACE/missing): {skipped}")

    if shot_pairs:
        print("\nShot confusion (gold -> pred -> count):")
        cm = defaultdict(int)
        for g, p in shot_pairs:
            cm[(g, p)] += 1
        for (g, p), c in sorted(cm.items()):
            print(f"  {g} -> {p}: {c}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--test", default="data/test_vqa_pairs.json")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent
    main(root / args.test, root)
