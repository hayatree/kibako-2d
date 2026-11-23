"""Curate generated candidates into a training dataset.

Usage:
    python -m kibako.curate check  RAW_DIR
    python -m kibako.curate sheets RAW_DIR SHEETS_DIR [--per-sheet 20]
    python -m kibako.curate build  RAW_DIR SELECTION.yaml DATASET_DIR

check   writes checked.jsonl next to candidates.jsonl, adding `auto_ok` and `reasons`.
        Images that fail are never offered for selection.
sheets  writes one contact sheet per group of items, four seeds per row.
build   copies the selected images into DATASET_DIR (which must be new or empty) together with <name>.txt captions
        (the item name alone) and a manifest.jsonl that passes `kibako.manifest check`.

SELECTION.yaml maps an item name to the list of seeds to keep:
    an iron sword: [0, 2]
Candidates made from several sketches (kibako.synth_sketch) share a seed, so they are chosen by
their `variant` id instead:
    a wooden bow: [simple_-35_0__s0, recurve_0_1__s1]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageDraw

from kibako.manifest import SYNTHETIC_LICENSE, validate

WHITE = 235  # a channel value at or above this counts as background
MIN_MARGIN = 0.03  # subject must stay this far from every edge (fraction of image size)
MIN_EXTENT = 0.25  # the subject's longer side (bounding box) as a fraction of the image; thin items such as bows have little area
MAX_AREA = 0.75  # subject area as a fraction of the image


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def write_jsonl(path: Path, records: list[dict]) -> None:
    path.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in records), encoding="utf-8")


def inspect(path: Path) -> list[str]:
    """Return the reasons an image is unusable as an icon; empty means it passes."""
    a = np.asarray(Image.open(path).convert("RGB"))
    h, w, _ = a.shape
    fg = ~(a >= WHITE).all(axis=2)  # not-white pixels
    border = np.concatenate([fg[0], fg[-1], fg[:, 0], fg[:, -1]])
    reasons = []
    if border.mean() > 0.02:
        reasons.append("background is not plain white")
    if not fg.any():
        return reasons + ["empty"]
    ys, xs = np.where(fg)
    if min(xs.min() / w, ys.min() / h, 1 - xs.max() / w, 1 - ys.max() / h) < MIN_MARGIN:
        reasons.append("subject touches the edge")
    if max((xs.max() - xs.min()) / w, (ys.max() - ys.min()) / h) < MIN_EXTENT:
        reasons.append("subject too small")
    if fg.mean() > MAX_AREA:
        reasons.append("subject too large")
    return reasons


def cmd_check(raw: Path) -> None:
    recs = read_jsonl(raw / "candidates.jsonl")
    for r in recs:
        r["reasons"] = inspect(raw / r["path"])
        r["auto_ok"] = not r["reasons"]
    write_jsonl(raw / "checked.jsonl", recs)
    ok = sum(r["auto_ok"] for r in recs)
    print(f"{ok}/{len(recs)} candidates pass the automatic check")


def cmd_sheets(raw: Path, out: Path, per_sheet: int, cell: int = 200) -> None:
    recs = read_jsonl(raw / "checked.jsonl")
    items: dict[str, list[dict]] = {}
    for r in recs:
        items.setdefault(r["item"], []).append(r)
    names = list(items)
    out.mkdir(parents=True, exist_ok=True)
    label_h = 16
    for n, start in enumerate(range(0, len(names), per_sheet), start=1):
        group = names[start : start + per_sheet]
        seeds = max(len(items[g]) for g in group)
        sheet = Image.new("RGB", (seeds * cell + 230, len(group) * (cell + label_h)), "white")
        d = ImageDraw.Draw(sheet)
        for row, name in enumerate(group):
            y = row * (cell + label_h)
            d.text((4, y + cell // 2), name[:34], fill="black")
            for col, r in enumerate(sorted(items[name], key=lambda r: (r["seed"], r.get("variant", "")))):
                x = 230 + col * cell
                im = Image.open(raw / r["path"]).convert("RGB").resize((cell, cell))
                sheet.paste(im, (x, y))
                d.text((x + 4, y + cell), (r["variant"].split("__", 1)[-1] if "variant" in r else f"s{r['seed']}") + ("" if r["auto_ok"] else " X " + r["reasons"][0][:22]),
                       fill="black" if r["auto_ok"] else "red")
        sheet.save(out / f"sheet_{n:02d}.png")
    print(f"wrote {n} sheets to {out}")


def cmd_build(raw: Path, selection: Path, dataset: Path) -> None:
    chosen = yaml.safe_load(selection.read_text()) or {}
    checked = {(r["item"], r.get("variant") or r["seed"]): r for r in read_jsonl(raw / "checked.jsonl")}
    if dataset.exists() and any(dataset.iterdir()):
        # Old files would stay in the training folder without a manifest entry.
        sys.exit(f"{dataset} is not empty; build into a new folder")
    dataset.mkdir(parents=True, exist_ok=True)
    manifest = []
    for item, seeds in chosen.items():
        for seed in seeds:
            r = checked.get((item, seed))
            if r is None:
                sys.exit(f"no candidate for {item!r} {seed!r}")
            if not r["auto_ok"]:
                sys.exit(f"{item!r} seed {seed} failed the automatic check: {r['reasons']}")
            src = raw / r["path"]
            shutil.copy2(src, dataset / r["path"])
            (dataset / r["path"]).with_suffix(".txt").write_text(item + "\n", encoding="utf-8")
            manifest.append(
                {
                    "sha256": hashlib.sha256(src.read_bytes()).hexdigest(),
                    "path": r["path"],
                    "source": "synthetic",
                    "source_url": f"https://huggingface.co/{r['generator']}",
                    "license": SYNTHETIC_LICENSE,
                    "generator": r["generator"],
                    "generator_revision": r["generator_revision"],
                    "prompt": r["prompt"],
                    "seed": r["seed"],
                    "item": item,
                    "category": r["category"],
                    **({"condition": r["condition"]} if "condition" in r else {}),
                }
            )
    errors = validate(manifest)
    if errors:
        sys.exit("\n".join(errors))
    write_jsonl(dataset / "manifest.jsonl", manifest)
    print(f"built {len(manifest)} images in {dataset}")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    c = sub.add_parser("check")
    c.add_argument("raw", type=Path)
    s = sub.add_parser("sheets")
    s.add_argument("raw", type=Path)
    s.add_argument("out", type=Path)
    s.add_argument("--per-sheet", type=int, default=20)
    b = sub.add_parser("build")
    b.add_argument("raw", type=Path)
    b.add_argument("selection", type=Path)
    b.add_argument("dataset", type=Path)
    a = ap.parse_args()
    if a.cmd == "check":
        cmd_check(a.raw)
    elif a.cmd == "sheets":
        cmd_sheets(a.raw, a.out, a.per_sheet)
    else:
        cmd_build(a.raw, a.selection, a.dataset)


if __name__ == "__main__":
    main()
