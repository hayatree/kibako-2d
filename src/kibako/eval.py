"""Render the fixed evaluation prompts and build a contact sheet.

Usage:
    python -m kibako.eval models/items-klein4b/eval_prompts.yaml OUT_DIR \
        [--model black-forest-labs/FLUX.2-klein-4B] [--lora PATH] [--steps 4] [--guidance 1.0]

Writes one PNG per (template, item, seed) and contact_<template>.png sheets.
Requires the `gen` dependency group (torch, diffusers, pyyaml, pillow).
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import yaml
from PIL import Image, ImageDraw


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def contact_sheet(paths: list[Path], labels: list[str], cols: int, cell: int = 256) -> Image.Image:
    rows = (len(paths) + cols - 1) // cols
    label_h = 18
    sheet = Image.new("RGB", (cols * cell, rows * (cell + label_h)), "white")
    draw = ImageDraw.Draw(sheet)
    for i, (p, label) in enumerate(zip(paths, labels)):
        x, y = (i % cols) * cell, (i // cols) * (cell + label_h)
        sheet.paste(Image.open(p).convert("RGB").resize((cell, cell)), (x, y))
        draw.text((x + 4, y + cell + 2), label[:40], fill="black")
    return sheet


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("prompts", type=Path)
    ap.add_argument("out", type=Path)
    ap.add_argument("--model", default="black-forest-labs/FLUX.2-klein-4B")
    ap.add_argument("--lora", type=Path)
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--guidance", type=float, default=1.0)
    args = ap.parse_args()

    import torch
    from diffusers import DiffusionPipeline

    cfg = yaml.safe_load(args.prompts.read_text())
    pipe = DiffusionPipeline.from_pretrained(args.model, torch_dtype=torch.bfloat16).to("cuda")
    if args.lora:
        pipe.load_lora_weights(str(args.lora))

    args.out.mkdir(parents=True, exist_ok=True)
    for tname, template in cfg["templates"].items():
        paths, labels = [], []
        for item in cfg["items"]:
            for seed in cfg["seeds"]:
                path = args.out / f"{tname}__{slug(item)}__s{seed}.png"
                if not path.exists():
                    image = pipe(
                        prompt=template.format(item=item),
                        width=cfg["width"],
                        height=cfg["height"],
                        num_inference_steps=args.steps,
                        guidance_scale=args.guidance,
                        generator=torch.Generator("cuda").manual_seed(seed),
                    ).images[0]
                    image.save(path)
                paths.append(path)
                labels.append(f"{item} s{seed}")
        cols = len(cfg["seeds"]) * 4
        contact_sheet(paths, labels, cols).save(args.out / f"contact_{tname}.png")
        print(f"wrote {args.out / f'contact_{tname}.png'}")


if __name__ == "__main__":
    main()
