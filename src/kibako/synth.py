"""Generate the synthetic training candidates for a model.

Usage:
    python -m kibako.synth models/<model>/items.yaml data/<model>/raw

For every item and seed in items.yaml it writes <slug>__s<seed>.png and appends one
record to candidates.jsonl: generator, weights revision, prompt, seed, category,
item. Already generated files are skipped, so the run can be resumed.

Requires the `gen` dependency group.
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import yaml


def slug(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", text.lower()).strip("-")


def revision_of(model: str) -> str:
    """Commit hash of the locally cached weights (the snapshot directory name)."""
    from huggingface_hub import snapshot_download

    return Path(snapshot_download(model, local_files_only=True)).name


def jobs(cfg: dict):
    """Yield (category, caption, subject, seed).

    An entry is either a string (the caption and the subject in the prompt are the same)
    or a one-key mapping `caption: subject`, for items the base model only draws well
    when the shape is spelled out. The training caption stays the short one.
    """
    for category, items in cfg["categories"].items():
        for item in items:
            caption, subject = next(iter(item.items())) if isinstance(item, dict) else (item, item)
            for seed in cfg["seeds"]:
                yield category, caption, subject, seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    import torch
    from diffusers import DiffusionPipeline

    cfg = yaml.safe_load(args.items.read_text())
    model = cfg["generator"]
    revision = revision_of(model)
    pipe = DiffusionPipeline.from_pretrained(model, revision=revision, torch_dtype=torch.bfloat16).to("cuda")

    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "candidates.jsonl"
    done = set()
    if log.exists():
        done = {json.loads(line)["path"] for line in log.read_text().splitlines() if line}

    with open(log, "a", encoding="utf-8") as f:
        for category, item, subject, seed in jobs(cfg):
            name = f"{slug(item)}__s{seed}.png"
            if name in done and (args.out / name).exists():
                continue
            prompt = cfg["template"].format(item=subject)
            image = pipe(
                prompt=prompt,
                width=cfg["width"],
                height=cfg["height"],
                num_inference_steps=cfg["steps"],
                guidance_scale=cfg["guidance"],
                generator=torch.Generator("cuda").manual_seed(seed),
            ).images[0]
            image.save(args.out / name)
            rec = {
                "path": name,
                "category": category,
                "item": item,
                "generator": model,
                "generator_revision": revision,
                "prompt": prompt,
                "seed": seed,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    print("done")


if __name__ == "__main__":
    main()
