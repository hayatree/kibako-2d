"""Generate training candidates from procedural sketches (see kibako.sketch).

Usage:
    python -m kibako.synth_sketch models/<model>/items_bows.yaml data/<model>/raw-bows

Same output layout as kibako.synth, plus `variant` (the file stem, unique per sketch and
seed) and `condition` (the sketch parameters) on every record. Requires the `gen` group.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path

import yaml

from kibako.sketch import render
from kibako.synth import revision_of, slug


def jobs(cfg: dict):
    sk = cfg["sketch"]
    for caption, shape in cfg["items"].items():
        for tilt, flip, seed in itertools.product(sk["tilts"], sk["flips"], sk["seeds"]):
            yield caption, {"sketch": sk["function"], "shape": shape, "tilt": tilt, "flip": flip}, seed


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("items", type=Path)
    ap.add_argument("out", type=Path)
    args = ap.parse_args()

    import torch
    from diffusers import Flux2KleinPipeline

    cfg = yaml.safe_load(args.items.read_text())
    model = cfg["generator"]
    revision = revision_of(model)
    pipe = Flux2KleinPipeline.from_pretrained(model, revision=revision, torch_dtype=torch.bfloat16).to("cuda")

    args.out.mkdir(parents=True, exist_ok=True)
    log = args.out / "candidates.jsonl"
    done = {json.loads(line)["path"] for line in log.read_text().splitlines() if line} if log.exists() else set()
    with open(log, "a", encoding="utf-8") as f:
        for caption, cond, seed in jobs(cfg):
            variant = f"{slug(caption)}__{cond['shape']}_{cond['tilt']}_{int(cond['flip'])}__s{seed}"
            name = variant + ".png"
            if name in done and (args.out / name).exists():
                continue
            prompt = cfg["template"].format(item=caption)
            image = pipe(
                image=render(cond).convert("RGB"),
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
                "variant": variant,
                "category": cfg["category"],
                "item": caption,
                "generator": model,
                "generator_revision": revision,
                "prompt": prompt,
                "seed": seed,
                "condition": cond,
            }
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
    print("done")


if __name__ == "__main__":
    main()
