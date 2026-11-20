import json

import pytest
from PIL import Image, ImageDraw

from kibako import curate
from kibako.manifest import validate


def icon(tmp_path, name="a.png", box=(300, 300, 700, 700), bg="white", shape="ellipse"):
    im = Image.new("RGB", (1000, 1000), bg)
    getattr(ImageDraw.Draw(im), shape)(box, fill=(200, 30, 30))
    p = tmp_path / name
    im.save(p)
    return p


def test_centered_icon_on_white_passes(tmp_path):
    assert curate.inspect(icon(tmp_path)) == []


def test_rejects_non_white_background(tmp_path):
    assert "background is not plain white" in curate.inspect(icon(tmp_path, bg=(40, 40, 40)))


def test_rejects_subject_touching_edge(tmp_path):
    assert "subject touches the edge" in curate.inspect(icon(tmp_path, box=(0, 300, 400, 700)))


def test_rejects_tiny_and_huge_subjects(tmp_path):
    assert "subject too small" in curate.inspect(icon(tmp_path, box=(480, 480, 520, 520)))
    assert "subject too large" in curate.inspect(icon(tmp_path, box=(40, 40, 960, 960), shape="rectangle"))


def test_build_produces_valid_manifest_and_captions(tmp_path):
    raw, ds = tmp_path / "raw", tmp_path / "ds"
    raw.mkdir()
    icon(raw, "an-iron-sword__s0.png")
    rec = {
        "path": "an-iron-sword__s0.png", "category": "melee_weapons", "item": "an iron sword",
        "generator": "black-forest-labs/FLUX.2-klein-4B", "generator_revision": "abc",
        "prompt": "p", "seed": 0, "reasons": [], "auto_ok": True,
    }
    (raw / "checked.jsonl").write_text(json.dumps(rec) + "\n")
    sel = tmp_path / "sel.yaml"
    sel.write_text("an iron sword: [0]\n")
    curate.cmd_build(raw, sel, ds)
    assert (ds / "an-iron-sword__s0.txt").read_text() == "an iron sword\n"
    assert validate(curate.read_jsonl(ds / "manifest.jsonl")) == []


def test_build_refuses_images_that_failed_the_check(tmp_path):
    raw = tmp_path / "raw"
    raw.mkdir()
    rec = {"path": "x.png", "item": "x", "seed": 0, "auto_ok": False, "reasons": ["empty"]}
    (raw / "checked.jsonl").write_text(json.dumps(rec) + "\n")
    sel = tmp_path / "sel.yaml"
    sel.write_text("x: [0]\n")
    with pytest.raises(SystemExit):
        curate.cmd_build(raw, sel, tmp_path / "ds")


def test_build_selects_sketch_candidates_by_variant_and_keeps_condition(tmp_path):
    raw, ds = tmp_path / "raw", tmp_path / "ds"
    raw.mkdir()
    recs = []
    for variant, tilt in (("a-wooden-bow__simple_0_0__s0", 0), ("a-wooden-bow__simple_-35_0__s0", -35)):
        icon(raw, variant + ".png")
        recs.append({
            "path": variant + ".png", "variant": variant, "category": "ranged_weapons", "item": "a wooden bow",
            "generator": "black-forest-labs/FLUX.2-klein-4B", "generator_revision": "abc", "prompt": "p", "seed": 0,
            "condition": {"sketch": "bow", "shape": "simple", "tilt": tilt, "flip": False},
            "reasons": [], "auto_ok": True,
        })
    (raw / "checked.jsonl").write_text("".join(json.dumps(r) + "\n" for r in recs))
    sel = tmp_path / "sel.yaml"
    sel.write_text("a wooden bow: [a-wooden-bow__simple_-35_0__s0]\n")  # same seed, two sketches
    curate.cmd_build(raw, sel, ds)
    manifest = curate.read_jsonl(ds / "manifest.jsonl")
    assert [m["path"] for m in manifest] == ["a-wooden-bow__simple_-35_0__s0.png"]
    assert manifest[0]["condition"]["tilt"] == -35
    assert (ds / "a-wooden-bow__simple_-35_0__s0.txt").read_text() == "a wooden bow\n"
    assert validate(manifest) == []


def test_thin_but_large_subject_is_not_too_small(tmp_path):
    # a long thin item (a bow, a spear) has little area but fills the frame
    assert curate.inspect(icon(tmp_path, box=(480, 100, 520, 900), shape="rectangle")) == []
