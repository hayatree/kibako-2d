"""Procedural sketches used to pin the shape of an item.

FLUX.2 [klein] cannot draw some items from a text prompt alone (a bow comes out as a stick or
as a symmetric diamond with an arrow through it). A flat drawing of the right shape, given to
the model as the input image, fixes the shape and leaves the rendering style to the model.

The sketches are original work of the maintainers (CC0). The parameters that make one are
recorded in the manifest as `condition`.
"""

from __future__ import annotations

import math
from PIL import Image, ImageDraw

def bow(shape: str = "simple", tilt: float = 0, size: int = 1024, ss: int = 3, flip: bool = False) -> Image.Image:
    """A strung bow, limbs bulging to the right, string on the left. `tilt` in degrees; `flip` mirrors it."""
    W = size * ss
    span = math.radians({"simple": 50, "long": 36, "recurve": 48}[shape])
    R = 1.0
    n = 120
    raw = []
    for i in range(n + 1):
        a = -span + 2 * span * i / n
        r = R
        if shape == "recurve" and abs(a) > span * 0.72:          # tips flick forward
            k = (abs(a) - span * 0.72) / (span * 0.28)
            r = R - 0.16 * k * k
        raw.append((r * math.cos(a), r * math.sin(a)))
    # rotate (tilt), then fit the bow plus its string into 72% of the canvas, centred
    th = math.radians(tilt)
    rot = lambda p: (p[0] * math.cos(th) - p[1] * math.sin(th), p[0] * math.sin(th) + p[1] * math.cos(th))
    raw = [rot(p) for p in raw]
    xs = [p[0] for p in raw]; ys = [p[1] for p in raw]
    k = 0.72 * W / max(max(xs) - min(xs), max(ys) - min(ys))
    ox = W / 2 - k * (max(xs) + min(xs)) / 2; oy = W / 2 - k * (max(ys) + min(ys)) / 2
    pts = [(ox + k * x, oy + k * y) for x, y in raw]
    im = Image.new("RGB", (W, W), "white"); d = ImageDraw.Draw(im)
    def thick(i):                                                # fat in the middle, thin at the tips
        t = i / n; return W * (0.014 + 0.022 * math.sin(math.pi * t) ** 0.7)
    for i in range(n):
        d.line([pts[i], pts[i + 1]], fill=(120, 78, 38), width=int(thick(i)))
        d.ellipse([pts[i][0] - thick(i) / 2, pts[i][1] - thick(i) / 2, pts[i][0] + thick(i) / 2, pts[i][1] + thick(i) / 2], fill=(120, 78, 38))
    d.line([pts[0], pts[-1]], fill=(90, 80, 70), width=int(W * 0.004))   # string
    mx, my = pts[n // 2]                                                 # grip wrap
    g = W * 0.045
    gx, gy = -math.sin(th) * g, math.cos(th) * g
    d.line([(mx - gx, my - gy), (mx + gx, my + gy)], fill=(60, 38, 20), width=int(W * 0.03))
    im = im.resize((size, size), Image.LANCZOS)
    return im.transpose(Image.FLIP_LEFT_RIGHT) if flip else im


def render(condition: dict):
    """Rebuild the sketch described by a manifest `condition`."""
    if condition["sketch"] != "bow":
        raise ValueError(f"unknown sketch {condition['sketch']!r}")
    return bow(condition["shape"], condition["tilt"], flip=condition["flip"])
