import numpy as np
import pytest

from kibako.sketch import bow, render


@pytest.mark.parametrize("shape", ["simple", "long", "recurve"])
@pytest.mark.parametrize("tilt", [-50, 0])
def test_bow_sketch_is_centered_with_margin(shape, tilt):
    a = np.asarray(bow(shape, tilt, size=256, ss=2).convert("L"))
    ys, xs = np.where(a < 235)
    assert xs.min() > 256 * 0.05 and xs.max() < 256 * 0.95
    assert ys.min() > 256 * 0.05 and ys.max() < 256 * 0.95


def test_flip_mirrors_and_render_reproduces_the_condition():
    cond = {"sketch": "bow", "shape": "recurve", "tilt": -35, "flip": True}
    flipped = np.asarray(render(cond))
    plain = np.asarray(bow("recurve", -35, flip=False))
    assert np.array_equal(flipped, plain[:, ::-1])


def test_render_rejects_unknown_sketch():
    with pytest.raises(ValueError):
        render({"sketch": "dragon"})
