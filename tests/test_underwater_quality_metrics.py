import numpy as np
import pytest

from src.evaluation.underwater_quality_metrics import uiqm, uciqe


def _inputs():
    h, w = 32, 40
    yy, xx = np.mgrid[0:h, 0:w]
    constant = np.full((h, w, 3), 0.4, dtype=np.float64)
    gray = np.repeat((xx / (w - 1))[..., None], 3, axis=2)
    gradient = np.stack((xx/(w-1), yy/(h-1), (xx+yy)/(w+h-2)), axis=-1)
    rng = np.random.default_rng(42)
    random_rgb = rng.random((h, w, 3))
    return constant, gray, gradient, random_rgb


@pytest.mark.parametrize("fn", [uiqm, uciqe])
def test_underwater_metrics_are_finite_and_deterministic(fn):
    for image in _inputs():
        first = fn(image)
        assert np.isfinite(first)
        assert first == fn(image)


def test_underwater_metrics_require_rgb_and_unit_range():
    for fn in (uiqm, uciqe):
        with pytest.raises(ValueError):
            fn(np.zeros((12, 12), dtype=np.float32))
        with pytest.raises(ValueError):
            fn(np.full((12, 12, 3), 255.0, dtype=np.float32))


def test_channel_order_affects_color_sensitive_metrics():
    image = np.zeros((32, 32, 3), dtype=np.float64)
    image[..., 0] = 0.8
    image[..., 1] = 0.2
    image[..., 2] = 0.1
    assert uiqm(image) != uiqm(image[..., ::-1])
    assert uciqe(image) != uciqe(image[..., ::-1])
