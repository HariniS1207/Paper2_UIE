"""Explicit RGB/[0,1] implementations of UCIQE and a Panetta UIQM variant.

The UIQM component definitions follow Panetta et al. (2016): alpha-trimmed
opponent-color statistics, Sobel-weighted EME sharpness, and PLIP logAMEE
contrast. Numerical conventions are made explicit here because published
implementations differ in pooling, border, and zero handling.
"""
from __future__ import annotations

import math
import numpy as np


def _rgb01(image: np.ndarray) -> np.ndarray:
    x = np.asarray(image, dtype=np.float64)
    if x.ndim != 3 or x.shape[2] != 3:
        raise ValueError("expected HxWx3 RGB image")
    if not np.isfinite(x).all():
        raise ValueError("image contains non-finite values")
    if x.min() < 0 or x.max() > 1:
        raise ValueError("expected RGB values in [0,1]")
    return x


def _lab(image: np.ndarray) -> np.ndarray:
    """sRGB D65 to CIE 1976 Lab, D65/2 degree observer, L*=0..100."""
    x = _rgb01(image)
    linear = np.where(x <= 0.04045, x / 12.92, ((x + 0.055) / 1.055) ** 2.4)
    xyz = linear @ np.array([[0.4124564, 0.3575761, 0.1804375],
                             [0.2126729, 0.7151522, 0.0721750],
                             [0.0193339, 0.1191920, 0.9503041]]).T
    xyz /= np.array([0.95047, 1.0, 1.08883])
    d = 6 / 29
    f = np.where(xyz > d**3, np.cbrt(xyz), xyz / (3*d*d) + 4/29)
    fx, fy, fz = np.moveaxis(f, -1, 0)
    return np.stack((116*fy-16, 500*(fx-fy), 200*(fy-fz)), axis=-1)


def uciqe(image: np.ndarray) -> float:
    """Yang & Sowmya UCIQE with population std and 1% tail-mean contrast."""
    lab = _lab(image)
    light, a, b = np.moveaxis(lab, -1, 0)
    chroma = np.hypot(a, b)
    saturation = np.divide(chroma, light, out=np.zeros_like(chroma), where=light > 0)
    flat_l = light.ravel()
    n_tail = max(1, int(math.ceil(0.01 * flat_l.size)))
    ordered = np.sort(flat_l)
    contrast = float(ordered[-n_tail:].mean() - ordered[:n_tail].mean())
    value = 0.4680 * float(chroma.std(ddof=0)) + 0.2745 * contrast + 0.2576 * float(saturation.mean())
    return float(value)


def _trim_stats(v: np.ndarray, trim: float = 0.1) -> tuple[float, float]:
    s = np.sort(v.ravel())
    k = int(math.floor(trim * s.size))
    kept = s[k:s.size-k] if 2*k < s.size else s
    return float(kept.mean()), float(kept.var(ddof=0))


def _sobel_magnitude(x: np.ndarray) -> np.ndarray:
    p = np.pad(x, 1, mode="reflect")
    gx = (p[:-2, 2:] + 2*p[1:-1, 2:] + p[2:, 2:] - p[:-2, :-2] - 2*p[1:-1, :-2] - p[2:, :-2])
    gy = (p[2:, :-2] + 2*p[2:, 1:-1] + p[2:, 2:] - p[:-2, :-2] - 2*p[:-2, 1:-1] - p[:-2, 2:])
    return np.hypot(gx, gy)


def _eme(x: np.ndarray, block: int = 10) -> float:
    h, w = x.shape
    blocks = [(x[i:min(i+block,h), j:min(j+block,w)])
              for i in range(0,h,block) for j in range(0,w,block)]
    total = 0.0
    for q in blocks:
        lo, hi = float(q.min()), float(q.max())
        total += math.log((hi + 1e-12) / (lo + 1e-12))
    return 2.0 * total / len(blocks)


def _log_amee(x: np.ndarray, block: int = 10, gamma: float = 1026.0) -> float:
    """PLIP log-AMEE; image intensity is scaled to [0,255]."""
    h, w = x.shape
    vals = []
    for i in range(0, h, block):
        for j in range(0, w, block):
            q = x[i:min(i+block,h), j:min(j+block,w)]
            lo, hi = float(q.min()), float(q.max())
            # PLIP subtraction / addition, preserving their bounded domain.
            numerator = gamma * (hi - lo) / max(gamma - lo, 1e-12)
            denominator = hi + lo - (hi * lo / gamma)
            vals.append(math.log(max(numerator, 1e-12) / max(denominator, 1e-12)))
    return float(np.mean(vals))


def uiqm(image: np.ndarray) -> float:
    """Panetta et al. UIQM, RGB input [0,1], scaled internally to [0,255]."""
    rgb = _rgb01(image) * 255.0
    r, g, b = np.moveaxis(rgb, -1, 0)
    rg, yb = r-g, 0.5*(r+g)-b
    mrg, vrg = _trim_stats(rg)
    myb, vyb = _trim_stats(yb)
    uicm = -0.0268*math.hypot(mrg, myb) + 0.1586*math.sqrt(max(vrg+vyb, 0.0))
    channels = (r, g, b)
    weights = (0.299, 0.587, 0.114)
    uism = sum(w * _eme(c * _sobel_magnitude(c)) for c, w in zip(channels, weights))
    intensity = 0.299*r + 0.587*g + 0.114*b
    uiconm = _log_amee(intensity)
    return float(0.0282*uicm + 0.2953*uism + 3.5753*uiconm)
