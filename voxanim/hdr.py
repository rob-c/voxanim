"""HDR and wide-gamut output (ITU-R BT.2100).

The renderer works in display-referred sRGB / BT.709 in 0..1, plus a scalar
*emission* layer of extra highlight light (glows, glints, the white
multiply/accumulate wave).  This module turns that into BT.2100 signals:

1. linearise (sRGB EOTF) and place SDR diffuse white at **203 cd/m2**, the
   ITU-R BT.2408 reference level for graphics white in HDR productions --
   so the grey studio backdrop and every voxel colour look exactly as in SDR;
2. add the emission as white light above that, up to ``HIGHLIGHT`` nits --
   that is where HDR earns its keep: glints and glows really shine;
3. convert BT.709 primaries to BT.2020 (colorimetric, BT.2087: no gamut
   stretching, colours stay what they were designed as);
4. encode with **PQ** (SMPTE ST 2084, HDR10) or **HLG** (ARIB STD-B67, the
   broadcast format; nominal 1000 cd/m2 display, reference white at 75%).

Code values come back as floats in 0..1 (full range); the encoder converts
them to limited-range 10-bit Y'CbCr with BT.2020 non-constant-luminance
matrix coefficients.
"""

from __future__ import annotations

import numpy as np

WHITE = 203.0           # cd/m2: SDR diffuse / graphics white (BT.2408)
HIGHLIGHT = 600.0       # cd/m2 added per unit of emission
PEAK = 1000.0           # mastering display peak (and HLG nominal peak)

# BT.709 -> BT.2020 for linear light (ITU-R BT.2087)
M709_2020 = np.array([[0.627404, 0.329283, 0.043313],
                      [0.069097, 0.919540, 0.011362],
                      [0.016391, 0.088013, 0.895595]], np.float32)

LUMA_2020 = np.array([0.2627, 0.6780, 0.0593], np.float32)


def srgb_to_linear(v):
    """sRGB code values (0..1) to linear light (0..1), the IEC 61966-2-1 EOTF."""
    v = np.clip(v, 0.0, 1.0)
    return np.where(v <= 0.04045, v / 12.92, ((v + 0.055) / 1.055) ** 2.4).astype(np.float32)


def display_light(rgb, emit=None):
    """Display light in cd/m2, BT.2020 primaries, ``(h, w, 3)``."""
    lin = srgb_to_linear(rgb) @ M709_2020.T * np.float32(WHITE)
    if emit is not None:
        lin = lin + (np.clip(emit, 0.0, None) * np.float32(HIGHLIGHT))[..., None]
    return np.minimum(lin, np.float32(PEAK))


# ---- PQ (SMPTE ST 2084) ---------------------------------------------------------

_M1, _M2 = 2610 / 16384, 2523 / 4096 * 128
_C1, _C2, _C3 = 3424 / 4096, 2413 / 4096 * 32, 2392 / 4096 * 32


def pq_encode(nits):
    """Absolute light (cd/m2) to a PQ signal in 0..1 (SMPTE ST 2084 inverse EOTF)."""
    y = np.clip(np.asarray(nits, np.float32) / 10000.0, 0.0, 1.0) ** _M1
    return ((_C1 + _C2 * y) / (1.0 + _C3 * y)) ** _M2


def pq_decode(code):
    """PQ signal in 0..1 back to cd/m2 (the ST 2084 EOTF)."""
    p = np.clip(np.asarray(code, np.float64), 0, 1) ** (1 / _M2)
    return 10000.0 * (np.maximum(p - _C1, 0) / (_C2 - _C3 * p)) ** (1 / _M1)


# ---- HLG (ARIB STD-B67 / BT.2100) ---------------------------------------------------

_A, _B, _C = 0.17883277, 0.28466892, 0.55991073
_GAMMA = 1.2                     # system gamma for a 1000 cd/m2 display


def hlg_oetf(e):
    """Normalised scene light (0..1) to an HLG signal (BT.2100 OETF)."""
    e = np.clip(np.asarray(e, np.float32), 0.0, 1.0)
    return np.where(e <= 1 / 12, np.sqrt(3 * e),
                    _A * np.log(np.maximum(12 * e - _B, 1e-12)) + _C).astype(np.float32)


def hlg_encode(nits):
    """Display light (BT.2020, cd/m2) -> HLG signal, via the inverse OOTF."""
    d = np.asarray(nits, np.float32) / np.float32(PEAK)
    yd = np.maximum(d @ LUMA_2020, 1e-9)[..., None]
    scene = d * yd ** ((1 - _GAMMA) / _GAMMA)
    return hlg_oetf(scene)


# ---- whole frames -------------------------------------------------------------------


def encode(rgb, emit, transfer):
    """Renderer output -> BT.2100 code values in 0..1, ``(h, w, 3)`` float32."""
    nits = display_light(rgb, emit)
    if transfer == "pq":
        return pq_encode(nits)
    if transfer == "hlg":
        return hlg_encode(nits)
    raise ValueError(transfer)


def to_uint16(code):
    """Code values in 0..1 to full-range 16-bit integers."""
    return np.clip(np.round(np.asarray(code) * 65535.0), 0, 65535).astype(np.uint16)


def light_levels(rgb, emit):
    """``(max content light, frame-average light)`` in cd/m2, as HDR10 MaxCLL /
    MaxFALL count them (max of R, G, B per pixel)."""
    m = display_light(rgb, emit).max(-1)
    return float(m.max()), float(m.mean())


def preview_sdr(rgb, emit):
    """A quick SDR look at the HDR grade: highlights compressed with a soft
    shoulder above white (for checking on an SDR monitor)."""
    lin = srgb_to_linear(rgb) + (np.clip(emit, 0, None) * HIGHLIGHT / WHITE)[..., None]
    out = np.where(lin <= 0.8, lin, 0.8 + 0.2 * (1 - np.exp(-(lin - 0.8) / 0.2)))
    out = np.clip(out, 0, 1)
    return np.where(out <= 0.0031308, out * 12.92, 1.055 * out ** (1 / 2.4) - 0.055)
