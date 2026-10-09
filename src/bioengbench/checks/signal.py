"""Signal-domain measurements used by checks."""

from __future__ import annotations

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

from bioengbench.checks.io import FloatArray


def band_power(x: FloatArray, fs: float, band: tuple[float, float], seg_s: float = 20.0) -> float:
    """Power in `band` (Hz) from Welch's PSD, integrated with the trapezoid rule.

    Units: if x is in mV, the result is in mV^2. Ratios of band powers are
    dimensionless, and 10*log10(ratio) is the attenuation in dB.
    """
    f, p = welch(x, fs=fs, nperseg=int(seg_s * fs))
    mask = (f >= band[0]) & (f <= band[1])
    return float(np.trapezoid(p[mask], f[mask]))


def bandlimited_rmse(
    x: FloatArray, ref: FloatArray, fs: float, band: tuple[float, float], trim: int
) -> float:
    """RMS difference between x and ref after both pass the same zero-phase band-pass.

    Comparing inside a band isolates the property being graded (for ECG,
    1-40 Hz morphology) from legitimate choices outside it (high-pass cut-off,
    DC offset of an AC-coupled amplifier, residual broadband noise).
    """
    sos = butter(4, band, btype="bandpass", fs=fs, output="sos")
    a = sosfiltfilt(sos, x)[trim:-trim]
    b = sosfiltfilt(sos, ref)[trim:-trim]
    return float(np.sqrt(np.mean((a - b) ** 2)))
