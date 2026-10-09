#!/usr/bin/env python3
"""Boundary probe for t01: a published method the grader is EXPECTED to reject.

Usage: python pipeline_median_hilbert.py --input ecg_raw.csv --fs 360 --outdir out

Why it exists: the task contract requires the cleaned trace to preserve QRS and
T-wave morphology within 0.05 mV RMS (1-40 Hz). Cascaded median-filter baseline
removal (200 ms then 600 ms windows; de Chazal et al., IEEE TBME 2004) is widely
used for beat detection, but because the 600 ms window spans the T wave, the
estimated baseline partly follows the T wave and the subtraction distorts it.
Measured on this task: up to about 0.09 mV RMS, concentrated in the T-wave
region. Everything else in this script is correct, so the grader should fail it
on FIDELITY only. This shows the grader discriminates a specific, clinically
relevant distortion rather than penalising "not the oracle's method".

Methods:
  * Baseline: two cascaded median filters, 200 ms then 600 ms, subtracted.
  * Mains: block-wise least-squares fit of a 60 Hz sine and cosine (2 s blocks,
    Hann-weighted overlap-add) is subtracted. This tracks slow amplitude and
    phase drift without a notch filter's ringing.
  * R peaks: same detector as pipeline_spline_hilbert.py (Savitzky-Golay
    derivative, Hilbert envelope, adaptive threshold, T-wave discrimination).
"""

import argparse
import json
import os

import numpy as np
from scipy.ndimage import maximum_filter1d, median_filter
from scipy.signal import hilbert, savgol_filter

LINE_HZ = 60.0


def remove_baseline(x, fs):
    w1 = int(round(0.2 * fs)) | 1  # odd widths keep the median centred
    w2 = int(round(0.6 * fs)) | 1
    base = median_filter(median_filter(x, size=w1, mode="reflect"), size=w2, mode="reflect")
    return x - base


def remove_mains(x, fs, block_s=2.0):
    n = len(x)
    t = np.arange(n) / fs
    blk = int(block_s * fs)
    hop = blk // 2
    est = np.zeros(n)
    weight = np.zeros(n)
    window = np.hanning(blk)
    for start in range(0, max(n - blk, 0) + hop, hop):
        seg = slice(start, min(start + blk, n))
        ts = t[seg]
        design = np.column_stack([np.sin(2 * np.pi * LINE_HZ * ts), np.cos(2 * np.pi * LINE_HZ * ts)])
        coef, *_ = np.linalg.lstsq(design, x[seg], rcond=None)
        w = window[: len(ts)]
        est[seg] += w * (design @ coef)
        weight[seg] += w
    weight[weight == 0] = 1.0
    return x - est / weight


def detect_rpeaks(y, fs):
    win = int(round(0.025 * fs)) | 1
    d = savgol_filter(y, win, polyorder=2, deriv=1)
    env = np.abs(hilbert(d))
    k = int(0.04 * fs)
    env = np.convolve(env, np.ones(k) / k, mode="same")
    thr = 0.5 * maximum_filter1d(env, size=int(2.0 * fs), mode="nearest")
    above = env > thr
    starts = np.flatnonzero(above & ~np.roll(above, 1))
    half = int(0.06 * fs)
    t_wave_window = int(0.36 * fs)
    hard_refractory = int(0.25 * fs)
    peaks, heights = [], []
    for s in starts:
        lo, hi = max(s - half, 0), min(s + 2 * half, len(y))
        p = lo + int(np.argmax(y[lo:hi]))
        h = float(env[lo:hi].max())
        if peaks:
            gap = p - peaks[-1]
            if gap < hard_refractory or (gap < t_wave_window and h < 0.5 * heights[-1]):
                if gap < hard_refractory and y[p] > y[peaks[-1]]:
                    peaks[-1], heights[-1] = p, h
                continue
        peaks.append(p)
        heights.append(h)
    return np.array(peaks, dtype=int)


def hrv(peaks, fs):
    rr = np.diff(peaks) / fs * 1000.0
    return {
        "mean_hr_bpm": float(60000.0 / rr.mean()),
        "sdnn_ms": float(rr.std(ddof=1)),
        "rmssd_ms": float(np.sqrt(np.mean(np.diff(rr) ** 2))),
        "n_beats": int(len(peaks)),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--fs", type=float, required=True)
    ap.add_argument("--outdir", required=True)
    a = ap.parse_args()

    data = np.genfromtxt(a.input, delimiter=",", names=True)
    t, x = data["time_s"], data["ecg_mV"]
    y = remove_baseline(remove_mains(x, a.fs), a.fs)
    peaks = detect_rpeaks(y, a.fs)

    os.makedirs(a.outdir, exist_ok=True)
    np.savetxt(os.path.join(a.outdir, "ecg_filtered.csv"), np.column_stack([t, y]),
               delimiter=",", header="time_s,ecg_mV", comments="", fmt="%.6f")
    np.savetxt(os.path.join(a.outdir, "rpeaks.csv"), np.column_stack([peaks, t[peaks]]),
               delimiter=",", header="sample_index,time_s", comments="", fmt=["%d", "%.6f"])
    with open(os.path.join(a.outdir, "metrics.json"), "w") as fh:
        json.dump(hrv(peaks, a.fs), fh, indent=2)


if __name__ == "__main__":
    main()
