#!/usr/bin/env python3
"""Independent reference solution for t01 using different methods from the oracle.

Usage: python pipeline_spline_hilbert.py --input ecg_raw.csv --fs 360 --outdir out

Methods (numpy/scipy only; none shared with the oracle's IIR filters or
Pan-Tompkins detector):
  * Mains: block-wise least-squares fit of a 60 Hz sine/cosine pair (2 s blocks,
    Hann-weighted overlap-add), subtracted. Tracks slow amplitude and phase drift
    without a notch filter's ringing.
  * R peaks: Savitzky-Golay smoothing differentiator, Hilbert-transform envelope
    (Benitez et al., Comput Biol Med 2001), adaptive threshold at 50 % of the
    running 2 s maximum, and T-wave discrimination: a candidate within 360 ms of
    the previous beat is kept only if its envelope is at least half the previous
    beat's.
  * Baseline: a cubic spline through one knot per cardiac cycle placed in the
    TP segment (the isoelectric interval between the end of T and the start of
    the next P), then subtracted. This is the classical isoelectric-reference
    method (Meyer and Keiser, Comput Biomed Res 1977). It preserves the ST
    segment and T wave because the baseline is only sampled where the heart is
    electrically quiet.
"""

import argparse
import itertools
import json
import os

import numpy as np
from scipy.interpolate import CubicSpline
from scipy.ndimage import maximum_filter1d
from scipy.signal import hilbert, savgol_filter

LINE_HZ = 60.0


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
    edge = int(0.1 * fs)  # Hilbert and smoothing transients: no complete beat this close to the ends
    peaks, heights = [], []
    for s in starts:
        lo, hi = max(s - half, 0), min(s + 2 * half, len(y))
        p = lo + int(np.argmax(y[lo:hi]))
        if p < edge or p >= len(y) - edge:
            continue
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


def remove_baseline(x, peaks, fs):
    """Spline through TP-segment knots, one per RR interval."""
    knots_t, knots_v = [], []
    half = int(0.04 * fs)
    for a, b in itertools.pairwise(peaks):
        c = a + int(0.56 * (b - a))  # end of T to start of P, for this model and physiology
        lo, hi = max(c - half, 0), min(c + half + 1, len(x))
        knots_t.append(c)
        knots_v.append(np.median(x[lo:hi]))
    knots_t = np.array(knots_t)
    if len(knots_t) < 4:
        return x - np.median(x)
    spline = CubicSpline(knots_t, knots_v, bc_type="natural")
    idx = np.arange(len(x))
    base = spline(np.clip(idx, knots_t[0], knots_t[-1]))
    # Outside the first and last knot, extend linearly using the end slopes.
    base[idx < knots_t[0]] = knots_v[0] + spline(knots_t[0], 1) * (idx[idx < knots_t[0]] - knots_t[0])
    base[idx > knots_t[-1]] = knots_v[-1] + spline(knots_t[-1], 1) * (idx[idx > knots_t[-1]] - knots_t[-1])
    return x - base


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
    x = remove_mains(x, a.fs)
    peaks = detect_rpeaks(x, a.fs)
    y = remove_baseline(x, peaks, a.fs)
    peaks = detect_rpeaks(y, a.fs)  # second pass on the baseline-free signal

    os.makedirs(a.outdir, exist_ok=True)
    np.savetxt(os.path.join(a.outdir, "ecg_filtered.csv"), np.column_stack([t, y]),
               delimiter=",", header="time_s,ecg_mV", comments="", fmt="%.6f")
    np.savetxt(os.path.join(a.outdir, "rpeaks.csv"), np.column_stack([peaks, t[peaks]]),
               delimiter=",", header="sample_index,time_s", comments="", fmt=["%d", "%.6f"])
    with open(os.path.join(a.outdir, "metrics.json"), "w") as fh:
        json.dump(hrv(peaks, a.fs), fh, indent=2)


if __name__ == "__main__":
    main()
