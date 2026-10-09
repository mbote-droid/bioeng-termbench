#!/usr/bin/env python3
"""Reference solution (Python): ECG denoising, R-peak detection, HRV metrics.

Usage: python pipeline.py --input ecg_raw.csv --fs 360 --outdir out

Design choices (documented assumptions):
  * 2nd-order Butterworth high-pass at 0.5 Hz, applied forward and backward
    (zero phase), removes baseline wander. 0.5 Hz is the monitoring-bandwidth
    convention; diagnostic ST analysis would use 0.05 Hz.
  * IIR notch at 60 Hz (Q = 30) removes powerline interference. Change
    LINE_HZ for 50 Hz mains regions.
  * R peaks: band-pass 5-15 Hz, derivative, squaring, 150 ms moving-window
    integration (Pan-Tompkins style), peak picking with a 250 ms refractory
    period, then refinement to the maximum of the filtered ECG within 80 ms.
"""
import argparse
import json
import os

import numpy as np
from scipy.signal import butter, find_peaks, iirnotch, sosfiltfilt, tf2sos

LINE_HZ = 60.0


def denoise(x, fs):
    sos_hp = butter(2, 0.5, btype="highpass", fs=fs, output="sos")
    b, a = iirnotch(LINE_HZ, 30.0, fs=fs)
    y = sosfiltfilt(sos_hp, x)
    return sosfiltfilt(tf2sos(b, a), y)


def detect_rpeaks(y, fs):
    sos_bp = butter(2, [5.0, 15.0], btype="bandpass", fs=fs, output="sos")
    d = np.gradient(sosfiltfilt(sos_bp, y))
    win = int(round(0.150 * fs))
    integ = np.convolve(d ** 2, np.ones(win) / win, mode="same")
    thr = 0.30 * np.percentile(integ, 99)
    cand, _ = find_peaks(integ, height=thr, distance=int(round(0.25 * fs)))
    half = int(round(0.08 * fs))
    peaks = []
    for c in cand:
        lo, hi = max(c - half, 0), min(c + half + 1, len(y))
        peaks.append(lo + int(np.argmax(y[lo:hi])))
    return np.unique(np.array(peaks, dtype=int))


def hrv(peaks, fs):
    rr = np.diff(peaks) / fs * 1000.0  # ms
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
    y = denoise(x, a.fs)
    peaks = detect_rpeaks(y, a.fs)

    os.makedirs(a.outdir, exist_ok=True)
    np.savetxt(os.path.join(a.outdir, "ecg_filtered.csv"), np.column_stack([t, y]),
               delimiter=",", header="time_s,ecg_mV", comments="", fmt="%.6f")
    np.savetxt(os.path.join(a.outdir, "rpeaks.csv"), np.column_stack([peaks, t[peaks]]),
               delimiter=",", header="sample_index,time_s", comments="",
               fmt=["%d", "%.6f"])
    with open(os.path.join(a.outdir, "metrics.json"), "w") as fh:
        json.dump(hrv(peaks, a.fs), fh, indent=2)


if __name__ == "__main__":
    main()
