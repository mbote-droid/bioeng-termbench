"""Output time grid."""

import numpy as np


def output_times(start_h: float, stop_h: float, step_h: float) -> np.ndarray:
    n = round((stop_h - start_h) / step_h)
    return start_h + step_h * np.arange(n + 1, dtype=np.float64)
