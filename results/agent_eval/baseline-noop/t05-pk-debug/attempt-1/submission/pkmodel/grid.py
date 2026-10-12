"""Output time grid."""

import numpy as np


def output_times(start_h: float, stop_h: float, step_h: float) -> np.ndarray:
    return np.arange(start_h, stop_h + step_h / 2, step_h, dtype=np.float32)
