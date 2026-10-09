"""Reference closed-loop insulin controller for t03.

Protocol: one JSON object per line on stdin, one reply per line on stdout.

    in : {"t_min", "cgm_mg_dL" (null on dropout), "basal_mU_per_min",
          "pump_max_mU_per_min", "exercise_announced"}
    out: {"insulin_mU_per_min": <rate>}

Design (each layer answers a failure mode in the FMEA):

1. Estimation. A constant-velocity Kalman filter on the CGM gives glucose and
   its rate of change. Process noise q = 0.02 (mg/dL)^2/min^3; measurement
   noise grows with glucose (CGM error is roughly proportional). Dropouts are
   handled by prediction alone.
2. Artefact rejection. A reading whose innovation exceeds max(25, 4 sqrt(S))
   is rejected (false-high spikes, compression lows). Four consecutive
   rejections (20 min) mean the glucose really moved: re-initialise on the
   sensor. While rejecting, or with no reading at all (dropout), never
   deliver above basal: commercial pumps fall back to basal on signal loss.
3. Control. PID around 115 mg/dL with gains scaled by the basal rate (a
   patient-specific proxy for insulin need): Kp = basal/50 per mg/dL,
   Td = 30 min, Ti = 300 min, integral clamped to [-basal, 2 basal].
4. Insulin-on-board limit. Insulin given above basal decays with a 60 min
   time constant; while it exceeds what the glucose excess above 100 mg/dL
   justifies, delivery is capped at basal. This is what prevents the late
   post-meal hypoglycaemia that a plain PID causes (insulin stacking).
5. Predictive low-glucose suspend: zero insulin if glucose is below 85 or
   predicted below 90 within 30 min.
6. Exercise. When exercise is announced (60 min ahead), raise the target to
   160 mg/dL and cap delivery at half the basal rate, the usual clinical
   advice for a temporary basal reduction. Exercise amplifies the action of
   insulin already on board, so the reduction must start before the bout.
   For 3 h afterwards insulin sensitivity stays raised (not announced): halve
   the gains and the insulin-on-board allowance and aim for 140 mg/dL.
"""

import json
import math
import sys

import numpy as np

DT = 5.0
TARGET = 115.0
EXERCISE_TARGET = 160.0
EXERCISE_BASAL_FRACTION = 0.5
POST_EXERCISE_MIN = 180.0
POST_EXERCISE_TARGET = 140.0


class Controller:
    def __init__(self, basal: float, umax: float) -> None:
        self.b, self.umax = basal, umax
        self.kp = basal / 50.0
        self.td, self.ti = 30.0, 300.0
        self.x: np.ndarray | None = None
        self.p = np.diag([400.0, 4.0])
        self.integral = 0.0
        self.iob = 0.0
        self.rejected = 0
        self.since_exercise = math.inf
        self.f = np.array([[1.0, DT], [0.0, 1.0]])
        self.q = np.diag([DT**3 / 3.0, DT]) * 0.02

    def _reset(self, m: float) -> None:
        self.x = np.array([m, 0.0])
        self.p = np.diag([400.0, 4.0])
        self.rejected = 0

    def _estimate(self, m: float | None) -> bool:
        if self.x is None:
            if m is None:
                return False
            self._reset(m)
        self.x = self.f @ self.x
        self.p = self.f @ self.p @ self.f.T + self.q
        if m is not None:
            r = (5.0 + 0.04 * self.x[0]) ** 2
            s = self.p[0, 0] + r
            innovation = m - self.x[0]
            if abs(innovation) < max(25.0, 4.0 * math.sqrt(s)):
                k = self.p[:, 0] / s
                self.x = self.x + k * innovation
                self.p = self.p - np.outer(k, self.p[0, :])
                self.rejected = 0
            else:
                self.rejected += 1
                if self.rejected >= 4:
                    self._reset(m)
        return True

    def step(self, m: float | None, exercise: bool = False) -> float:
        if not self._estimate(m):
            return self.b
        g, dg = float(self.x[0]), float(self.x[1])
        self.since_exercise = 0.0 if exercise else self.since_exercise + DT
        recovering = self.since_exercise < POST_EXERCISE_MIN
        target = EXERCISE_TARGET if exercise else (POST_EXERCISE_TARGET if recovering else TARGET)
        gain = 0.5 if recovering else 1.0  # insulin is more potent after exercise
        kp = self.kp * gain
        e = g - target
        self.integral = float(np.clip(self.integral + kp * e * DT / self.ti, -self.b, 2 * self.b))
        u = self.b + kp * e + self.integral + kp * self.td * dg
        if self.iob > max(0.0, g - 100.0) * self.b * 1.2 * gain:
            u = min(u, self.b)
        if g + 30.0 * dg < 90.0 or g < 85.0:
            u = 0.0
            self.integral = min(self.integral, 0.0)
        if self.rejected > 0 or m is None:
            u = min(u, self.b)  # no trusted reading: no correction insulin
        if exercise:
            u = min(u, EXERCISE_BASAL_FRACTION * self.b)
        u = float(np.clip(u, 0.0, self.umax))
        self.iob = self.iob * math.exp(-DT / 60.0) + (u - self.b) * DT
        return u


def main() -> None:
    controller: Controller | None = None
    for line in sys.stdin:
        if not line.strip():
            continue
        msg = json.loads(line)
        if controller is None:
            controller = Controller(msg["basal_mU_per_min"], msg["pump_max_mU_per_min"])
        rate = controller.step(msg["cgm_mg_dL"], bool(msg.get("exercise_announced", False)))
        sys.stdout.write(json.dumps({"insulin_mU_per_min": rate}) + "\n")
        sys.stdout.flush()


if __name__ == "__main__":
    main()
