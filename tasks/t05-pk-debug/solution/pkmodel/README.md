# pkmodel

Two-compartment intravenous pharmacokinetics for the ward dosing calculator.

## Specification

    dA1/dt = -(CL + Q)/V1 A1 + Q/V2 A2 + R(t)
    dA2/dt =  Q/V1 A1 - Q/V2 A2,         C = A1 / V1   (mg/L)

* Amounts in mg, time in h, volumes in L, Q in L/h. **CL is supplied in
  mL/min** (the laboratory's unit) and converted on input.
* Each dose is `{"start_h", "duration_h", "amount_mg"}`. `duration_h > 0` is a
  constant-rate infusion of `amount_mg / duration_h` mg/h over
  `[start_h, start_h + duration_h)`; `duration_h == 0` is an IV bolus that
  adds `amount_mg` to A1 at `start_h`. Doses may overlap or share a start time;
  every dose counts.
* The body is drug-free at t = 0 (before any dose at t = 0 is given).
* Output grid: `{"start_h", "stop_h", "step_h"}` gives the times
  `start_h + k * step_h`, k = 0 .. round((stop_h - start_h) / step_h),
  computed in double precision. Concentrations are right-continuous: a bolus
  at an output time is included at that time.
* Accuracy: concentrations within 1e-6 of the peak (relative) of the exact
  solution, for any regimen, including short infusions and long horizons.

## Command line

    python -m pkmodel.cli --params params.json --regimen regimen.json --out conc.csv

`params.json`: `CL_mL_per_min`, `V1_L`, `Q_L_per_h`, `V2_L`.
Output CSV header `time_h,conc_mg_per_L`, one row per grid time, values
written with 12 significant digits.
