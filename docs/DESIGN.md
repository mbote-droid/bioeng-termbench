# Task design

Each task is a real engineering workflow in which the mathematics drives a
decision, and in which a plausible but wrong solution exists that a weak
grader would accept. For each task: the decision, the hidden cases, the
checks, and the mistakes the checks are designed to catch.

Common to all: hidden cases from 128-bit seeds, binary reward over every
check on every case, `RUN` fails if the submission crashes, times out,
breaks the protocol or writes a symbolic link as an output.

## t01 ECG: denoise, detect beats, report heart-rate variability

**Workflow.** An ambulatory ECG with baseline wander, 60 Hz mains and
electrode noise. Clean it without distorting diagnostic morphology, find the
R peaks, compute HR, SDNN, RMSSD.

**Hidden cases.** McSharry model with per-case heart rate, rhythm
variability, morphology and amplitude; 60 s at 360 Hz.

| code | requirement | catches |
|---|---|---|
| BASELINE | wander power (0.05-0.40 Hz) reduced >= 10 dB | no high-pass |
| POWERLINE | 59-61 Hz power reduced >= 13 dB | no notch |
| FIDELITY | 1-40 Hz RMSE <= 0.05 mV vs noise-free ECG | over-filtering; a median baseline sits at this limit (T-wave distortion) |
| AMPLITUDE | R amplitude within 15 % | volts vs millivolts |
| PEAK_SENS / PEAK_PREC | >= 0.98 within 40 ms | delayed peaks, T waves counted |
| NBEATS | within 2 | missed or extra beats |
| HRV_CONSISTENT | metrics follow from the reported peaks | unit slips in HRV |

The 40 ms window is stricter than the AAMI 150 ms convention so a peak on
the QRS upstroke counts but a T wave never does.

## t02 Pharmacokinetics: fit, quantify uncertainty, design a regimen

**Decision.** A maintenance regimen (interval from a menu, dose) that keeps
the *true* patient in the therapeutic window, using parameters estimated
from 14 noisy samples.

**Hidden cases.** Two-compartment parameters by rejection sampling so a
feasible regimen exists with margin; log-normal assay error (sigma 0.10).

| code | requirement | catches |
|---|---|---|
| PARAMS_LR | inside the 99.9 % likelihood-ratio region | wrong error model, local minima |
| SE_CONSISTENT | log-parameter SEs within 20 % of Fisher | SEs of parameters, not logs |
| HALF_LIVES | within 0.5 % of the fitted model's | ln 2 missing |
| PRED_ACCURACY | prediction within 1 % of peak | Euler, bolus instead of infusion |
| REGIMEN_SELF | predicted peak/trough follow from own model | daily vs per-dose slip |
| REGIMEN_TARGETS | own predicted steady state within the 11.5-35 mg/L targets | designing to the window edges |
| REGIMEN_SAFE | window holds at the true parameters | unsafe regimens; poor fits |
| REGIMEN_EFFICIENT | daily dose <= 1.20 x optimum | needlessly high doses |

Physics in the decision: by linearity the feasible doses for each interval
form an exact interval, so the optimum needs no iterative optimiser; the gap
between targets (11.5-35 mg/L) and window (10-40 mg/L) is the margin for
estimation error.

## t03 Closed-loop insulin delivery

**Decision.** An insulin rate every 5 minutes from a noisy, faulty CGM,
without seeing the patient.

**Hidden cases.** Bergman model patients (sensitivity, basal, volumes, lag),
a 24 h day (3 unannounced meals, announced exercise, dropout, false-high
spike, compression low) and a 6 h correction from 250 mg/dL.

| code | requirement | catches |
|---|---|---|
| ACTUATOR | requests within [0, pump max] | interface violations |
| HYPO | glucose never < 70 mg/dL | insulin stacking, trusting a false high, ignoring exercise |
| TIME_IN_RANGE | >= 70 % in 70-180 | sluggish control |
| TIME_ABOVE_250 | <= 5 % | uncovered meals |
| CORRECTION_TIME | < 180 mg/dL within 2 h | timid correction |
| SETTLING | last 2 h p2p <= 50, mean 80-160 | limit cycles |

Calibration: a plain PID tracks glucose well (TIR > 90 %) but causes
hypoglycaemia on about 40 % of patients, so it almost never survives 8 hidden
cases. Exercise was added to the day scenario for this reason: without it a
plain PID passed about 90 % of patients.

## t04 Thermal ablation planning

**Decision.** Heating time (menu of six) and power for a probe at a tumour's
centre: cover tumour + 5 mm at the highest perfusion, stay at or below 100 C
at the lowest, with near-minimal energy.

**Hidden cases.** Source width 5-9 mm, tumour radius 4-8 mm, nominal
perfusion 0.001-0.006 1/s (+-20 % uncertainty), rejection-sampled so the
optimal plan leaves a >= 5 % power window. The optimum is the shortest
feasible heating time in 40 % of cases, a longer one otherwise.

| code | requirement | catches |
|---|---|---|
| CONVERGENCE_SPACE / _TIME | observed order of the stored energy >= 1.6 | first-order schemes, lagged perfusion, ignoring the resolution |
| ACCURACY | peak 0.05 C, radius 0.1 mm, energy 0.2 % | wrong geometry, constant perfusion, wrong constants |
| ENERGY_BUDGET | closes to 0.1 %, terms within 1 % | non-conservative bookkeeping |
| PLAN_CONSISTENT | predictions within 0.15 mm / 0.15 C | guessed predictions |
| COVERAGE | radius >= R_t + 5 mm at highest perfusion | non-robust plans |
| SAFETY | peak <= 100 C at lowest perfusion | overheating |
| EFFICIENCY | energy <= 1.10 x minimum | longest-time plans, large margins |

## t05 Debug a pharmacokinetics package

**Decision.** What is wrong, judged against a written specification.

**Defects.** Units (CL mL/min converted the wrong way), an adaptive explicit
solver over the whole horizon with loose tolerance (inaccurate, and it steps
over a short infusion), boluses skipped, float32 time grid, doses keyed by
start time. Two are reported by users; three are silent.

| code | scenario | the defect it isolates |
|---|---|---|
| STANDARD | three infusions q12h | units, solver tolerance |
| SHORT_INFUSION | short late infusion, sparse output | solver steps over the pulse |
| BOLUS | bolus then infusion | boluses skipped |
| LONG_HORIZON | 14 days at 0.1 h | float32 grid drift |
| COINCIDENT_DOSES | loading + maintenance at t = 0 | doses keyed by time |

The bug-isolation matrix (`tests/tasks/test_t05.py`) injects each defect
alone into the fixed package and shows it fails exactly its scenario.

## Difficulty and expert time

| task | difficulty | expert minutes | what makes it hard |
|---|---|---|---|
| t01 | medium | 45 | fidelity vs suppression trade-off; strict peak window |
| t02 | hard | 90 | weak identifiability; uncertainty; design at the truth |
| t03 | hard | 120 | safety layers a PID lacks; faults; unannounced meals |
| t04 | hard | 150 | provable second order; nonlinear coupling; robust planning |
| t05 | medium | 60 | three silent defects of different kinds |
