# Failure mode and effects analysis

Two analyses. Part A is the medical-device FMEA that shaped t03's scenarios:
each high-risk failure mode of a closed-loop insulin system has a scenario
in the grader and a layer in the reference controller. Part B applies the
same method to the benchmark itself: how a grader can fail and what detects
it.

Scales 1-10: S severity, O occurrence, D detection (10 = undetectable).
RPN = S x O x D. Actions are taken for RPN >= 100 or S >= 9.

## Part A: closed-loop insulin delivery (t03)

| # | failure mode | cause | effect | S | O | D | RPN | scenario in grader | mitigation in reference controller | O' | D' | RPN' |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | over-delivery after meals | integral and derivative action keep acting on insulin already given (stacking) | late hypoglycaemia | 9 | 7 | 6 | 378 | three unannounced meals | insulin-on-board limit | 2 | 3 | 54 |
| A2 | false-high sensor reading | compression release, artefact | bolus-sized correction for a glucose that is not there | 9 | 4 | 7 | 252 | +60-90 mg/dL spike for 15 min | Kalman innovation gate, no above-basal delivery while rejecting | 2 | 3 | 54 |
| A3 | false-low reading | pressure on the sensor at night (compression low) | needless suspension, rebound high | 5 | 5 | 6 | 150 | -40-60 mg/dL for 20 min overnight | innovation gate; re-initialise after 20 min of consistent disagreement | 3 | 3 | 45 |
| A4 | sensor dropout | radio loss, warm-up | controller acts on stale estimate | 8 | 5 | 5 | 200 | 30 min of `null` readings | prediction only; basal fallback (no correction insulin) | 2 | 3 | 48 |
| A5 | exercise | unannounced or announced physical activity raises insulin action | hypoglycaemia during and up to hours after | 9 | 6 | 6 | 324 | 45 min afternoon exercise, announced 60 min ahead; post-exercise sensitivity unannounced | halve basal and raise target when announced; reduced gains and IOB allowance for 3 h after | 3 | 3 | 81 |
| A6 | sluggish correction | timid gains | prolonged hyperglycaemia | 6 | 5 | 3 | 90 | correction from 250 mg/dL | PID tuned per basal rate | 3 | 2 | 36 |
| A7 | oscillation (limit cycle) | phase lag (integral-dominant control, sensor lag) | repeated highs and lows | 8 | 4 | 5 | 160 | settling window of the correction | derivative action on the KF rate, integral clamp | 2 | 2 | 32 |
| A8 | command beyond the pump | unbounded output | pump saturates silently; model mismatch | 5 | 3 | 4 | 60 | ACTUATOR check | output clipped to [0, pump max] | 1 | 1 | 5 |
| A9 | controller crash or hang | unhandled `null`, exceptions, slow code | no insulin decisions | 9 | 3 | 2 | 54 | protocol deadlines | defensive parsing; constant-time step | 1 | 1 | 9 |
| A10 | predicted low not acted on | no look-ahead | hypoglycaemia from falling glucose | 9 | 5 | 5 | 225 | all scenarios | suspend if G < 85 or G + 30 dG/dt < 90 | 2 | 3 | 54 |

Residual (measured, `docs/VALIDATION_REPORT.md`): over 4000 random virtual
patients the reference controller's lowest glucose was 73 mg/dL; a plain PID
without these layers falls below 70 mg/dL on about 40 % of patients.

## Part B: the benchmark's graders

| # | failure mode | effect on training | S | O | D | RPN | detection now in place | O' | D' | RPN' |
|---|---|---|---|---|---|---|---|---|---|---|
| B1 | false accept (wrong solution passes) | reward hacking learned | 10 | 5 | 7 | 350 | negative controls and mutants per task; P(reward over 8 cases) bound | 2 | 2 | 40 |
| B2 | false reject (correct solution fails) | correct behaviour punished; noise in reward | 8 | 5 | 6 | 240 | independent references (other methods, languages) on many seeds; Wilson bounds | 2 | 2 | 32 |
| B3 | check never fires (dead check) | silent loss of coverage | 7 | 4 | 8 | 224 | grader mutation testing (`check_off`) | 2 | 2 | 28 |
| B4 | threshold tuned to the reference | rejects other valid methods | 7 | 5 | 6 | 210 | thresholds from physics or practice; margin table; FEM and Octave references | 2 | 3 | 42 |
| B5 | grader's own model wrong | everything judged against wrong truth | 10 | 3 | 7 | 210 | exact solutions, LSODA cross-checks, observed order, independent re-implementations | 1 | 2 | 20 |
| B6 | flaky verdicts (thread count, hash order) | noisy reward | 6 | 4 | 4 | 96 | single-threaded BLAS, hash seed, canonical JSON; CI byte comparison | 1 | 1 | 6 |
| B7 | sandbox escape or record forgery | reward without work | 10 | 3 | 7 | 210 | sandbox user, 0700 grader paths, reaping, red team | 1 | 2 | 20 |
| B8 | verification metric ill-conditioned (t04 joint refinement) | false rejects of correct solvers | 7 | 3 | 7 | 147 | found by the 60-case sweep (order 0.76); fixed by separate space and time refinement | 1 | 2 | 14 |
| B9 | negative control too weak (passes often) | false sense of coverage | 6 | 5 | 5 | 150 | per-probe pass rate with confidence interval in the report | 2 | 2 | 24 |
| B10 | task contract ambiguous | correct-by-reading solutions fail | 7 | 4 | 5 | 140 | contract states every criterion and number; boundary probes (t01 median baseline) | 2 | 3 | 42 |

Items B8 and the t01 median-baseline finding were discovered by the
process, not anticipated: the process works only if the campaign is rerun
after every grader change (`make validate`).
