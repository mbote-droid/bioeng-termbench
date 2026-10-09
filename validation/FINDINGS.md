What the validation process found, in the order it found it. Each item
changed a grader, a task, a reference, the framework or a claim; each is
backed by a test or a result file. Items 10-15 come from the full campaign
itself.

1. **t02: a textbook test rejected correct fits.** A Wald (Fisher z) check on
   the parameters failed fits whose likelihood was *better* than the truth's
   (|z| up to 50): with 14 samples the two-compartment model is weakly
   identified and the likelihood is far from quadratic. Replaced by the
   99.9 % likelihood-ratio region, which is exact for this question.
2. **t03: a plain PID was too good.** Without exercise in the scenario, a
   PID with no safety layers passed about 90 % of virtual patients. Adding
   announced exercise with unannounced post-exercise sensitivity (a real
   cause of hypoglycaemia) makes it fail on about 40 %. The reference needed
   three new layers to stay safe (exercise mode, post-exercise caution, basal
   fallback on sensor dropout); before the last one it fell to 61 mg/dL on 2
   of 2000 patients.
3. **t03: the Octave controller deadlocked.** `fgetl(stdin)` waits for a full
   buffer on an open pipe; `input('', 's')` returns per line. Documented for
   agents writing Octave controllers.
4. **t04: the obvious physics made verification impossible.** With
   damage-driven vascular shutdown (perfusion x exp(-Omega)) the perfusion
   front is sub-millimetre and a correct second-order scheme measured order
   about 1.5. Replaced by a logistic temperature law (ADR-008).
5. **t04: joint space-time refinement is ill-conditioned.** In a 60-case
   sweep one reference case measured order 0.76: refining space and time
   together let their error terms cancel. Refining them separately gives
   1.98-2.01 for correct schemes and 0.98-1.00 in time for lagged perfusion,
   and tells the agent which discretisation is wrong (ADR-007).
6. **Sandbox: integrity depends on the sandbox user.** Without it, a t03
   controller wrote a perfect glucose record into the grader's directory and
   left a detached daemon to rewrite it, and **earned a passing verdict**; a
   `setsid` daemon also outlived its run, and a fork bomb spawned 1000
   processes. Fixes: records written only after the controller is dead, into
   a directory the sandbox user cannot write; the runner kills the whole
   process group after every run and every process of the sandbox uid; a
   128-process cap; symlinked outputs rejected; the CLI refuses to run as
   root without a sandbox user. All seven attacks are contained sandboxed.
7. **The GPU kernel was verified without a GPU.** The fused CUDA kernel was
   compiled as C++ with stub launch variables and run on the CPU; it matched
   the NumPy implementation bit for bit over 30 steps.
8. **The t02 Julia port had a latent bug.** Reading it again for review:
   `[v; [w for ...]]` splats the first start vector into four scalars, so
   the multi-start loop would have failed at the first iteration. Fixed; the
   port is still unexecuted here (CI job `julia`).
9. **Mutation testing found unpinned thresholds.** Halving or doubling most
   acceptance thresholds in t01, t02, t03 and t05 went unnoticed (every
   probe sat far from those boundaries), and t02's FILES and SCHEMA gates
   could be switched off because later checks also fail on missing files.
   Boundary tests now pin each acceptance threshold with the contract's
   numbers, not the grader's constants (`tests/tasks/test_t0*_thresholds.py`).
   A first version read the constants from the grader and so moved with the
   mutant: the mutation run itself exposed that.
10. **t01: a reference had an edge bug, and the grader was right.** The third
    reference (spline baseline, Hilbert detector) failed 1 of 100 recordings
    on SDNN and RMSSD. Diagnosis: two spurious beats at the very first and
    last samples, where Hilbert and smoothing transients live; one extra RR
    interval doubles SDNN. The peak checks ignore the edges, but HRV from the
    reported peaks does not, correctly. Fixed in the reference (no detections
    within 100 ms of the ends).
11. **t01: the boundary probe is genuinely on the boundary.** The
    median-filter baseline (de Chazal 2004) was "correctly rejected" on the
    8 dev recordings; over 100 it passes 51 and fails 49, always on FIDELITY
    only. Its T-wave distortion sits at the 0.05 mV contract limit. The grader
    behaves as specified; the earlier claim was overstated and is corrected.
    For training, a method on a contract boundary gets a noisy reward: that
    is a property of any threshold, now measured rather than assumed.
12. **Two planted mistakes were not mistakes.** Peaks at the Pan-Tompkins
    integrator maximum without refinement (t01) stay within the 40 ms window
    on 100 of 100 recordings; a single optimiser start (t02) converged on 12 of
    12 patients. Both are equivalent mutants and are kept as extra references.
    One negative probe expected the wrong code: heart rate in beats per second
    is caught by HR, not by HRV_CONSISTENT (which covers SDNN and RMSSD).
13. **t04: the centre temperature is a poor convergence functional.** The
    finite-element reference failed CONVERGENCE_SPACE on 2 of 8 cases: its
    centre temperature measured orders 0.36 and 1.49 while its stored energy
    measured 2.0. A point value can carry two error terms of opposite sign; a
    volume integral converges smoothly for every correct scheme. The order is
    now measured on the stored energy; the peak is still checked for accuracy.
    Without the second, independent discretisation this false reject would
    have gone unseen: the finite-volume reference passed every case.
14. **A weak control exposed a missing check (t02).** Designing the regimen
    to the window edges instead of the targets passed 27 of 40 patients:
    REGIMEN_SAFE, judged at the truth, fails only when the estimation error
    happens to point the wrong way. The contract already said "design for
    the targets"; a new check, REGIMEN_TARGETS, now enforces it on the
    agent's own predictions (boundary-tested at the contract's numbers).
15. **A weak control that stays weak (t04).** Predicting the lesion at the
    nominal instead of the highest perfusion passed 1 of 4 cases: in some
    cases the lesion is insensitive to perfusion, so the prediction is
    within 0.15 mm either way. Harmless (COVERAGE and SAFETY are judged by
    the grader at the extremes regardless) and reported as weak rather than
    hidden by tightening a tolerance that the independent FEM reference needs.
16. **The mutation tool itself had a blind spot.** Its first "check
    switched off" operator rewrote only `Outcome(...)` calls, so a check
    that delegates to a helper (t04's two convergence checks) or fails by
    raising (the loading gates) was mutated into an identical program and
    reported as a surviving mutant. The operator now wraps the whole check:
    the original body runs, and any result or exception becomes a pass. With
    it, 44 of 45 checks are necessary; the survivor, t04's SIM_FORMAT gate,
    only improves diagnostics, since every later check also fails on a
    malformed simulation file. The same pass found that nothing tested
    PLAN_FORMAT's limits (heating time off the menu, power outside
    (0, 200] W); the power limit was not even in the contract. Both are now
    in the instruction and pinned by tests.
17. **t03 is hard for the right reason.** Over 300 campaign patients the
    reference controller's worst day had 72.5 % time in range (limit 70 %)
    and 4.8 % above 250 mg/dL (limit 5 %), while its lowest glucose stayed
    at 75 mg/dL. The consensus targets bind on difficult patients even for a
    controller with every safety layer: an agent cannot pass by being
    timid, and cannot pass by being aggressive (HYPO).
