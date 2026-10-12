# BioEng-TermBench: physics-aware graders for AI agents on biomedical engineering tasks

**Technical report, v1.0.0 (October 2026)**
Samuel Ngigi Mbote, MBChB, MCS (COSECSA). ORCID [0009-0003-2319-8873](https://orcid.org/0009-0003-2319-8873).
Code, data and every result file: <https://github.com/mbote-droid/bioeng-termbench>

## Abstract

AI agents are increasingly trained with rewards from automated graders. A
grader that checks only that a script ran and wrote well-formed files rewards
answers that are physically wrong: unit slips, unconverged solvers, and
controllers that are fine on average but dangerous for some patients.
BioEng-TermBench is five terminal tasks from biomedical engineering (ECG signal
processing, pharmacokinetic fitting and dosing, closed-loop insulin delivery,
thermal ablation planning and debugging a pharmacokinetics package), each with
a grader that checks engineering correctness: units, conservation, order of
accuracy, safety at the true parameters and robustness under uncertainty. The
graders are validated like measuring instruments. Thirteen correct solutions,
in Python and Octave and with different numerical methods, passed all 820
hidden cases they were graded on (0 false rejects). Each of 33 realistic
wrong solutions failed with its expected diagnostic code, with at most a 5 %
chance (95 % upper bound) of earning reward over 8 hidden cases. Of those 33,
28 (85 %; 95 % CI 69-93 %) write well-formed output on every case, so a grader
that only checks the run and the file formats would have rewarded them.
Switching off any one of 44 of the 45 checks makes the test suite fail, and
seven attacks on the grading sandbox are contained. A 3D GPU version of the
ablation solver, verified against the 1D grader model, runs a full treatment
on 17 million nodes in 12 s on a free Tesla T4.

## 1. Problem

In reinforcement learning from automated rewards the grader is the sensor in
a closed control loop: task, agent, attempt, grader, reward, model update. A
biased sensor produces a controller that optimises the bias. This is reward
hacking (Amodei et al. 2016; Skalse et al. 2022). In engineering tasks it
takes a specific form: the output has the right shape but the wrong physics.
Common examples:

* a dose reported per day instead of per infusion: a valid number in a valid file;
* a solver that ignores the resolution it was asked for: it still converges to *something*;
* an insulin controller that tracks glucose well on average but drives some
  patients below 70 mg/dL.

None of these is caught by checking that the script exited cleanly and that
the files parse. Catching them needs domain checks, and those checks have to
be validated themselves: a grader that rejects correct work is as harmful as
one that accepts wrong work, because it teaches the agent to avoid the
correct method.

**Hypotheses** (set before building the graders):

* **H1.** Graders that check only execution and output format accept at
  least 30 % of realistic wrong solutions.
* **H2.** Physics-aware graders reject at least 95 % of realistic wrong
  solutions while accepting every correct one.

## 2. System overview

Each task is in the [Harbor](https://harborframework.com) format: an
instruction, an agent image holding only what the agent may see, and a
separate verifier image holding the grader. The core is a small framework
(`src/bioengbench/core`) with tasks as plugins:

* **Hidden cases from seeds.** Each case is regenerated from a 128-bit seed
  at grading time. No truth is stored, and the generator is absent from the
  agent image, so an agent cannot recover answers by brute force.
* **Lifecycle (Template Method).** Make the case, write the agent-visible
  inputs, run the submission one or more times, check outputs. Checks are pure
  functions grouped with gates, so a missing file reports one clear failure
  instead of many confusing ones.
* **Sandbox.** The grader runs as root and the submission as an unprivileged
  `sandbox` user, with limits on memory, CPU time, file size and process
  count. After every run the grader kills the submission's process group and
  every process of the sandbox user, rejects symbolic links as outputs, and
  writes trusted records only after the submission is dead, into a directory
  it cannot write.
* **Binary reward.** Reward is 1 only if every check passes on every hidden
  case (8 by default). Every failure carries a diagnostic code, the measured
  value and the threshold.
* **Reproducibility.** Locked dependencies (`uv.lock`, hash-pinned image
  requirements), single-threaded maths libraries, canonical verdict JSON. CI
  grades the same submission twice and fails unless the verdicts are
  byte-identical.

Details: [ARCHITECTURE](ARCHITECTURE.md) (12 decision records),
[ICD](ICD.md), [THREAT_MODEL](THREAT_MODEL.md).

## 3. Task design

Every task is a real workflow in which the mathematics drives a decision,
and in which a plausible wrong solution exists that a weak grader would
accept.

| task | engineering decision | physics and mathematics that drive it | plausible wrong solution |
|---|---|---|---|
| t01 ECG | filter design that removes wander and mains without distorting diagnostic morphology; beat detection; HRV | McSharry dynamical ECG model; filter attenuation in dB bands; 40 ms peak window | over-filtering, volts for millivolts, SDNN in seconds labelled ms |
| t02 PK | a dosing regimen safe for the *true* patient from 14 noisy samples | two-compartment model, exact matrix-exponential solution; likelihood-ratio region; superposition gives the feasible dose interval exactly | designing to the window edges, ignoring estimation error |
| t03 insulin | an insulin rate every 5 min from a noisy, faulty glucose sensor | Bergman minimal model with exercise; Kalman filtering; insulin-on-board limits | a PID that tracks well but causes hypoglycaemia |
| t04 ablation | heating time and power covering tumour + 5 mm at the highest perfusion, at or below 100 C at the lowest | Pennes bioheat equation, temperature-dependent perfusion, Arrhenius damage; observed order of accuracy | a first-order solver; a plan valid only at nominal perfusion |
| t05 debugging | what is wrong with a PK package, judged against its specification | units, solver tolerance and step control, float32 drift, dose bookkeeping | fixing the two reported bugs and missing the three silent ones |

Three design choices recur:

1. **Judge decisions at the truth.** t02's regimen and t04's plan are
   evaluated with the true parameters, not the agent's estimates, so an
   agent cannot pass by being confidently wrong.
2. **Verify the agent's numerics, not just its numbers.** t04 runs the
   agent's solver on three space and three time resolutions and measures the
   observed order of accuracy of the stored energy. A correct second-order
   scheme measures 1.98-2.17; lagging the perfusion one step measures 1.00 in
   time, and the check says which discretisation is wrong.
3. **Test closed loops in the loop.** t03 runs the agent's controller as a
   live process that receives a sensor reading every 5 simulated minutes
   (2 s deadline) through meals, announced exercise, sensor dropout, a false
   high spike and a compression low.

Thresholds come from the physics or from the written contract, never from
the reference solution. Examples: time in range ≥ 70 % and glucose never
below 70 mg/dL follow the international consensus on time in range
(Battelino et al. 2019); 100 C is where tissue boils; the 40 ms peak window
admits a peak on the QRS upstroke but never a T wave. Full rationale per
check: [DESIGN](DESIGN.md); equations and numerical methods:
[EQUATIONS](EQUATIONS.md).

## 4. Validating the graders

A grader is validated the way an instrument is calibrated: against known
inputs whose correct reading is known in advance.

**Reference probes (false rejects).** Independent correct solutions using
different methods and languages, graded on many hidden cases. Any failure is
a false reject. With no failures in n cases the 95 % upper bound on the
per-case false-reject rate is about 3/n (Hanley and Lippman-Hand 1983).

**Negative probes (false accepts).** Realistic mistakes, mostly one-edit
mutants of a reference. Each must fail with its expected code. Its per-case
acceptance rate gets a Wilson interval (Wilson 1927), and from the upper
bound p the chance of earning reward over 8 hidden cases is p^8, which must be
below 5 %.

**Boundary probes.** Defensible methods at or beyond a contract limit. They
must fail on exactly that clause and no other.

**Grader mutation testing.** Mutate the grader (switch a check off so it can
never fail, or halve and double each acceptance threshold) and run the test
suite; a surviving mutant is a grader defect the tests would not notice (Jia
and Harman 2011).

**Red team.** Seven attacks on the sandbox: escaping the workspace, symlinked
outputs, a fork bomb, a detached daemon, filling the disk, exhausting memory,
and forging t03's simulation record. Each is run with and without the
sandbox user, to show what the sandbox provides.

The campaign is a Snakemake workflow (`make validate`). Every number in
[VALIDATION_REPORT](VALIDATION_REPORT.md) is generated from a result file
carrying its seed, versions and provenance, listed in a SHA-256 manifest.

## 5. Results

| quantity | result |
|---|---|
| reference solutions | 13 (Python, Octave; finite volumes and finite elements; four t01 detectors) |
| reference cases graded / false rejects | 820 / 0 |
| tightest false-reject bound | 0.99 % per case (t03, 300 cases) |
| negative probes failing with the expected codes | 33 / 33 |
| negatives with P(reward over 8 cases) < 5 % (95 % bound) | 33 / 33 |
| negatives with well-formed output on every case | 28 / 33 (85 %; 95 % CI 69-93 %) |
| boundary probe failing on exactly its clause | 1 / 1 |
| checks whose removal the tests detect | 44 / 45 |
| acceptance-threshold mutants killed | 74 / 76 |
| red-team attacks contained (sandboxed) | 7 / 7 |

**H1 holds.** 28 of the 33 realistic mistakes write well-formed output on
every hidden case. A grader that checked only that the program ran and that
its files parse would have rewarded all 28, i.e. 85 % of realistic mistakes,
against the 30 % threshold. The other 5 are interface failures in t03
(crashing on a sensor dropout, malformed or late replies), which any grader
catches.

**H2 holds.** All 33 negatives are rejected at the reward level (at most a
5 % chance of passing 8 hidden cases, at 95 % confidence), and no correct
solution was rejected in 820 cases. The closest negative is a plausible PID
without safety layers in t03: it passes 62 % of single patients but
hypoglycaemia on the others makes its chance of passing 8 at most 4.1 %.
This is the reason for multi-case binary reward: an average-case controller
that is unsafe for a third of patients must not be rewarded.

**Margins.** For each check, the report lists the measured values of the
references and of the negatives that fail it. Most separate by orders of
magnitude: on t05 the references sit near 10^-12 against a 10^-5 tolerance,
and the failing negatives reach 1 to 10.
The tight ones are stated rather than hidden: t03's reference has a worst
patient at 72.5 % time in range against a 70 % limit, because the consensus
targets bind on difficult patients even for a fully layered controller.

**Survivors, explained.** The one check whose removal goes unnoticed is
t04's simulation-format gate, which only improves diagnostics: every later
check also fails on a malformed file. The two surviving threshold mutants
are t01's edge margin (2 s), which no probe comes near.

## 6. What validation changed

The process found problems that design review had not. The full list, with
evidence, is in [validation/FINDINGS.md](../validation/FINDINGS.md); the
most instructive:

1. **A textbook test rejected correct fits.** A Wald test on t02's
   parameters failed fits whose likelihood was *better* than the truth's:
   with 14 samples the model is weakly identified and the likelihood far
   from quadratic. It was replaced by the exact likelihood-ratio region.
2. **A plain PID was too good.** Before exercise was modelled, a PID with no
   safety layers passed about 90 % of t03 patients. Announced exercise with
   unannounced after-effects made it fail on about 40 %. The reference then
   needed three new safety layers to stay safe on every patient.
3. **The obvious physics made verification impossible.** Damage-driven
   vascular shutdown in t04 created a sub-millimetre front on which a correct
   second-order scheme measured order 1.5. Refining space and time together
   let their errors cancel (one correct case measured 0.76). And the centre
   temperature turned out to be a poor convergence measure: the independent
   finite-element reference failed on it in 2 of 8 cases while its stored
   energy converged at exactly 2. Each problem changed the grader.
4. **Without the sandbox user a controller can forge its own record.** A
   t03 controller wrote a perfect glucose trace into the grader's directory,
   left a daemon to rewrite it, and earned a passing verdict. Records are now
   written only after the controller is dead, into a directory it cannot
   write, and the command line refuses to grade as root without isolation.
5. **Mutation testing found unpinned thresholds**, and then a blind spot in
   the mutation tool itself. Boundary tests now pin every acceptance
   threshold with the contract's numbers, not the grader's constants.
6. **A reference had a bug and the grader was right.** One t01 reference
   detected spurious beats at the first and last samples of 1 recording in
   100. One extra RR interval doubled the reported variability, and the
   grader correctly rejected it.

## 7. GPU benchmark

Grading uses the 1D spherical solver: spherical symmetry makes it exact and
cheap. Clinical planning with off-centre probes and vessels is 3D, which is
where a GPU pays. `gpu/` holds a 3D version of the same physics (7-point
stencil, Heun time stepping) with three backends: NumPy, CuPy array code,
and a hand-written fused CUDA kernel. Correctness is established before any
timing. Measured on a Tesla T4 (Colab):

* The GPU backends agree with NumPy to 7×10^-15 K.
* The 3D solution converges to the 1D reference at observed orders 2.02,
  2.01 and 2.00.
* A full treatment on 257^3 nodes gives a peak of 96.08 C and a lesion
  radius of 9.206 mm, against 96.06 C and 9.204 mm in 1D.
* The fused kernel reaches 4.0 billion cell updates per second in float32,
  about 75 % of the T4's memory bandwidth, and runs the full treatment in
  12 s. CuPy array code takes 90 s.
* In float64 the fused kernel reaches only about 22 % of peak bandwidth,
  because the T4's double-precision units are slow (two exponentials and two
  divisions per node per stage). The notebook had predicted a 2× gap from
  float64 to float32; the measured gap was 6.7×. The prediction was
  corrected, not the measurement.

The 110-230× speed-ups against NumPy are against plain array code on one CPU
core, so they overstate what an optimised multi-core CPU code would lose.
The GPU-to-GPU comparisons do not depend on that baseline. Full results:
[gpu/README.md](../gpu/README.md).

## 8. Agent evaluation

`agent_eval/harness.py` runs a tool-use agent (one `bash` tool) against the
Anthropic API in a sandboxed workspace, grades the result with the real
grader and keeps the full transcript. Harbor is the faithful path when Docker
is available. Two scripted baselines check the pipeline without an API key:
an *oracle* that writes the reference solution through the bash tool (reward
1 on every task) and a *no-op* agent (reward 0 on every task, including t05,
where it submits the shipped buggy package). Results:
[results/agent_eval/SUMMARY.md](../results/agent_eval/SUMMARY.md). The
protocol for real models (private seeds, repeated attempts, pass rates with
Wilson intervals, failure taxonomy, cost control) is in
[AGENT_EVAL](AGENT_EVAL.md). Results for real models are added there as
they are run; none are claimed here.

## 9. Limitations and threats to validity

* **The wrong solutions were written by the author.** They are realistic
  mistakes (unit slips, missing safety layers, first-order schemes), but not
  sampled from real agents. Agent transcripts are the test of whether real
  agents make other mistakes the graders miss.
* **Synthetic physiology.** Patients, ECGs and tissue are generated by
  published models with realistic parameter ranges ([DATA](DATA.md)).
  Clinical data could show behaviour the models lack.
* **Compute-limited bounds.** t04 references were graded on 30 cases, so
  the false-reject bound there is 10 % per case, not 1 %.
* **A weak control in t04.** Predicting the lesion at the nominal perfusion
  is caught on 8 of 12 cases by the consistency check. It is harmless,
  because coverage and safety are judged at the extremes regardless, and it
  is reported rather than hidden by tightening a tolerance the independent
  finite-element reference needs.
* **Containers not exercised here.** The Docker images and Harbor runs are
  written but were not built in the development environment. Everything else
  (tests, grading, privilege drop, limits, red team, determinism) has been
  run, and CI runs on every push.
* **Not a medical device.** IEC 62304 and ISO 14971 informed the design
  (FMEA, hazard thinking), but nothing here is for clinical use.

## 10. Future work

* Real-agent results across several models, with failure labels from the
  transcripts, and graders tightened wherever an agent finds a gap.
* More tasks: population pharmacokinetics (mixed-effects models), Monte
  Carlo uncertainty propagation, and a physics-informed neural network
  submission the t04 grader must reject.
* An embedded variant of t03 (fixed-point arithmetic, a latency budget).
* Pass-rate drift and saturation monitoring, retiring tasks that agents
  saturate.

## 11. Reproducing

```
uv sync --frozen
make check              # lint, strict typing, full test suite
make validate           # the whole grader validation campaign -> docs/VALIDATION_REPORT.md
make agent-baselines    # scripted agents through the agent harness (no API key)
```

GPU: open `gpu/BioEng_GPU_Benchmark.ipynb` in Colab with a T4 runtime.
Every workflow: [WORKFLOWS](WORKFLOWS.md).

## References

* Amodei D, Olah C, Steinhardt J, Christiano P, Schulman J, Mané D. Concrete problems in AI safety. arXiv:1606.06565, 2016.
* Battelino T, et al. Clinical targets for continuous glucose monitoring data interpretation: recommendations from the international consensus on time in range. *Diabetes Care* 2019;42(8):1593-1603.
* Bergman RN, Ider YZ, Bowden CR, Cobelli C. Quantitative estimation of insulin sensitivity. *Am J Physiol* 1979;236(6):E667-E677.
* de Chazal P, O'Dwyer M, Reilly RB. Automatic classification of heartbeats using ECG morphology and heartbeat interval features. *IEEE Trans Biomed Eng* 2004;51(7):1196-1206.
* Hanley JA, Lippman-Hand A. If nothing goes wrong, is everything all right? Interpreting zero numerators. *JAMA* 1983;249(13):1743-1745.
* Henriques FC. Studies of thermal injury V: the predictability and the significance of thermally induced rate processes leading to irreversible epidermal injury. *Arch Pathol* 1947;43:489-502.
* Jia Y, Harman M. An analysis and survey of the development of mutation testing. *IEEE Trans Softw Eng* 2011;37(5):649-678.
* McSharry PE, Clifford GD, Tarassenko L, Smith LA. A dynamical model for generating synthetic electrocardiogram signals. *IEEE Trans Biomed Eng* 2003;50(3):289-294.
* Pan J, Tompkins WJ. A real-time QRS detection algorithm. *IEEE Trans Biomed Eng* 1985;32(3):230-236.
* Pennes HH. Analysis of tissue and arterial blood temperatures in the resting human forearm. *J Appl Physiol* 1948;1(2):93-122.
* Roache PJ. *Verification and Validation in Computational Science and Engineering.* Hermosa, 1998.
* Skalse J, Howe N, Krasheninnikov D, Krueger D. Defining and characterizing reward hacking. *NeurIPS* 2022.
* Wilson EB. Probable inference, the law of succession, and statistical inference. *J Am Stat Assoc* 1927;22(158):209-212.
