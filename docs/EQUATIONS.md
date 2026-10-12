# Equations and numerical methods

Every model the graders use, with units, the numerical method, and how the
method itself is verified. Symbols match the code; file names point to the
implementation.

## t01: ECG (McSharry dynamical model)

`src/bioengbench/models/ecg.py`

State on the unit limit cycle plus the voltage coordinate z:

```
dx/dt = a x - w(t) y          a = 1 - sqrt(x^2 + y^2)
dy/dt = a y + w(t) x
dz/dt = - sum_i a_i dth_i exp(-dth_i^2 / (2 b_i^2)) - (z - z0)
```

theta = atan2(y, x), dth_i = (theta - theta_i) wrapped to (-pi, pi], one
(theta_i, a_i, b_i) per wave P, Q, R, S, T (McSharry et al. 2003, Table 1),
w(t) = 2 pi / RR(t).

**Method.** Starting on the cycle (r = 1) gives a = 0 for all time, so the
phase is the integral of w (trapezoid rule on a fine grid). The z equation
is then linear, dz/dt = -z + u(t), and is solved with the exact first-order
hold exponential integrator

```
z[n+1] = e^-h z[n] + (1 - e^-h) u[n] + (h - 1 + e^-h)/h (u[n+1] - u[n])
```

second-order accurate, run as one IIR filter (`scipy.signal.lfilter`).

**Rhythm.** RR(t) = mean RR modulated by respiratory sinus arrhythmia, Mayer
waves (about 0.1 Hz) and a smoothed AR(1) component; R events are the phase
crossings of theta_R = 0.

**Contaminants.** Baseline wander in 0.05-0.40 Hz, mains near 60 Hz
(+-0.05 Hz drift), white electrode noise.

**Verification.** `tests/unit/test_ecg_model.py`: Richardson extrapolation
against a tight-tolerance RK45 solution (observed order 2), R-event timing,
amplitude scaling.

**Grader metrics.** Band power ratio (Welch) for wander and mains;
band-limited RMSE in 1-40 Hz against the noise-free ECG; event matching
within 40 ms by an O(n + m) two-pointer sweep; HRV:

```
mean HR = 60000 / mean(RR) bpm,  SDNN = std(RR, ddof=1) ms,
RMSSD = sqrt(mean(diff(RR)^2)) ms
```

## t02 and t05: two-compartment pharmacokinetics

`src/bioengbench/models/pk.py`, `src/bioengbench/tasks/t05_pk_debug.py`

```
dA1/dt = -(k10 + k12) A1 + k21 A2 + R(t)       k10 = CL/V1, k12 = Q/V1
dA2/dt =  k12 A1 - k21 A2                       k21 = Q/V2
C = A1 / V1                                     (mg, h, L, L/h, mg/L)
```

**Method: exact.** R(t) is piecewise constant, so on each interval between
infusion start/stop times the system is linear time-invariant and

```
d/dt [x; 1] = [[K, r], [0, 0]] [x; 1]  =>  x(t + h) = E11 x(t) + E12,   E = expm(M h)
```

with no time-stepping error. Boluses (t05) are jumps A1 += dose, and
concentrations are right-continuous at bolus times.

**Half-lives.** Eigenvalues of K are -alpha < -beta < 0;
t1/2,alpha = ln 2 / alpha, t1/2,beta = ln 2 / beta.

**Steady state.** One dosing period is the affine map x -> Phi x + b,
fixed point x_ss = (I - Phi)^-1 b. Every steady-state concentration is
linear in dose, so the feasible doses for interval tau are exactly
[lo / trough_per_mg, hi / peak_per_mg]; the cheapest regimen is the lower end,
minimised over the allowed tau. No iterative optimiser is needed.

**Statistics (t02).** Log-normal proportional error, sigma known:

```
log C_obs = log C(p) + e,   e ~ N(0, sigma^2)
SSR(p) = sum (log C_obs - log C(p))^2,   -2 log L = SSR / sigma^2 + const
```

* Fit acceptance: likelihood-ratio region
  (SSR(p) - SSR(p_hat)) / sigma^2 <= chi2_4(0.999) = 18.47.
  A Wald (Fisher z) test was tried first and rejected correct fits: with 14
  samples the model is weakly identified and the likelihood is far from
  quadratic (fits with lower SSR than the truth sat at |z| up to 50).
* Standard errors of log-parameters: SE = sqrt(diag((J^T J / sigma^2)^-1)),
  J the Jacobian of log C with respect to log p (central differences).

**Verification.** `tests/unit/test_pk_model.py`: exact solution against LSODA
(rtol 1e-11) segment by segment, terminal slope = -beta, steady state as the
limit of repeated dosing, linearity, mass balance. t05 adds an independent
LSODA check of the bolus solution (`tests/tasks/test_t05.py`).

## t03: glucose-insulin (Bergman minimal model, type 1 diabetes)

`src/bioengbench/models/glucose.py`

```
dG/dt  = -p1 (G - Gb) - (X + (a(t) - 1) max(X, 0)) G - k_ex(t) G + Ra(t) / VG
dX/dt  = -p2 X + p3 (I - Ib)
dI/dt  = -n I + u(t) / VI
dGi/dt = (G - Gi) / tau_s
```

G mg/dL, X 1/min, I mU/L, u mU/min, t min; VI = 0.12 BW L, VG = 1.6 BW dL,
Ib = ub / (n VI), insulin sensitivity S_I = p3 / p2.

* Meals: Ra = 1000 CHO f s / tmax^2 exp(-s / tmax) mg/min, s = t - t_meal,
  f = 0.8 (integrates to 1000 f CHO mg).
* Exercise: during the bout k_ex = k_uptake and a = 1 + gain; afterwards
  a = 1 + gain exp(-(t - t_end) / 90 min). Exercise amplifies only insulin
  above basal (X > 0), not the effect of an insulin deficit.
* CGM: Gi + AR(1) drift (phi 0.7, innovation sd 3 mg/dL) + 3 % white noise,
  clipped to 40-400 mg/dL; scheduled faults (dropout, +60-90 spike, -40-60
  compression low).

**Method.** Classical RK4, 0.5 min steps inside each 5 min pump period, insulin
rate held constant over the period (zero-order hold, as a pump delivers).

**Verification.** `tests/unit/test_glucose_model.py`: basal equilibrium,
RK4 against LSODA (rtol 1e-11) period by period, meal integral, exercise
effect and its sign restriction, interstitial lag, CGM reproducibility. The
agent's test bench (`environment/data/simulate.py`) is checked equal to the
grader's plant to 1e-12 (`tests/tasks/test_t03.py`).

**Scoring.** Time fractions from the 0.5 min trace; TIR = fraction in
70-180 mg/dL (consensus target > 70 %), TAR250 < 5 %, no G < 70 mg/dL
(stricter than consensus: a controller under test should cause none).

## t04: Pennes bioheat with vascular collapse and Arrhenius damage

`src/bioengbench/models/bioheat.py`

```
rho c dT/dt = (1/r^2) d/dr (k r^2 dT/dr) - omega rho_b c_b s(T) (T - T_body) + Q(r, t)
s(T)        = 1 / (1 + exp((T - 60 C) / 3 K))
Q(r, t)     = P(t) / (pi^(3/2) sigma^3) exp(-(r/sigma)^2)          (integral = P)
dOmega/dt   = A exp(-Ea / (R (T + 273.15))),  A = 3.1e98 1/s, Ea = 6.28e5 J/mol
```

Symmetry at r = 0, T = T_body at r = R (60 mm). Ablation zone Omega >= 1
(63 % of cells killed; Henriques 1947 constants).

**Why a logistic perfusion law.** A damage-driven shutdown (s = exp(-Omega))
is closer to the irreversible biology, but Omega rises by decades within a
fraction of a millimetre (Ea/R is about 75 000 K). The perfusion front is then
under-resolved on any practical grid and the observed order degrades to
about 1.5, so a grader could not tell a second-order scheme from a poor one.
The logistic law keeps the physics (heated tissue stops being cooled) and the
solution smooth. This was measured, not assumed (see the validation report).

**Method (reference).** Cell-centred finite volumes on N shells, face
conductance k A / h with A = 4 pi r^2, half-cell conductance to the Dirichlet
wall, source integrated exactly per shell:

```
F(r) = erf(r/sigma) - 2 (r/sigma) / sqrt(pi) exp(-(r/sigma)^2),   P_i = P (F(r_i+1/2) - F(r_i-1/2))
```

Crank-Nicolson in time with the source averaged exactly over each step (the
power switch lies on a step boundary). Perfusion depends on T^{n+1}, so each
step is a fixed point solved by three Picard sweeps; one sweep (lagged
perfusion) is only first order in time. Omega by the trapezoidal rule.
Centre value from the even quadratic through the first two cells:
T(0) = (9 T_1 - T_2) / 8.

**Energy budget.** Deposited = stored + perfusion + boundary, each integrated
with the scheme's own quadratures, so the residual is round-off (about 1e-9
relative) at any resolution: a direct check of discrete conservation.

**Verification.** `tests/unit/test_bioheat_model.py`:

* exact conduction solution of a Gaussian source in an infinite medium,
  T(0, t) - T_body = P / (rho c pi^(3/2)) / (2 alpha) (1/sigma - 1/sqrt(sigma^2 + 4 alpha t));
* exact steady Pennes solution (Gaussian source convolved with the screened
  Green's function e^(-m r) / (4 pi k r), m = sqrt(w/k)):
  T - T_body = P / (8 pi k r) e^(m^2 sigma^2 / 4) [e^(-m r) erfc(m sigma/2 - r/sigma) - e^(m r) erfc(m sigma/2 + r/sigma)];
* observed order 2 with Picard, 1 with lagged perfusion;
* an independent P1 finite-element solver (`references/ablation_fem.py`)
  and an Octave port pass the grader;
* a 3D Cartesian solver (`gpu/bioheat3d.py`) converges to the 1D solution at
  measured order 2.0.

**Grader convergence study.** The agent's solver is rerun refining space
(200, 400, 800 cells at dt = 1 s) and time (dt = 2, 1, 0.5 s at 400 cells)
separately; observed order p = log2(|E1 - E2| / |E2 - E3|) of the stored
energy at the end of heating. Refining both together let the space and time
error terms cancel and the order scattered (0.76 in one reference case of 60);
refined separately, second-order schemes measure 1.98-2.01 and lagged
perfusion 0.98-1.00 in time. The energy is used rather than the centre
temperature because a volume integral converges smoothly for every correct
scheme, whereas a point value can carry two error terms of opposite sign: the
P1 finite-element reference's peak measured orders 0.36 and 1.49 on two of
eight cases while its energy measured 2.0 (validation campaign). The peak is
still checked for accuracy.

**Reference values.** Richardson extrapolation from (200, 2 s) and (400, 1 s):
f* = f_fine + (f_fine - f_coarse) / 3.

**Planning.** Peak temperature rises with power and the lesion radius rises
with power, so for each heating time the robust feasible powers are
[P_lo, P_hi]: P_lo from coverage (radius >= R_t + 5 mm) at the highest
perfusion, P_hi from the 100 C limit at the lowest. Energy P t is minimised
at P_lo; the optimal heating time is the one with the smallest P_lo t.
Found by bracketing root finding (Brent) on the reference solver.

## GPU benchmark: 3D bioheat

`gpu/bioheat3d.py`. Same physics on a (n+1)^3 vertex grid over [-L, L]^3,
7-point Laplacian (second order), Heun's method (explicit RK2) at 90 % of the
stability limit dt <= 2 / (12 alpha / h^2 + w / rho c). The fused kernel
moves 15 arrays per cell update (two stages). In float32 that is
bandwidth bound: about 75 % of peak on a T4. In float64 the T4's slow
double-precision exponentials and divisions dominate, at about 22 % of
peak bandwidth (`gpu/README.md`).
