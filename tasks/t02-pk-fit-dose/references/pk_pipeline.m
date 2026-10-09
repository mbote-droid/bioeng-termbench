1;
% Reference solution for t02 in GNU Octave (core functions only).
% Usage: octave-cli --no-gui --quiet pk_pipeline.m <indir> <outdir>
%
% Independent of the Python reference in method: the fit uses Nelder-Mead
% (fminsearch) on the log-likelihood instead of trust-region least squares,
% with a coarse-to-fine multi-start; the model is still solved exactly with
% expm on the augmented system.

function K = kmat(p)
  K = [-(p(1) + p(3)) / p(2), p(3) / p(4); p(3) / p(2), -p(3) / p(4)];
end

function x = stepx(K, x, dt, rate)
  if dt <= 0, return; end
  M = zeros(3); M(1:2, 1:2) = K; M(1, 3) = rate;
  E = expm(M * dt);
  x = E(1:2, 1:2) * x + E(1:2, 3);
end

function c = conc(p, doses, t)
  K = kmat(p);
  ev = unique([0; t(:); doses(:, 1); doses(:, 1) + doses(:, 2)]);
  x = [0; 0]; vals = zeros(size(ev));
  for i = 2:numel(ev)
    mid = 0.5 * (ev(i - 1) + ev(i));
    on = doses(:, 1) <= mid & mid < doses(:, 1) + doses(:, 2);
    rate = sum(doses(on, 3) ./ doses(on, 2));
    x = stepx(K, x, ev(i) - ev(i - 1), rate);
    vals(i) = x(1) / p(2);
  end
  [~, idx] = ismember(round(t(:) * 1e9), round(ev * 1e9));
  c = vals(idx);
end

function [peak, trough] = steady(p, dose, tau, tinf)
  K = kmat(p); r = dose / tinf;
  b = stepx(K, stepx(K, [0; 0], tinf, r), tau - tinf, 0);
  x0 = (eye(2) - expm(K * tau)) \ b;
  xe = stepx(K, x0, tinf, r);
  ts = linspace(0, tau, 2001); c = zeros(size(ts));
  for i = 1:numel(ts)
    if ts(i) <= tinf, xx = stepx(K, x0, ts(i), r); else, xx = stepx(K, xe, ts(i) - tinf, 0); end
    c(i) = xx(1);
  end
  peak = max(max(c), xe(1)) / p(2); trough = x0(1) / p(2);
end

args = argv(); indir = args{1}; outdir = args{2};
if ~exist(outdir, 'dir'), mkdir(outdir); end
D = dlmread(fullfile(indir, 'observations.csv'), ',', 1, 0);
tobs = D(:, 1); cobs = D(:, 2);
prob = jsondecode(fileread(fullfile(indir, 'problem.json')));
ds = prob.doses; doses = [[ds.start_h]', [ds.duration_h]', [ds.amount_mg]'];
sigma = prob.error_model.sigma_log;

ssr = @(lp) sum((log(max(conc(exp(lp), doses, tobs), 1e-300)) - log(cobs)).^2);
opts = optimset('TolX', 1e-10, 'TolFun', 1e-12, 'MaxFunEvals', 20000, 'MaxIter', 20000);
rand('seed', 7);
starts = [log([4.5 20 9 32]); log([3 15 5 20]); log([7 30 15 50]); log([5 12 20 40]); ...
          log([4 25 6 25]); log([6 18 12 45])];
best = inf;
for i = 1:rows(starts)
  [lp, f] = fminsearch(ssr, starts(i, :)', opts);
  [lp, f] = fminsearch(ssr, lp, opts);   % restart polishes Nelder-Mead stalls
  if f < best, best = f; lpb = lp; end
end
p = exp(lpb);

h = 1e-5; J = zeros(numel(tobs), 4);
for j = 1:4
  e = zeros(4, 1); e(j) = h;
  J(:, j) = (log(conc(exp(lpb + e), doses, tobs)) - log(conc(exp(lpb - e), doses, tobs))) / (2 * h);
end
se = sqrt(abs(diag(pinv(J' * J / sigma^2))));
ev = sort(-real(eig(kmat(p))), 'descend');

g = prob.prediction_grid_h;
grid = round((g.start:g.step:g.stop)' * 1e10) / 1e10;
pred = conc(p, doses, grid);

lo = prob.design_targets_mg_per_L(1); hi = prob.design_targets_mg_per_L(2);
tinf = prob.maintenance_infusion_h; bestd = inf;
for tau = prob.allowed_tau_h(:)'
  [pk1, tr1] = steady(p, 1, tau, tinf);
  dmin = lo / tr1; dmax = hi / pk1;
  if dmin <= dmax && dmin * 24 / tau < bestd
    bestd = dmin * 24 / tau; btau = tau; bdose = dmin;
  end
end
[pk, tr] = steady(p, bdose, btau, tinf);

P = struct('CL_L_per_h', p(1), 'V1_L', p(2), 'Q_L_per_h', p(3), 'V2_L', p(4), ...
           'se_log', struct('CL', se(1), 'V1', se(2), 'Q', se(3), 'V2', se(4)), ...
           't_half_alpha_h', log(2) / ev(1), 't_half_beta_h', log(2) / ev(2));
fid = fopen(fullfile(outdir, 'params.json'), 'w'); fputs(fid, jsonencode(P)); fclose(fid);
fid = fopen(fullfile(outdir, 'prediction.csv'), 'w'); fprintf(fid, 'time_h,conc_mg_per_L\n');
fprintf(fid, '%.10g,%.10g\n', [grid'; pred']); fclose(fid);
R = struct('tau_h', btau, 'dose_mg', bdose, 'infusion_h', tinf, ...
           'predicted_peak_mg_per_L', pk, 'predicted_trough_mg_per_L', tr);
fid = fopen(fullfile(outdir, 'regimen.json'), 'w'); fputs(fid, jsonencode(R)); fclose(fid);
