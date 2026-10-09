% Independent Octave implementation of the t04 solver and planner.
%   octave-cli ablation.m simulate --problem P.json --power W --heating-time S ...
%       --perfusion OMEGA --nr N --dt DT --outdir DIR
%   octave-cli ablation.m plan --problem P.json --outdir DIR
% Same mathematics as solution/ablation.py, written separately: sparse
% tridiagonal solves, fzero root finding, its own JSON output.
1;
crash_dumps_octave_core(false);

function res = simulate_ablation(cfg, power, heating, perfusion, nr, dt)
  tis = cfg.tissue; arr = cfg.arrhenius;
  k = tis.conductivity_W_per_m_K; rho_c = tis.density_kg_per_m3 * tis.specific_heat_J_per_kg_K;
  rbcb = tis.blood_volumetric_heat_capacity_J_per_m3_K; tb = cfg.body_temperature_C;
  R = cfg.domain_radius_m; sg = cfg.source_sigma_m;
  n_on = round(heating / dt); n_steps = round((heating + cfg.cooling_time_s) / dt);
  h = R / nr; f = (0:nr)' * h; rc = 0.5 * (f(1:end-1) + f(2:end));
  vol = 4 * pi / 3 * (f(2:end).^3 - f(1:end-1).^3); cap = rho_c * vol;
  g = k * 4 * pi * f(2:end-1).^2 / h; gb = k * 4 * pi * R^2 / (h / 2);
  x = f / sg; cum = erf(x) - 2 * x / sqrt(pi) .* exp(-x.^2); src = power * diff(cum);
  wfull = perfusion * rbcb * vol;
  d = zeros(nr, 1); d(1:end-1) -= g; d(2:end) -= g; d(end) -= gb;
  L = spdiags([[g; 0], d, [0; g]], [-1 0 1], nr, nr);
  collapse = @(th) 1 ./ (1 + exp((tb + th - tis.stasis_temperature_C) / tis.stasis_width_K));
  rate = @(th) arr.A_per_s * exp(-arr.Ea_J_per_mol ./ (arr.R_J_per_mol_K * (tb + th + 273.15)));
  th = zeros(nr, 1); om = zeros(nr, 1);
  peak = 0; e_on = 0; dep = 0; perf = 0; bnd = 0;
  C = spdiags(cap, 0, nr, nr);
  for n = 0:n_steps-1
    q = src * (n < n_on);
    wold = wfull .* collapse(th);
    rhs = cap .* th + 0.5 * dt * (L * th - wold .* th) + dt * q;
    th0 = th; om0 = om; w = wold;
    for it = 1:3
      wused = w;
      A = C - 0.5 * dt * L + spdiags(0.5 * dt * wused, 0, nr, nr);
      th = A \ rhs;
      om = om0 + 0.5 * dt * (rate(th0) + rate(th));
      w = wfull .* collapse(th);
    end
    dep += dt * sum(q);
    perf += 0.5 * dt * (wold' * th0 + wused' * th);
    bnd += 0.5 * dt * gb * (th0(end) + th(end));
    peak = max(peak, (9 * th(1) - th(2)) / 8);
    if n + 1 == n_on, e_on = cap' * th; end
  end
  idx = find(om >= 1, 1, 'last');
  if isempty(idx)
    radius = 0;
  elseif idx == nr
    radius = rc(end) * 1e3;
  else
    a = log(om(idx)); b = log(max(om(idx + 1), 1e-300));
    radius = (rc(idx) - a / (b - a) * h) * 1e3;
  end
  res = struct('peak', tb + peak, 'e_on', e_on, 'radius', radius, ...
               'dep', dep, 'stored', cap' * th, 'perf', perf, 'bnd', bnd);
end

args = argv();
mode = args{1};
opt = struct('power', NaN, 'heating_time', NaN, 'perfusion', NaN, 'nr', 400, 'dt', 1.0);
for i = 2:2:numel(args)
  key = strrep(args{i}(3:end), '-', '_');
  val = args{i + 1};
  if any(strcmp(key, {'problem', 'outdir'})), opt.(key) = val; else opt.(key) = str2double(val); end
end
cfg = jsondecode(fileread(opt.problem));
if ~exist(opt.outdir, 'dir'), mkdir(opt.outdir); end

if strcmp(mode, 'simulate')
  r = simulate_ablation(cfg, opt.power, opt.heating_time, opt.perfusion, opt.nr, opt.dt);
  fid = fopen(fullfile(opt.outdir, 'simulation.json'), 'w');
  fprintf(fid, ['{"t_center_peak_C": %.15g, "energy_at_heating_end_J": %.15g, ' ...
                '"ablation_radius_mm": %.15g, "energy_budget_J": {"deposited": %.15g, ' ...
                '"stored": %.15g, "perfusion": %.15g, "boundary": %.15g}}\n'], ...
          r.peak, r.e_on, r.radius, r.dep, r.stored, r.perf, r.bnd);
  fclose(fid);
else
  lo = cfg.tissue.perfusion_range_per_s(1); hi = cfg.tissue.perfusion_range_per_s(2);
  need = cfg.tumour_radius_mm + cfg.margin_mm; lim = cfg.peak_limit_C;
  fz = optimset('TolX', 1e-4);
  best = [];
  for t = cfg.heating_times_s'
    peakf = @(p) getfield(simulate_ablation(cfg, p, t, lo, 200, 2.0), 'peak') - lim;
    coverf = @(p) getfield(simulate_ablation(cfg, p, t, hi, 200, 2.0), 'radius') - need;
    p_hi = fzero(peakf, [0.1, 200], fz);
    if coverf(p_hi) < 0, continue; end
    p_lo = fzero(coverf, [0.1, p_hi], fz);
    p = min(1.01 * p_lo, 0.5 * (p_lo + p_hi));
    if isempty(best) || p * t < best(1) * best(2), best = [p, t]; end
  end
  if isempty(best), error('no feasible plan'); end
  rh = simulate_ablation(cfg, best(1), best(2), hi, 400, 1.0);
  rl = simulate_ablation(cfg, best(1), best(2), lo, 400, 1.0);
  fid = fopen(fullfile(opt.outdir, 'plan.json'), 'w');
  fprintf(fid, ['{"heating_time_s": %.15g, "power_W": %.15g, ' ...
                '"predicted_ablation_radius_mm": %.15g, "predicted_peak_C": %.15g}\n'], ...
          best(2), best(1), rh.radius, rl.peak);
  fclose(fid);
end
