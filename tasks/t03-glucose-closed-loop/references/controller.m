% Independent Octave implementation of the t03 reference controller
% (same design as solution/controller.py: Kalman filter, artefact rejection,
% PID with insulin-on-board limit, predictive suspend, exercise handling).
% Run: octave-cli --no-gui --quiet controller.m   (JSON lines on stdin/stdout)
1;
crash_dumps_octave_core(false);
sighup_dumps_octave_core(false);
sigterm_dumps_octave_core(false);

DT = 5.0; TARGET = 115.0; EX_TARGET = 160.0; POST_TARGET = 140.0; POST_MIN = 180.0;
F = [1 DT; 0 1]; Q = diag([DT^3 / 3, DT]) * 0.02;

x = []; P = diag([400 4]); integ = 0; iob = 0; rej = 0; since_ex = Inf; started = false;

while true
  % fgetl(stdin) waits for a full buffer on a pipe and deadlocks the loop;
  % input() returns as soon as a line arrives. It raises at end of input.
  try
    line = input('', 's');
  catch
    break;
  end
  if isempty(strtrim(line)), continue; end
  msg = jsondecode(line);
  if ~started
    b = msg.basal_mU_per_min; umax = msg.pump_max_mU_per_min; kp0 = b / 50; started = true;
  end
  m = msg.cgm_mg_dL;  % [] on dropout
  exercise = isfield(msg, 'exercise_announced') && logical(msg.exercise_announced);

  if isempty(x) && isempty(m)
    u = b;
  else
    if isempty(x), x = [m; 0]; P = diag([400 4]); rej = 0; end
    x = F * x; P = F * P * F' + Q;
    if ~isempty(m)
      R = (5 + 0.04 * x(1))^2; S = P(1,1) + R; inn = m - x(1);
      if abs(inn) < max(25, 4 * sqrt(S))
        K = P(:,1) / S; x = x + K * inn; P = P - K * P(1,:); rej = 0;
      else
        rej = rej + 1;
        if rej >= 4, x = [m; 0]; P = diag([400 4]); rej = 0; end
      end
    end
    g = x(1); dg = x(2);
    if exercise, since_ex = 0; else since_ex = since_ex + DT; end
    recovering = since_ex < POST_MIN;
    if exercise, target = EX_TARGET; elseif recovering, target = POST_TARGET; else target = TARGET; end
    if recovering, gain = 0.5; else gain = 1.0; end
    kp = kp0 * gain; e = g - target;
    integ = min(max(integ + kp * e * DT / 300, -b), 2 * b);
    u = b + kp * e + integ + kp * 30 * dg;
    if iob > max(0, g - 100) * b * 1.2 * gain, u = min(u, b); end
    if g + 30 * dg < 90 || g < 85, u = 0; integ = min(integ, 0); end
    if rej > 0 || isempty(m), u = min(u, b); end
    if exercise, u = min(u, 0.5 * b); end
    u = min(max(u, 0), umax);
    iob = iob * exp(-DT / 60) + (u - b) * DT;
  end
  printf('{"insulin_mU_per_min": %.10g}\n', u);
  fflush(stdout);
end
