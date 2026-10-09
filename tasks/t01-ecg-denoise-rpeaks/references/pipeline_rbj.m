1;
% Reference solution (Octave/MATLAB-compatible): ECG denoising, R peaks, HRV.
% Usage: octave-cli --no-gui --quiet pipeline.m <input.csv> <fs> <outdir>
% Uses only core Octave (no signal package): RBJ-cookbook biquads with
% Butterworth Q, applied forward-backward with odd-extension padding.

function [b, a] = rbj(kind, f0, fs, Q)
  w0 = 2*pi*f0/fs; c = cos(w0); al = sin(w0)/(2*Q);
  switch kind
    case 'hp',    b = [(1+c)/2, -(1+c), (1+c)/2];
    case 'lp',    b = [(1-c)/2,  (1-c), (1-c)/2];
    case 'notch', b = [1, -2*c, 1];
  end
  a = [1+al, -2*c, 1-al];
  b = b / a(1); a = a / a(1);
end

function y = zerophase(b, a, x, npad)
  xp = [2*x(1) - flipud(x(2:npad+1)); x; 2*x(end) - flipud(x(end-npad:end-1))];
  y  = filter(b, a, xp);
  y  = flipud(filter(b, a, flipud(y)));
  y  = y(npad+1:end-npad);
end

function idx = pick_peaks(s, thr, mindist)
  m = s(2:end-1);
  cand = find(m > s(1:end-2) & m >= s(3:end) & m > thr) + 1;
  [~, ord] = sort(s(cand), 'descend');
  cand = cand(ord); keep = [];
  for i = 1:numel(cand)
    if isempty(keep) || all(abs(cand(i) - keep) >= mindist)
      keep(end+1) = cand(i);
    end
  end
  idx = sort(keep(:));
end

args = argv();
infile = args{1}; fs = str2double(args{2}); outdir = args{3};
if ~exist(outdir, 'dir'), mkdir(outdir); end

D = dlmread(infile, ',', 1, 0);
t = D(:,1); x = D(:,2); npad = round(2*fs);

% --- denoise: 0.5 Hz high-pass, 60 Hz notch ---
[b, a] = rbj('hp', 0.5, fs, 1/sqrt(2));  y = zerophase(b, a, x, npad);
[b, a] = rbj('notch', 60, fs, 30);       y = zerophase(b, a, y, npad);

% --- R-peak detection (Pan-Tompkins style) ---
[b, a] = rbj('hp', 5, fs, 1/sqrt(2));    z = zerophase(b, a, y, npad);
[b, a] = rbj('lp', 15, fs, 1/sqrt(2));   z = zerophase(b, a, z, npad);
d = gradient(z);
win = round(0.150*fs);
integ = conv(d.^2, ones(win,1)/win, 'same');
srt = sort(integ); thr = 0.30 * srt(ceil(0.99*numel(srt)));
cand = pick_peaks(integ, thr, round(0.25*fs));
half = round(0.08*fs); peaks = zeros(numel(cand),1);
for i = 1:numel(cand)
  lo = max(cand(i)-half, 1); hi = min(cand(i)+half, numel(y));
  [~, k] = max(y(lo:hi)); peaks(i) = lo + k - 1;
end
peaks = unique(peaks);                      % 1-based sample numbers

% --- HRV metrics (RR in ms) ---
rr = diff(peaks)/fs*1000;
hr = 60000/mean(rr); sdnn = std(rr); rmssd = sqrt(mean(diff(rr).^2));

% --- write outputs (sample_index is 0-based to match the Python task spec) ---
fid = fopen(fullfile(outdir,'ecg_filtered.csv'),'w'); fprintf(fid,'time_s,ecg_mV\n');
fprintf(fid,'%.6f,%.6f\n',[t'; y']); fclose(fid);
fid = fopen(fullfile(outdir,'rpeaks.csv'),'w'); fprintf(fid,'sample_index,time_s\n');
fprintf(fid,'%d,%.6f\n',[(peaks-1)'; t(peaks)']); fclose(fid);
fid = fopen(fullfile(outdir,'metrics.json'),'w');
fprintf(fid,'{\n  "mean_hr_bpm": %.6f,\n  "sdnn_ms": %.6f,\n  "rmssd_ms": %.6f,\n  "n_beats": %d\n}\n', ...
        hr, sdnn, rmssd, numel(peaks));
fclose(fid);
