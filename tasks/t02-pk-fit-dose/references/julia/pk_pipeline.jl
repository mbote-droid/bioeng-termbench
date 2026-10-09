# Reference solution for t02 in Julia (third language, for cross-implementation checks).
# Usage: julia --project=<this dir> pk_pipeline.jl <indir> <outdir>
#
# Same mathematics as the Python and Octave references, written independently:
# exact augmented matrix exponential for the model, and a hand-written
# Levenberg-Marquardt fit in log-parameter space (no optimisation package).
#
# Status: written but NOT executed in the build environment (no Julia available
# there); the `julia` CI job runs it against the grader.

using JSON3, LinearAlgebra, Random, DelimitedFiles

kmat(p) = [-(p[1] + p[3]) / p[2]  p[3] / p[4];  p[3] / p[2]  -p[3] / p[4]]

function stepx(K, x, dt, rate)
    dt <= 0 && return x
    M = zeros(3, 3); M[1:2, 1:2] = K; M[1, 3] = rate
    E = exp(M * dt)
    return E[1:2, 1:2] * x + E[1:2, 3]
end

function conc(p, doses, t)
    K = kmat(p)
    ev = sort(unique(vcat(0.0, t, [d[1] for d in doses], [d[1] + d[2] for d in doses])))
    vals = Dict(0.0 => 0.0); x = zeros(2)
    for i in 2:length(ev)
        mid = 0.5 * (ev[i-1] + ev[i])
        rate = sum((d[3] / d[2] for d in doses if d[1] <= mid < d[1] + d[2]); init=0.0)
        x = stepx(K, x, ev[i] - ev[i-1], rate)
        vals[ev[i]] = x[1] / p[2]
    end
    return [vals[ti] for ti in t]
end

function steady(p, dose, tau, tinf)
    K = kmat(p); r = dose / tinf
    b = stepx(K, stepx(K, zeros(2), tinf, r), tau - tinf, 0.0)
    x0 = (I - exp(K * tau)) \ b
    xe = stepx(K, x0, tinf, r)
    c = [t <= tinf ? stepx(K, x0, t, r)[1] : stepx(K, xe, t - tinf, 0.0)[1] for t in range(0, tau; length=2001)]
    return max(maximum(c), xe[1]) / p[2], x0[1] / p[2]
end

function jacobian(f, lp; h=1e-6)
    hcat([(f(lp .+ h .* e) .- f(lp .- h .* e)) ./ (2h) for e in eachcol(Matrix(1.0I, length(lp), length(lp)))]...)
end

function levenberg_marquardt(f, lp; iters=500)
    lambda = 1e-3; r = f(lp); cost = sum(abs2, r)
    for _ in 1:iters
        J = jacobian(f, lp)
        A = J' * J; g = J' * r
        step = -(A + lambda * Diagonal(diag(A) .+ 1e-12)) \ g
        trial = clamp.(lp .+ step, log.([0.1, 1, 0.1, 1]), log.([100, 500, 500, 2000]))
        rt = f(trial); ct = sum(abs2, rt)
        if ct < cost
            done = cost - ct < 1e-14 * (1 + cost)
            lp, r, cost, lambda = trial, rt, ct, lambda / 3
            done && break
        else
            lambda *= 4
            lambda > 1e12 && break
        end
    end
    return lp, cost
end

function main(indir, outdir)
    data, _ = readdlm(joinpath(indir, "observations.csv"), ',', Float64; header=true)
    tobs, cobs = data[:, 1], data[:, 2]
    prob = JSON3.read(read(joinpath(indir, "problem.json"), String))
    doses = [(Float64(d.start_h), Float64(d.duration_h), Float64(d.amount_mg)) for d in prob.doses]
    sigma = Float64(prob.error_model.sigma_log)
    f(lp) = log.(max.(conc(exp.(lp), doses, tobs), 1e-300)) .- log.(cobs)

    rng = MersenneTwister(2024)
    # A vector of start vectors (vcat of [v] and a comprehension; `[v; [...]]` would splat v).
    starts = vcat([log.([4.5, 20, 9, 32])],
                  [log.([2, 8, 2, 8]) .+ rand(rng, 4) .* log.([6, 7.5, 20, 15]) for _ in 1:12])
    best = (fill(NaN, 4), Inf)
    for s in starts
        lp, c = levenberg_marquardt(f, s)
        c < best[2] && (best = (lp, c))
    end
    lp = best[1]; p = exp.(lp)
    J = jacobian(f, lp; h=1e-5)
    se = sqrt.(abs.(diag(pinv(J' * J / sigma^2))))
    ev = sort(-real.(eigvals(kmat(p))); rev=true)

    g = prob.prediction_grid_h
    grid = round.(collect(g.start:g.step:g.stop); digits=10)
    pred = conc(p, doses, grid)

    lo, hi = prob.design_targets_mg_per_L
    tinf = Float64(prob.maintenance_infusion_h)
    bestreg = nothing
    for tau in prob.allowed_tau_h
        pk1, tr1 = steady(p, 1.0, tau, tinf)
        dmin, dmax = lo / tr1, hi / pk1
        if dmin <= dmax && (bestreg === nothing || dmin * 24 / tau < bestreg[3])
            bestreg = (Float64(tau), dmin, dmin * 24 / tau)
        end
    end
    tau, dose, _ = bestreg
    pk, tr = steady(p, dose, tau, tinf)

    mkpath(outdir)
    params = Dict("CL_L_per_h" => p[1], "V1_L" => p[2], "Q_L_per_h" => p[3], "V2_L" => p[4],
                  "se_log" => Dict("CL" => se[1], "V1" => se[2], "Q" => se[3], "V2" => se[4]),
                  "t_half_alpha_h" => log(2) / ev[1], "t_half_beta_h" => log(2) / ev[2])
    write(joinpath(outdir, "params.json"), JSON3.write(params))
    open(joinpath(outdir, "prediction.csv"), "w") do io
        println(io, "time_h,conc_mg_per_L")
        for (t, c) in zip(grid, pred)
            println(io, t, ",", c)
        end
    end
    reg = Dict("tau_h" => tau, "dose_mg" => dose, "infusion_h" => tinf,
               "predicted_peak_mg_per_L" => pk, "predicted_trough_mg_per_L" => tr)
    write(joinpath(outdir, "regimen.json"), JSON3.write(reg))
end

main(ARGS[1], ARGS[2])
