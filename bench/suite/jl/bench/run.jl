# One ITensor benchmark run: read the run (TOML, written by runners/itensor_runner.py), load the
# packages, warm up on a tiny problem of the same kind, time the real one, write raw outputs.
#   julia --project=ENV --threads=n run.jl RUN.toml OUT.json
# OUT is always one JSON object with "status": "ok" (and the outputs) or "error" (and
# "error"); failures are never thrown past this file.
using TOML, LinearAlgebra, Random
const P = TOML.parsefile(ARGS[1])
const OUT = ARGS[2]
const T0 = time_ns()
using ITensors, ITensorMPS
include(joinpath(@__DIR__, "..", "probe", "json.jl"))
const LOAD_S = (time_ns() - T0) / 1.0e9
const CUDA_ON = P["device"] == "cuda"
if CUDA_ON
    include(joinpath(@__DIR__, "cuda_dev.jl"))   # using CUDA, cuTENSOR; dev, sync, cuda_env
else
    dev(x) = x
    sync() = nothing
    cuda_env() = Dict{String, Any}()
end
const O = P["opts"]

function threading!(qn::Bool)
    n = P["threads"]
    if O["threading"] == "auto" && qn && !CUDA_ON     # ITensor's advice for block-sparse on CPU
        BLAS.set_num_threads(1)
        ITensors.enable_threaded_blocksparse()
    else
        BLAS.set_num_threads(n)
        ITensors.disable_threaded_blocksparse()
    end
    return Dict{String, Any}("julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(),
        "threaded_blocksparse" => ITensors.using_threaded_blocksparse())
end

"""Sites and MPO: the XXZ-type model on explicit bonds (0-based pairs), or the Ising chain
H = -sum X X - g sum Z (X = 2 Sx, Z = 2 Sz)."""
function model(p, elt)
    N = p["N"]
    if p["model"] == "tfim"
        s = siteinds("S=1/2", N)
        os = OpSum()
        for j in 1:(N - 1)
            os += -4.0, "Sx", j, "Sx", j + 1
        end
        for j in 1:N
            os += -2.0 * p["g"], "Sz", j
        end
        return s, MPO(elt, os, s), fill("Up", N)
    end
    s = siteinds("S=1/2", N; conserve_qns = p["qn"])
    os = OpSum()
    for (i, j) in p["bonds"]
        os += 0.5 * p["J"], "S+", i + 1, "S-", j + 1
        os += 0.5 * p["J"], "S-", i + 1, "S+", j + 1
        os += p["Jz"], "Sz", i + 1, "Sz", j + 1
    end
    return s, MPO(elt, os, s), p["states"]
end

"""Per-sweep energies and device-synced times; stops once `minsweeps` are done and the energy
changed by less than max(tol, rtol |E|) (tol < 0: never)."""
mutable struct SweepClock <: ITensorMPS.AbstractObserver
    tol::Float64
    rtol::Float64
    minsweeps::Int
    t::Vector{Float64}
    e::Vector{Float64}
end
ITensorMPS.measure!(::SweepClock; kwargs...) = nothing
function ITensorMPS.checkdone!(o::SweepClock; energy, sweep, kwargs...)
    sync()
    push!(o.t, time_ns() / 1.0e9)
    push!(o.e, real(energy))
    return o.tol >= 0 && sweep >= o.minsweeps && length(o.e) >= 2 &&
           abs(o.e[end] - o.e[end - 1]) < max(o.tol, o.rtol * abs(o.e[end]))
end

"Device-synced time after each TDVP step."
struct StepClock <: ITensorMPS.AbstractObserver
    t::Vector{Float64}
end
ITensorMPS.measure!(o::StepClock; kwargs...) = (sync(); push!(o.t, time_ns() / 1.0e9); nothing)

function run_dmrg(p)
    s, H, st = model(p, Float64)
    H = dev(H)
    obs = SweepClock(p["tol"], p["rtol"], p["minsweeps"], Float64[], Float64[])
    Random.seed!(1234)
    sync()
    t0 = time_ns() / 1.0e9
    psi0 = dev(random_mps(s, st; linkdims = p["maxdims"][1]))
    E, psi = dmrg(H, psi0; nsweeps = length(p["maxdims"]), maxdim = p["maxdims"],
        cutoff = O["cutoff"], noise = O["noise"], eigsolve_krylovdim = O["eigsolve_krylovdim"],
        eigsolve_maxiter = O["eigsolve_maxiter"], observer = obs, outputlevel = 0)
    sync()
    wall = time_ns() / 1.0e9 - t0
    return Dict{String, Any}("wall_s" => wall, "energy" => Float64(real(E)),
        "energies" => obs.e, "sweeps" => length(obs.e), "sweep_s" => diff([t0; obs.t]),
        "max_bond" => maxlinkdim(psi), "dtype" => "float64",
        "storage" => string(typeof(ITensors.storage(psi[length(psi) ÷ 2]))))
end

function run_excited(p)
    s, H, st = model(p, Float64)
    H = dev(H)
    states, energies, sweeps = MPS[], Float64[], Int[]
    sync()
    t0 = time_ns() / 1.0e9
    for k in 1:p["nstates"]
        Random.seed!(1000 + k)
        psi0 = dev(random_mps(s, st; linkdims = min(p["chi"], 8)))
        obs = SweepClock(p["tol"], p["rtol"], 2, Float64[], Float64[])
        kw = (; nsweeps = p["nsweeps"], maxdim = p["chi"], cutoff = O["cutoff"],
            noise = O["noise"], eigsolve_krylovdim = O["eigsolve_krylovdim"],
            eigsolve_maxiter = O["eigsolve_maxiter"], observer = obs, outputlevel = 0)
        _, psi = k == 1 ? dmrg(H, psi0; kw...) : dmrg(H, states, psi0; weight = O["weight"], kw...)
        push!(states, psi)
        push!(energies, real(inner(psi', H, psi)))
        push!(sweeps, length(obs.e))
    end
    sync()
    return Dict{String, Any}("wall_s" => time_ns() / 1.0e9 - t0, "energies_k" => energies,
        "sweeps_k" => sweeps, "dtype" => "float64")
end

function run_tdvp(p)
    s, H, st = model(p, ComplexF64)
    psi = MPS(ComplexF64, s, st)
    H, psi = dev(H), dev(psi)
    n, dt = p["steps"], p["dt"]
    clock = StepClock(Float64[])
    sync()
    t0 = time_ns() / 1.0e9
    psi = tdvp(H, -im * dt * n, psi; nsteps = n, maxdim = p["chi"], cutoff = p["cutoff_tdvp"],
        nsite = 2, normalize = false, outputlevel = 0, (step_observer!) = clock)
    sync()
    wall = time_ns() / 1.0e9 - t0
    host = ITensors.cpu(psi)
    return Dict{String, Any}("wall_s" => wall, "steps" => n, "step_s" => diff([t0; clock.t]),
        "sz" => Float64.(real.(expect(host, "Sz"))), "bonds" => linkdims(host),
        "dtype" => "complex128")
end

const KIND = Dict("dmrg" => run_dmrg, "excited" => run_excited, "tdvp" => run_tdvp)

"""A tiny run of the same kind on the same device and tensor type (compilation, handles)."""
function warmup(p)
    q = Dict{String, Any}(p)
    q["N"] = 8
    q["chi"] = 8
    if haskey(p, "bonds")
        q["bonds"] = [[i, i + 1] for i in 0:6]
        q["states"] = p["states"][1:8]
    end
    p["kind"] == "dmrg" && (q["maxdims"] = [8, 8]; q["tol"] = -1.0; q["minsweeps"] = 2)
    p["kind"] == "excited" && (q["nstates"] = 2; q["nsweeps"] = 2)
    p["kind"] == "tdvp" && (q["steps"] = 2)
    return KIND[p["kind"]](q)
end

function environment(threads)
    deps = Dict{String, Any}()
    m = joinpath(dirname(Base.active_project()), "Manifest.toml")
    for (name, entries) in Base.parsed_toml(m)["deps"]
        v = get(entries[1], "version", nothing)
        v === nothing || (deps[name] = v)
    end
    return merge(Dict{String, Any}("julia" => string(VERSION), "cpu_name" => Sys.CPU_NAME,
        "blas_config" => string(BLAS.get_config()), "packages" => deps,
        "cpu_target" => get(ENV, "JULIA_CPU_TARGET", ""), "load_s" => LOAD_S), threads,
        cuda_env())
end

"""The run's record. Built inside a function: in a script, assignments inside a top-level
`try` make new locals, so a global record would be written out empty."""
function main()
    try
        threads = threading!(P["qn"])
        t = time_ns()
        warmup(P)
        warm = (time_ns() - t) / 1.0e9
        rec = KIND[P["kind"]](P)
        rec["warmup_s"] = warm
        rec["julia_env"] = environment(threads)
        rec["peak_rss_mb_julia"] = Sys.maxrss() / 2^20
        rec["settings"] = Dict{String, Any}("opts" => O, "device" => P["device"], "qn" => P["qn"],
            "dmrg" => "dmrg(H, psi0; maxdim per sweep, cutoff, noise, eigsolve_krylovdim, observer)",
            "tdvp" => "tdvp(H, -im*dt*n, psi; nsteps=n, nsite=2, maxdim, cutoff, normalize=false)",
            "init" => "random_mps(sites, Neel or all-up; linkdims = first maxdim), seed 1234")
        rec["status"] = "ok"
        return rec
    catch err
        return Dict{String, Any}("status" => "error",
            "error" => first(sprint(showerror, err), 2000) * "\n" *
                       first(sprint(Base.show_backtrace, catch_backtrace()), 3000))
    end
end

write_json(OUT, main())
