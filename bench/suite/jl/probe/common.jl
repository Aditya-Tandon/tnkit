# Helpers shared by the probe scripts: the Heisenberg chain, a DMRG run, a JSON writer.
using ITensors, ITensorMPS
using LinearAlgebra, Random
include(joinpath(@__DIR__, "json.jl"))

"""Open spin-1/2 Heisenberg chain H = sum S_j . S_{j+1}; QN (U(1), S^z) sites if `qn`."""
function heisenberg(N; qn::Bool)
    s = siteinds("S=1/2", N; conserve_qns = qn)
    os = OpSum()
    for j in 1:(N - 1)
        os += "Sz", j, "Sz", j + 1
        os += 0.5, "S+", j, "S-", j + 1
        os += 0.5, "S-", j, "S+", j + 1
    end
    return s, MPO(os, s)
end

"""Two-site DMRG from a random Neel-sector MPS; `dev` moves H and the MPS to the device.
Returns a Dict with the energy, sweeps, wall time (first call includes compilation) and storage."""
function run_dmrg(N, chi; qn::Bool, dev = identity, sync = () -> nothing, noise = true)
    s, H = heisenberg(N; qn)
    Random.seed!(1234)
    psi0 = random_mps(s, [isodd(j) ? "Up" : "Dn" for j in 1:N]; linkdims = 8)
    H, psi0 = dev(H), dev(psi0)
    obs = DMRGObserver(; energy_tol = 1.0e-13, minsweeps = 4)
    nz = noise ? [1.0e-6, 1.0e-7, 1.0e-8, 0.0] : [0.0]
    t0 = time_ns()
    E, psi = dmrg(H, psi0; nsweeps = 12, maxdim = chi, cutoff = 1.0e-14, noise = nz,
        observer = obs, outputlevel = 0)
    sync()
    wall = (time_ns() - t0) / 1.0e9
    st = string(typeof(ITensors.storage(psi[N ÷ 2])))
    return Dict{String, Any}("energy" => Float64(real(E)), "sweeps" => length(obs.energies),
        "max_bond" => maxlinkdim(psi), "wall_s" => wall, "noise" => noise,
        "storage" => st, "mpo_storage" => string(typeof(ITensors.storage(H[N ÷ 2]))))
end

"""Run `f()`; return its Dict with status "ok", or the error with status "error"."""
function attempt(f)
    try
        r = f()
        r["status"] = "ok"
        return r
    catch err
        msg = sprint(showerror, err)
        bt = sprint(Base.show_backtrace, catch_backtrace())
        return Dict{String, Any}("status" => "error", "error" => first(msg, 2000),
            "backtrace" => first(bt, 4000))
    end
end

"""Versions, threads, BLAS and CPU of this process, plus the resolved package versions."""
function environment()
    deps = Dict{String, Any}()
    m = joinpath(dirname(Base.active_project()), "Manifest.toml")
    if isfile(m)
        for (name, entries) in Base.parsed_toml(m)["deps"]
            v = get(entries[1], "version", nothing)
            v === nothing || (deps[name] = v)
        end
    end
    return Dict{String, Any}("julia" => string(VERSION), "cpu_name" => Sys.CPU_NAME,
        "cpu_threads" => Sys.CPU_THREADS, "julia_threads" => Threads.nthreads(),
        "blas_threads" => BLAS.get_num_threads(), "blas_config" => string(BLAS.get_config()),
        "cpu_target" => get(ENV, "JULIA_CPU_TARGET", ""), "depot" => get(ENV, "JULIA_DEPOT_PATH", ""),
        "pkg_offline" => get(ENV, "JULIA_PKG_OFFLINE", ""), "project" => Base.active_project(),
        "packages" => deps)
end
