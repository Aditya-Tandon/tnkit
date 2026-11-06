# BLAS and device calibration in the Julia environment: dgemm (n 4096) and SVDs (n 1024) on the
# CPU at the pod's thread count, and on CUDA if asked.  julia --project=ENV calib.jl cpu|cuda OUT.json
using LinearAlgebra
const DEVICE, OUT = ARGS[1], ARGS[2]
using ITensors                                   # for ITensors.cu in cuda_dev.jl
include(joinpath(@__DIR__, "..", "probe", "json.jl"))
DEVICE == "cuda" && include(joinpath(@__DIR__, "cuda_dev.jl"))

function rate(f, flops; reps = 3, sync = () -> nothing)
    f(); sync()
    ts = Float64[]
    for _ in 1:reps
        t = time_ns(); f(); sync(); push!(ts, (time_ns() - t) / 1.0e9)
    end
    m = sort(ts)[(reps + 1) ÷ 2]
    return Dict{String, Any}("median_s" => m, "tflops" => flops / m / 1.0e12)
end

rec = Dict{String, Any}("device" => DEVICE, "blas_threads" => BLAS.get_num_threads(),
    "blas_config" => string(BLAS.get_config()), "julia" => string(VERSION))
n, m = 4096, 1024
if DEVICE == "cuda"
    A, B, Z = CUDA.rand(Float64, n, n), CUDA.rand(Float64, n, n), CUDA.rand(ComplexF64, m, m)
    S = CUDA.rand(Float64, m, m)
    rec["dgemm_4096"] = rate(() -> A * B, 2.0 * n^3; sync = CUDA.synchronize)
    rec["dsvd_1024"] = rate(() -> svd(S), 22.0 * m^3; sync = CUDA.synchronize)
    rec["zsvd_1024"] = rate(() -> svd(Z), 88.0 * m^3; sync = CUDA.synchronize)
    merge!(rec, cuda_env())
else
    A, B, Z, S = rand(n, n), rand(n, n), rand(ComplexF64, m, m), rand(m, m)
    rec["dgemm_4096"] = rate(() -> A * B, 2.0 * n^3)
    rec["dsvd_1024"] = rate(() -> svd(S), 22.0 * m^3)
    rec["zsvd_1024"] = rate(() -> svd(Z), 88.0 * m^3)
end
write_json(OUT, rec)
println("calib julia $DEVICE done")
