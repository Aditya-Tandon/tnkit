# GPU load for the device-monitoring check, plus CUDA.jl GEMM and SVD rates (float64, complex128).
#   julia --project=ENV burn.jl STARTED_MARKER SECONDS OUT.json
# Touches STARTED_MARKER once the device is busy, then multiplies for SECONDS.
include(joinpath(@__DIR__, "json.jl"))
using CUDA, LinearAlgebra
const MARK, SECS, OUT = ARGS[1], parse(Float64, ARGS[2]), ARGS[3]

function rate(f, flops; reps = 5)
    f(); CUDA.synchronize()                       # warm-up (compilation, handles)
    ts = Float64[]
    for _ in 1:reps
        t = time_ns(); f(); CUDA.synchronize(); push!(ts, (time_ns() - t) / 1.0e9)
    end
    return Dict{String, Any}("median_s" => sort(ts)[(reps + 1) ÷ 2], "tflops" => flops / sort(ts)[(reps + 1) ÷ 2] / 1.0e12)
end

rec = Dict{String, Any}()
n, m = 4096, 1024
A, B = CuArray(rand(n, n)), CuArray(rand(n, n))
Z, Y = CuArray(rand(ComplexF64, n, n)), CuArray(rand(ComplexF64, n, n))
S, SZ = CuArray(rand(m, m)), CuArray(rand(ComplexF64, m, m))
rec["dgemm_4096"] = rate(() -> A * B, 2.0 * n^3)
rec["zgemm_4096"] = rate(() -> Z * Y, 8.0 * n^3)
rec["dsvd_1024"] = rate(() -> svd(S), 22.0 * m^3; reps = 3)    # nominal SVD count, for comparison only
rec["zsvd_1024"] = rate(() -> svd(SZ), 88.0 * m^3; reps = 3)
touch(MARK)
t0, k = time(), 0
while time() - t0 < SECS
    A * B; CUDA.synchronize(); global k += 1
end
rec["burn_gemms"], rec["burn_s"] = k, time() - t0
rec["runtime_version"] = string(CUDA.runtime_version())
write_json(OUT, rec)
println("JPROBE BURN done $k GEMMs")
