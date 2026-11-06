# Probe smoke: ITensor DMRG on the Heisenberg chain, dense and QN, on the CPU or on CUDA.
#   julia --project=ENV smoke.jl cpu|cuda OUT.json [N] [chi]
# Every arm is attempted and failures are recorded, not thrown; pass/fail is read from OUT.
const DEVICE = ARGS[1]
const OUT = ARGS[2]
const N = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 16
const CHI = length(ARGS) >= 4 ? parse(Int, ARGS[4]) : 64
DEVICE in ("cpu", "cuda") || error("device must be cpu or cuda, got $DEVICE")

const T0 = time_ns()
include(joinpath(@__DIR__, "common.jl"))
const rec = Dict{String, Any}("device" => DEVICE, "N" => N, "chi" => CHI,
    "load_itensors_s" => (time_ns() - T0) / 1.0e9)
BLAS.set_num_threads(parse(Int, get(ENV, "JPROBE_BLAS_THREADS", "4")))

if DEVICE == "cuda"
    include(joinpath(@__DIR__, "cuda_setup.jl"))   # loads CUDA; defines dev, sync, cuda_info!
    cuda_info!(rec)
    const RUN = rec["cuda_functional"]
else
    const dev, sync = identity, () -> nothing
    const RUN = true
end

"""On CUDA: are the MPS and MPO tensors CuArrays, and still Float64?"""
function storage_flags!(r)
    if DEVICE == "cuda" && r["status"] == "ok"
        r["on_device"] = occursin("CuArray", r["storage"]) && occursin("CuArray", r["mpo_storage"])
        r["float64"] = occursin("Float64", r["storage"])
    end
    return r
end

if RUN
    for (name, qn) in (("dense", false), ("qn", true))
        r = attempt(() -> run_dmrg(N, CHI; qn, dev, sync))
        if r["status"] == "ok"            # second call is compiled: a warm wall time
            r2 = attempt(() -> run_dmrg(N, CHI; qn, dev, sync))
            r["wall_s_warm"] = get(r2, "wall_s", nothing)
            r["energy_warm"] = get(r2, "energy", nothing)
        else                              # retry once without the noise term
            r["retry_without_noise"] = storage_flags!(
                attempt(() -> run_dmrg(N, CHI; qn, dev, sync, noise = false)))
        end
        rec[name] = storage_flags!(r)
    end
    DEVICE == "cuda" && cuda_after!(rec)
end
rec["env"] = environment()
write_json(OUT, rec)
println("JPROBE SMOKE $DEVICE written $OUT")
