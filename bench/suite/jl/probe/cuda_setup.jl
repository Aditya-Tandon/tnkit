# CUDA side of smoke.jl, in its own file so that `using CUDA` runs as a top-level statement.
const T_CUDA = time_ns()
using CUDA, cuTENSOR
const LOAD_CUDA_S = (time_ns() - T_CUDA) / 1.0e9
const CC = CUDA.CUDACore

# ITensors.cu (from NDTensors) keeps the element type; CUDA.cu would demote Float64 to Float32.
dev(x) = ITensors.cu(x)
sync() = CUDA.synchronize()

info(f) = try
    f()
catch err
    "error: " * first(sprint(showerror, err), 300)
end

function cuda_info!(rec)
    rec["load_cuda_s"] = LOAD_CUDA_S
    rec["cuda_functional"] = CUDA.functional()
    rec["cuda_versioninfo"] = info(() -> sprint(CUDA.versioninfo))
    rec["cuda_functional"] || return rec
    d = CUDA.device()
    rec["device_name"] = info(() -> CC.name(d))
    rec["device_uuid"] = info(() -> string(CC.uuid(d)))
    rec["parent_uuid"] = info(() -> string(CC.parent_uuid(d)))
    rec["runtime_version"] = info(() -> string(CUDA.runtime_version()))
    rec["driver_version"] = info(() -> string(CUDA.driver_version()))
    rec["total_memory_gb"] = info(() -> CUDA.total_memory() / 2^30)
    rec["cutensor_functional"] = info(() -> cuTENSOR.functional())
    return rec
end

cuda_after!(rec) = (rec["pool_status"] = info(() -> sprint(CUDA.pool_status)); rec)
