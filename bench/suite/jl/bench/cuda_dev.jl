# The CUDA side of run.jl and calib.jl, in its own file so that `using CUDA` is a top-level
# statement. ITensors.cu keeps Float64 (CUDA.cu would demote it to Float32).
using CUDA, cuTENSOR
CUDA.functional() || error("CUDA.jl is not functional on this device")
const CC = CUDA.CUDACore
dev(x) = ITensors.cu(x)
sync() = CUDA.synchronize()

info(f) = try
    string(f())
catch err
    "error: " * first(sprint(showerror, err), 300)
end

function cuda_env()
    d = CUDA.device()
    return Dict{String, Any}("device_name" => info(() -> CC.name(d)),
        "device_uuid" => info(() -> CC.uuid(d)), "parent_uuid" => info(() -> CC.parent_uuid(d)),
        "cuda_runtime" => info(CUDA.runtime_version), "cuda_driver" => info(CUDA.driver_version),
        "cutensor_functional" => info(cuTENSOR.functional))
end
