# Reuse check: are the packages precompiled for this host, and how long does loading take?
#   julia --project=ENV loadcheck.jl OUT.json
# Run with JULIA_PKG_OFFLINE=true in a fresh process on an unpacked depot.
include(joinpath(@__DIR__, "json.jl"))
const OUT = ARGS[1]
const MANIFEST = Base.parsed_toml(joinpath(dirname(Base.active_project()), "Manifest.toml"))
uuid(name) = Base.UUID(MANIFEST["deps"][name][1]["uuid"])
ids = [Base.PkgId(uuid(n), n) for n in ("ITensors", "ITensorMPS", "NDTensors", "KrylovKit", "CUDA",
    "CUDACore", "cuTENSOR", "GPUCompiler")]
for e in ("NDTensorsCUDAExt", "NDTensorscuTENSORExt")      # package extensions of NDTensors
    push!(ids, Base.PkgId(Base.uuid5(uuid("NDTensors"), e), e))
end
pre = Dict{String, Any}(id.name => (try
    Base.isprecompiled(id)
catch err
    "error: " * sprint(showerror, err)
end) for id in ids)
rec = Dict{String, Any}("precompiled_before_load" => pre)
t = time_ns()
using ITensors, ITensorMPS
rec["load_itensors_s"] = (time_ns() - t) / 1.0e9
t = time_ns()
using CUDA, cuTENSOR
rec["load_cuda_s"] = (time_ns() - t) / 1.0e9
rec["cuda_ext_loaded"] = Base.get_extension(ITensors.NDTensors, :NDTensorsCUDAExt) !== nothing
rec["cutensor_ext_loaded"] = Base.get_extension(ITensors.NDTensors, :NDTensorscuTENSORExt) !== nothing
t = time_ns()
rec["cuda_functional"] = CUDA.functional()
rec["cuda_functional_s"] = (time_ns() - t) / 1.0e9
rec["cpu_name"] = Sys.CPU_NAME
rec["cpu_target"] = get(ENV, "JULIA_CPU_TARGET", "")
rec["pkg_offline"] = get(ENV, "JULIA_PKG_OFFLINE", "")
write_json(OUT, rec)
println("JPROBE LOADCHECK written $OUT")
