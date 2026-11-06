# Optional: MPSKit DMRG2 (dense, CPU) on the Heisenberg chain.
#   julia --project=ENV/mpskit mpskit_smoke.jl OUT.json [N] [chi]
include(joinpath(@__DIR__, "json.jl"))
const OUT = ARGS[1]
const N = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 16
const CHI = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 64
rec = Dict{String, Any}("N" => N, "chi" => CHI)
t = time_ns()
using MPSKit, TensorKit
rec["load_s"] = (time_ns() - t) / 1.0e9
try
    Sz, Sp = [0.5 0.0; 0.0 -0.5], [0.0 1.0; 0.0 0.0]
    Sm = collect(Sp')
    h = zeros(2, 2, 2, 2)                       # <i1 i2| S.S |j1 j2>
    for i1 in 1:2, i2 in 1:2, j1 in 1:2, j2 in 1:2
        h[i1, i2, j1, j2] = Sz[i1, j1] * Sz[i2, j2] +
            0.5 * (Sp[i1, j1] * Sm[i2, j2] + Sm[i1, j1] * Sp[i2, j2])
    end
    P = ComplexSpace(2)
    hop = TensorMap(h, P ⊗ P ← P ⊗ P)
    H = FiniteMPOHamiltonian(fill(P, N), (i, i + 1) => hop for i in 1:(N - 1))
    psi = FiniteMPS(N, P, ComplexSpace(CHI))
    t = time_ns()
    res = find_groundstate(psi, H, DMRG2(; trunc = truncrank(CHI), maxiter = 10, tol = 1.0e-10, verbosity = 0))
    rec["wall_s"] = (time_ns() - t) / 1.0e9
    rec["energy"] = real(expectation_value(res[1], H))
    rec["status"] = "ok"
catch err
    rec["status"] = "error"
    rec["error"] = first(sprint(showerror, err), 2000)
end
write_json(OUT, rec)
println("JPROBE MPSKIT ", rec["status"])
