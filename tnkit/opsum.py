"""Sums of local operator products, compiled to MPOs.

`compile_mpo` returns the list of rank-4 arrays the solvers take: ``W[i]`` has shape
``(D_i, D_{i+1}, d_i, d_i)`` and index order ``(left, right, phys_out, phys_in)``, with
``D_0 = D_N = 1``. As in `tnkit.models`, channel 0 of every inner bond means "nothing placed
yet" (START) and the last channel means "term complete" (DONE).

The compiler sweeps left to right. At each cut the terms that cross it form a coefficient
matrix between left keys (an incoming channel and the operator on the site just passed) and
right strings (what remains to be placed). Each connected block of that matrix gets
channels from a minimum vertex cover of its bipartite graph, which is exact and reproduces
the hand-written builders. Where the block's numerical rank is lower than its cover, it is
replaced by its SVD, with the isometric factor kept on the left.

For large, sparse long-range quadratic forms the compile cost grows roughly as N^3
(at density ~1e-3: 54 s at N = 1500, 414 s at N = 3000).
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

from .backend import _negligible_imag
from .models import I2, SX, SY, SZ

Label = tuple[str, ...]                       # operator names on one site, multiplied in order
String = tuple[tuple[int, Label], ...]        # site-ordered local factors of one term


@dataclass(frozen=True)
class SiteType:
    """A local Hilbert space: named ``d x d`` operators (including ``"Id"``), and the names
    that are fermionic (Jordan-Wigner transformed with the ``"F"`` parity operator)."""
    ops: dict[str, np.ndarray]
    fermionic: frozenset[str] = frozenset()

    @property
    def d(self) -> int:
        return self.ops["Id"].shape[0]


def spin_half() -> SiteType:
    """Paulis ``X, Y, Z`` with ``Z = diag(1, -1)``, and ``Sp = [[0, 1], [0, 0]]``, ``Sm = Sp.T``."""
    sp = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    return SiteType({"Id": I2, "X": SX, "Y": SY, "Z": SZ, "Sp": sp, "Sm": sp.T.copy()})


def spinless_fermion() -> SiteType:
    """Basis ``(|0>, |1>)``: ``C`` annihilates, ``N = Cdag C``, ``F = 1 - 2N``."""
    c = np.array([[0, 1], [0, 0]], dtype=np.complex128)
    n = np.diag([0, 1]).astype(np.complex128)
    return SiteType({"Id": I2, "C": c, "Cdag": c.T.copy(), "N": n, "F": SZ},
                    frozenset({"C", "Cdag"}))


def qudit(d: int, ops: dict[str, np.ndarray]) -> SiteType:
    """A ``d``-level site with the given operators; ``"Id"`` is added."""
    ops = {k: np.asarray(v) for k, v in ops.items()}
    for k, v in ops.items():
        if v.shape != (d, d):
            raise ValueError(f"operator {k!r} has shape {v.shape}, expected {(d, d)}")
    return SiteType({"Id": np.eye(d, dtype=np.complex128), **ops})


class OpSum:
    """A sum of products of named local operators.

    ``os += (coef, "A", i, "B", j, ...)`` adds ``coef * A_i B_j ...``; operators multiply in
    the order written. Fermionic operators may be written at any sites in any order.
    """

    def __init__(self) -> None:
        self.terms: list[tuple[complex, tuple[tuple[str, int], ...]]] = []
        self.forms: list[tuple[np.ndarray, str, str]] = []

    def __iadd__(self, term: tuple) -> "OpSum":
        coef, *factors = term
        if not factors or len(factors) % 2:
            raise ValueError(f"expected (coef, op, site, op, site, ...), got {term!r}")
        ops, sites = factors[::2], factors[1::2]
        for s in sites:
            if isinstance(s, (str, bytes)) or int(s) != s:
                raise ValueError(f"site {s!r} is not an integer")
        self.terms.append((coef, tuple((str(o), int(s)) for o, s in zip(ops, sites))))
        return self

    def add_quadratic_form(self, J: np.ndarray, A: str, B: str) -> None:
        """Add ``sum_{i<j} J[i, j] A_i B_j`` without expanding it into terms.

        ``J`` is ``(N, N)`` and strictly upper triangular; ``A`` and ``B`` must be bosonic.
        """
        J = np.asarray(J)
        if J.ndim != 2 or J.shape[0] != J.shape[1]:
            raise ValueError(f"J must be square, got shape {J.shape}")
        if np.any(np.tril(J) != 0):
            raise ValueError("J must be strictly upper triangular (only i < j is summed)")
        self.forms.append((J, A, B))

    def add_sum_of_squares(self, A: np.ndarray, b: np.ndarray, O: str) -> None:
        """Add ``sum_n (sum_k A[n, k] O_k + b[n])^2``, expanded exactly; ``O`` must be bosonic."""
        A, b = np.asarray(A), np.asarray(b)
        G = A.T @ A
        self.add_quadratic_form(2 * np.triu(G, 1), O, O)
        lin = 2 * (b @ A)
        for k in range(A.shape[1]):
            if G[k, k] != 0:
                self += (G[k, k], O, k, O, k)
            if lin[k] != 0:
                self += (lin[k], O, k)
        self += (b @ b, "Id", 0)


class MPOWidthError(ValueError):
    """A compiled MPO bond is wider than the requested ``max_width``."""


def _normalize(opsum: OpSum, sites: list[SiteType]) -> dict[String, complex]:
    """Site-ordered, Jordan-Wigner transformed terms with equal strings merged."""
    N = len(sites)
    out: dict[String, complex] = {}
    for coef, factors in opsum.terms:
        for name, s in factors:
            if not 0 <= s < N:
                raise ValueError(f"site {s} is out of range for {N} sites")
            if name not in sites[s].ops:
                raise ValueError(f"site {s} has no operator {name!r}")
        ferm = [k for k, (name, s) in enumerate(factors) if name in sites[s].fermionic]
        swaps = sum(factors[a][1] > factors[b][1] for a, b in combinations(ferm, 2))
        local: dict[int, list[str]] = {}
        for name, s in sorted(factors, key=lambda f: f[1]):     # stable: same-site order kept
            local.setdefault(s, []).append(name)
        fsites = [factors[k][1] for k in ferm]
        for s in range(max(fsites, default=0)):
            if sum(f > s for f in fsites) % 2:                  # odd fermion count to the right
                if "F" not in sites[s].ops:
                    raise ValueError(f"site {s} needs a parity operator 'F'")
                local.setdefault(s, []).append("F")
        string = tuple((s, tuple(n for n in local[s] if n != "Id")) for s in sorted(local))
        string = tuple(f for f in string if f[1]) or ((0, ("Id",)),)
        out[string] = out.get(string, 0) + (-coef if swaps % 2 else coef)
    return {k: v for k, v in out.items() if v != 0}


def _cover(rows: list[int], adj: dict[int, list[int]]) -> tuple[list[int], list[int]]:
    """Minimum vertex cover of a bipartite graph (König): (cover rows, cover columns).

    When every row can be matched the cover is all rows, which is how hand-written
    automata assign channels.
    """
    match: dict[int, int] = {}                  # column -> row

    def augment(r0):
        """Kuhn's depth-first augmenting-path search, with an explicit stack."""
        seen: set = set()
        stack, path = [(r0, iter(adj[r0]))], []  # path[k]: column taken at stack level k
        while stack:
            for c in stack[-1][1]:
                if c not in seen:
                    seen.add(c)
                    path.append(c)
                    if c not in match:
                        for (r, _), cc in zip(stack, path):
                            match[cc] = r
                        return
                    stack.append((match[c], iter(adj[match[c]])))
                    break
            else:
                stack.pop()
                if path:
                    path.pop()

    for r in rows:
        augment(r)
    zr = set(rows) - set(match.values())        # alternating paths from unmatched rows
    zc: set = set()
    stack = list(zr)
    while stack:
        for c in adj[stack.pop()]:
            if c not in zc:
                zc.add(c)
                if match[c] not in zr:
                    zr.add(match[c])
                    stack.append(match[c])
    return [r for r in rows if r not in zr], sorted(zc)


def _components(nz_rows: np.ndarray, nz_cols: np.ndarray) -> list[tuple[list[int], list[int]]]:
    """Connected components of the bipartite graph with edges ``(nz_rows[k], nz_cols[k])``."""
    parent: dict[tuple[str, int], tuple[str, int]] = {}

    def find(x):
        while parent.setdefault(x, x) != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for r, c in zip(nz_rows.tolist(), nz_cols.tolist()):
        parent[find(("r", r))] = find(("c", c))
    groups: dict[tuple[str, int], tuple[set, set]] = {}
    for kind, i in list(parent):
        rs, cs = groups.setdefault(find((kind, i)), (set(), set()))
        (rs if kind == "r" else cs).add(i)
    return sorted((sorted(rs), sorted(cs)) for rs, cs in groups.values())


def _channels(M: np.ndarray, rank_tol: float, row_open: list[int], row_label: list[Label],
              cols: list[String]):
    """Factor the cut's coefficient matrix as ``M = A @ B``, one channel per column of ``A``.

    Returns ``(channels, structural, dropped)``: each channel is ``(a, b, sort_key)`` with
    ``a`` a column of ``A`` and ``b`` a row of ``B``, sorted by opening site, closing site
    and operator names; ``structural`` is the vertex-cover width before the SVD pass and
    ``dropped`` the largest discarded singular value.
    """
    n_rows, n_cols = M.shape
    close = [c[-1][0] for c in cols]
    chans, structural, dropped = [], 0, 0.0
    nz_r, nz_c = np.nonzero(M)
    for rs, cs in _components(nz_r, nz_c):
        adj = {r: [c for c in cs if M[r, c] != 0] for r in rs}
        crow, ccol = _cover(rs, adj)
        width = len(crow) + len(ccol)
        structural += width
        if width > 1:
            U, S, Vh = np.linalg.svd(M[np.ix_(rs, cs)], full_matrices=False)
            keep = int(np.sum(S > rank_tol * S[0]))
            if keep < width:
                dropped = max(dropped, float(S[keep]))
                first = (min(row_open[r] for r in rs), min(close[c] for c in cs), ())
                for m in range(keep):
                    a = np.zeros(n_rows, M.dtype); a[rs] = U[:, m]
                    b = np.zeros(n_cols, M.dtype); b[cs] = S[m] * Vh[m]
                    chans.append((a, b, first + (m,)))
                continue
        for r in crow:                          # channel = the left key; coefficients go right
            a = np.zeros(n_rows, M.dtype); a[r] = 1
            b = np.zeros(n_cols, M.dtype); b[cs] = M[r, cs]
            chans.append((a, b, (row_open[r], min(close[c] for c in adj[r]), row_label[r], r)))
        for c in ccol:                          # channel = the right string; coefficients go left
            lefts = [r for r in rs if M[r, c] != 0 and r not in crow]
            a = np.zeros(n_rows, M.dtype); a[lefts] = M[lefts, c]
            b = np.zeros(n_cols, M.dtype); b[c] = 1
            names = tuple(n for _, lab in cols[c] for n in lab)
            chans.append((a, b, (min(row_open[r] for r in lefts), close[c], names, n_rows + c)))
    chans.sort(key=lambda ch: ch[2])
    return chans, structural, dropped


def compile_mpo(opsum: OpSum, sites: list[SiteType], rank_tol: float = 1e-13,
                dtype: str = "complex128", max_width: int | None = None):
    """Compile ``opsum`` on ``sites`` into an MPO; returns ``(mpo, report)``.

    ``rank_tol`` is relative to the largest singular value of each block. ``dtype`` is
    ``"complex128"`` (as every `tnkit.models` builder; imaginary parts are kept however small)
    or ``"auto"``, which gives float64 when every coefficient and operator used is real to
    the real backend's tolerance. ``max_width`` raises `MPOWidthError`
    at the first bond wider than it.

    ``report`` holds, for each cut ``p = 0..N``: ``width`` (``D_p``), ``structural`` (the
    exact channel count between START and DONE), ``numerical`` (the count after the SVD
    pass, ``D_p - 2`` on inner bonds), ``dropped`` (largest discarded singular value); and
    ``dtype``.
    """
    if dtype not in ("complex128", "auto"):
        raise ValueError(f"dtype must be 'complex128' or 'auto', got {dtype!r}")
    N = len(sites)
    if N == 0:
        raise ValueError("compile_mpo needs at least one site")
    terms = _normalize(opsum, sites)
    starts: list[list[tuple[String, complex]]] = [[] for _ in range(N)]
    used = set()
    for string, c in terms.items():
        starts[string[0][0]].append((string, c))
        used.update((s, n) for s, lab in string for n in lab)
    for J, A, B in opsum.forms:
        if J.shape != (N, N):
            raise ValueError(f"quadratic form has shape {J.shape}, expected {(N, N)}")
        for s in range(N):
            for name in (A, B):
                if name not in sites[s].ops or name in sites[s].fermionic:
                    raise ValueError(f"quadratic-form operator {name!r} must be a bosonic "
                                     f"operator of site {s}")
                used.add((s, name))
    # "auto" drops negligible imaginary parts; "complex128" computes in float64 only when
    # that is exact. `used` includes the parity "F" that `_normalize` inserts.
    tol = 1e-12 if dtype == "auto" else 0.0
    real = (all(_negligible_imag(c, tol)[0] for c in terms.values())
            and all(_negligible_imag(J, tol)[0] for J, _, _ in opsum.forms)
            and all(_negligible_imag(sites[s].ops[n], tol)[0] for s, n in used))
    ftype = np.float64 if real else np.complex128

    def cast(c):
        return np.real(c) if real else c

    mats: dict[tuple[int, Label], np.ndarray] = {}

    def mat(s: int, lab: Label) -> np.ndarray:
        if (s, lab) not in mats:
            m = sites[s].ops[lab[0]]
            for n in lab[1:]:
                m = m @ sites[s].ops[n]
            mats[s, lab] = m
        return mats[s, lab]

    report = {"width": [1], "structural": [0], "numerical": [0], "dropped": [0.0]}
    mpo: list[np.ndarray] = []
    Bprev = np.zeros((0, 0), ftype)             # mid channels x right strings at the last cut
    rights: list[String] = []
    opens: list[int] = []                       # opening site of each mid channel
    ident = ("Id",)
    for s in range(N):
        # Row = left key (c_in, label), c_in 0 for START or 1.. for a mid channel of cut s;
        # column = the right string still to be placed; col -1 means the term closes here.
        rows: dict[tuple[int, Label], int] = {}
        colmap: dict[String, int] = {}
        vals: list[tuple[int, int, complex]] = []

        def add(k, rest, v):
            vals.append((rows.setdefault(k, len(rows)),
                         colmap.setdefault(rest, len(colmap)) if rest else -1, v))

        for j, r in enumerate(rights):
            lab, rest = (r[0][1], r[1:]) if r[0][0] == s else (ident, r)
            for c in np.flatnonzero(Bprev[:, j]).tolist():
                add((c + 1, lab), rest, Bprev[c, j])
        for string, c in starts[s]:
            add((0, string[0][1]), string[1:], cast(c))
        for J, A, B in opsum.forms:
            for j in np.flatnonzero(J[s]).tolist():
                add((0, (A,)), ((j, (B,)),), cast(J[s, j]))
        keys, cols = list(rows), list(colmap)

        M = np.zeros((len(keys), len(cols)), ftype)
        done = np.zeros(len(keys), ftype)
        for k, cj, v in vals:
            if cj < 0:
                done[k] += v
            else:
                M[k, cj] += v

        chans, n_struct, dropped = _channels(
            M, rank_tol, [s if c == 0 else opens[c - 1] for c, _ in keys],
            [lab for _, lab in keys], cols)

        last = s == N - 1
        D_in = 1 if s == 0 else len(opens) + 2
        D_out = 1 if last else len(chans) + 2
        if max_width is not None and D_out > max_width:
            raise MPOWidthError(f"bond {s + 1} (between sites {s} and {s + 1}) needs width "
                                f"{D_out} > max_width={max_width}")
        report["width"].append(D_out)
        report["structural"].append(0 if last else n_struct)
        report["numerical"].append(0 if last else len(chans))
        report["dropped"].append(dropped)

        d = sites[s].d
        Idm = sites[s].ops["Id"]
        W = np.zeros((D_in, D_out, d, d), np.complex128)
        if not last:
            W[0, 0] = Idm
        if s > 0:
            W[D_in - 1, D_out - 1] = Idm
        for k in np.flatnonzero(done).tolist():
            W[keys[k][0], D_out - 1] += done[k] * mat(s, keys[k][1])
        for o, (a, _, _) in enumerate(chans, start=1):
            for k in np.flatnonzero(a).tolist():
                W[keys[k][0], o] += a[k] * mat(s, keys[k][1])
        mpo.append(W)

        Bprev = np.array([b for _, b, _ in chans], ftype).reshape(len(chans), len(cols))
        rights = cols
        opens = [ch[2][0] for ch in chans]

    if dtype == "auto" and real:
        mpo = [np.ascontiguousarray(W.real) for W in mpo]
    report["dtype"] = str(mpo[0].dtype)
    return mpo, report


def to_dense(mpo: list[np.ndarray]) -> np.ndarray:
    """Contract an MPO of any local dimensions into a dense matrix (small systems only)."""
    T = np.ones((1, 1, 1), dtype=mpo[0].dtype)
    for W in mpo:
        d = W.shape[2]
        T = np.einsum("xya,abst->xsytb", T, W)
        T = T.reshape(T.shape[0] * d, T.shape[2] * d, W.shape[1])
    return T[:, :, 0]
