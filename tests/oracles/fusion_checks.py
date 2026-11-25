"""Consistency checks for any `Symmetry`, multiplicity included. Each returns the largest
violation over the given labels (0 when exact), or, for the integer fusion rules, the list of
identities that fail. Coefficients are read through `cg_tensor`, `f_tensor` and `r_matrix`, so
unique, simple and generic fusion go through the same code.

F and R follow `tnkit.symmetry`: F(a, b, c, d, e, f)[mu, nu, kappa, lam] maps the tree
((a b)_e^mu c)_d^nu onto (a (b c)_f^kappa)_d^lam, and R(a, b, c)[mu, nu] the vertex (a b -> c)^mu
onto (b a -> c)^nu with the factors swapped."""
import itertools

import numpy as np

from tnkit.symmetry import cg_tensor, cup, f_tensor, r_matrix

from . import groups


def outs(sym, a, b):
    return [c for c, _ in sym.fusions(a, b)]


def _max(err, x):
    return max(err, float(np.abs(x).max(initial=0.0)))


def fusion_rule_errors(sym, labels):
    """Dimensions add up; N_ab^c = N_ba^c = N_(b* a*)^(c*); associativity of N; the unit; duals."""
    N, d, du, one = sym.multiplicity, sym.dim, sym.dual, sym.unit()
    bad = []
    for a in labels:
        if du(du(a)) != a or d(du(a)) != d(a):
            bad.append(f"dual of {a}")
        if sym.fusions(one, a) != ((a, 1),) or sym.fusions(a, one) != ((a, 1),):
            bad.append(f"unit x {a}")
    for a, b in itertools.product(labels, repeat=2):
        if sum(d(c) * n for c, n in sym.fusions(a, b)) != d(a) * d(b):
            bad.append(f"dimensions of {a} x {b}")
        if N(a, b, one) != (b == du(a)):
            bad.append(f"{a} x {b} -> unit")
        for c in set(outs(sym, a, b)) | set(outs(sym, b, a)):
            if not N(a, b, c) == N(b, a, c) == N(du(b), du(a), du(c)):
                bad.append(f"N({a}, {b}, {c}) symmetry")
    for a, b, c in itertools.product(labels, repeat=3):
        ds = {x for e in outs(sym, a, b) for x in outs(sym, e, c)}
        for x in ds:
            left = sum(n * N(e, c, x) for e, n in sym.fusions(a, b))
            right = sum(n * N(a, f, x) for f, n in sym.fusions(b, c))
            if left != right:
                bad.append(f"associativity {a} {b} {c} -> {x}")
    return bad


def cg_unitarity_error(sym, labels):
    """Shapes (dim a, dim b, dim c, N); the coupling matrix of a x b (columns c, mu, z) unitary."""
    err = 0.0
    for a, b in itertools.product(labels, repeat=2):
        cols = []
        for c in outs(sym, a, b):
            x = cg_tensor(sym, a, b, c)
            if x.shape != (sym.dim(a), sym.dim(b), sym.dim(c), sym.multiplicity(a, b, c)):
                return np.inf
            cols.append(x.transpose(0, 1, 3, 2).reshape(sym.dim(a) * sym.dim(b), -1))
        C = np.concatenate(cols, axis=1)
        eye = np.eye(C.shape[0])
        err = _max(_max(err, C.conj().T @ C - eye), C @ C.conj().T - eye)
    return err


def intertwiner_error(sym, labels, gs):
    """max |(D_a x D_b) X - X D_c| over the group elements gs: the CG arrays are equivariant."""
    err = 0.0
    for a, b in itertools.product(labels, repeat=2):
        for c in outs(sym, a, b):
            x = cg_tensor(sym, a, b, c)
            for g in gs:
                lhs = np.einsum("xX,yY,XYzm->xyzm", groups.rep(sym, a, g), groups.rep(sym, b, g), x)
                err = _max(err, lhs - np.einsum("xyZm,Zz->xyzm", x, groups.rep(sym, c, g)))
    return err


def _trees(sym, a, b, c):
    """{(e, d): ((a b)_e c)_d} and {(f, d): (a (b c)_f)_d} as dense arrays (x, y, z, w, mu, nu)."""
    X = lambda *x: cg_tensor(sym, *x)                              # noqa: E731
    left = {(e, d): np.einsum("xyeu,ezwv->xyzwuv", X(a, b, e), X(e, c, d))
            for e in outs(sym, a, b) for d in outs(sym, e, c)}
    right = {(f, d): np.einsum("yzfk,xfwl->xyzwkl", X(b, c, f), X(a, f, d))
             for f in outs(sym, b, c) for d in outs(sym, a, f)}
    return left, right


def f_against_cg_error(sym, labels, all_m=False):
    """max |F - <(a (b c)_f)_d | ((a b)_e c)_d>|, at the first state of d, or between every pair
    of states (which must give F times the identity) when `all_m`."""
    err = 0.0
    for a, b, c in itertools.product(labels, repeat=3):
        left, right = _trees(sym, a, b, c)
        for (e, d), L in left.items():
            for f in outs(sym, b, c):
                F = f_tensor(sym, a, b, c, d, e, f)
                if (f, d) not in right:
                    err = _max(err, F)
                    continue
                Rt = right[(f, d)]
                if F.shape != L.shape[4:] + Rt.shape[4:]:
                    return np.inf
                if all_m:
                    got = np.einsum("xyzwkl,xyzWuv->wWuvkl", Rt.conj(), L)
                    want = np.einsum("wW,uvkl->wWuvkl", np.eye(sym.dim(d)), F)
                else:
                    got = np.einsum("xyzkl,xyzuv->uvkl", Rt[:, :, :, 0].conj(), L[:, :, :, 0])
                    want = F
                err = _max(err, got - want)
    return err


def r_against_cg_error(sym, labels):
    """max |cg(a, b, c)[x, y, :, mu] - sum_nu R[mu, nu] cg(b, a, c)[y, x, :, nu]|."""
    err = 0.0
    for a, b in itertools.product(labels, repeat=2):
        for c in outs(sym, a, b):
            R = r_matrix(sym, a, b, c)
            swapped = cg_tensor(sym, b, a, c).transpose(1, 0, 2, 3)
            err = _max(err, cg_tensor(sym, a, b, c) - np.einsum("xyzn,mn->xyzm", swapped, R))
    return err


def r_symmetric_error(sym, labels):
    """Bosonic exchange: R(a, b, c) R(b, a, c) = 1."""
    err = 0.0
    for a, b in itertools.product(labels, repeat=2):
        for c in outs(sym, a, b):
            M = r_matrix(sym, a, b, c) @ r_matrix(sym, b, a, c)
            err = _max(err, M - np.eye(len(M)))
    return err


def f_unitarity_error(sym, labels):
    """For fixed a, b, c, d: rows (e, mu, nu), columns (f, kappa, lam) form a unitary matrix."""
    err = 0.0
    for a, b, c in itertools.product(labels, repeat=3):
        for d in {d for e in outs(sym, a, b) for d in outs(sym, e, c)}:
            es, fs = outs(sym, a, b), outs(sym, b, c)
            blocks = [[f_tensor(sym, a, b, c, d, e, f) for f in fs] for e in es]
            M = np.block([[x.reshape(x.shape[0] * x.shape[1], x.shape[2] * x.shape[3])
                           for x in row] for row in blocks])
            if M.shape[0] != M.shape[1]:
                return np.inf
            err = _max(err, M @ M.conj().T - np.eye(len(M)))
    return err


def pentagon_error(sym, labels):
    """(((a b)_f c)_g d)_e -> (a (b (c d)_h)_k)_e by two moves or by three, vertex labels
    included: sum_n F(f,c,d,e,g,h)[b,g,d,n] F(a,b,h,e,f,k)[a,n,e,z] = sum_l sum_{k l m}
    F(a,b,c,g,f,l)[a,b,k,l] F(a,l,d,e,g,k)[l,g,m,z] F(b,c,d,k,l,h)[k,m,d,e]."""
    err = 0.0
    F = lambda *x: f_tensor(sym, *x)                              # noqa: E731
    for a, b, c, d in itertools.product(labels, repeat=4):
        for f in outs(sym, a, b):
            for g in outs(sym, f, c):
                for e in outs(sym, g, d):
                    for h in outs(sym, c, d):
                        for k in outs(sym, b, h):
                            if not sym.multiplicity(a, k, e):
                                continue
                            lhs = np.einsum("bgdn,anez->abgdez", F(f, c, d, e, g, h),
                                            F(a, b, h, e, f, k))
                            rhs = sum(np.einsum("abKL,LgMz,KMde->abgdez", F(a, b, c, g, f, l),
                                                F(a, l, d, e, g, k), F(b, c, d, k, l, h))
                                      for l in outs(sym, b, c))
                            err = _max(err, lhs - rhs)
    return err


def hexagon_error(sym, labels):
    """Both hexagons, vertex labels included: a moved past (b c), F R F = R F R; and c moved past
    (a b), R F^dag = F R F^dag R."""
    err = 0.0
    F = lambda *x: f_tensor(sym, *x)                              # noqa: E731
    R = lambda *x: r_matrix(sym, *x)                              # noqa: E731
    for a, b, c in itertools.product(labels, repeat=3):
        for e in outs(sym, a, b):
            for d in outs(sym, e, c):
                for g in outs(sym, c, a):
                    if not sym.multiplicity(b, g, d):
                        continue
                    lhs = sum(np.einsum("uvpq,qr,prkl->uvkl", F(a, b, c, d, e, f), R(a, f, d),
                                        F(b, c, a, d, f, g)) for f in outs(sym, b, c))
                    rhs = np.einsum("us,svtl,tk->uvkl", R(a, b, e), F(b, a, c, d, e, g),
                                    R(a, c, g))
                    err = _max(err, lhs - rhs)
                for g in outs(sym, c, a):
                    if not sym.multiplicity(g, b, d):
                        continue
                    lhs = np.einsum("vs,klus->uvkl", R(e, c, d), F(c, a, b, d, g, e).conj())
                    rhs = sum(np.einsum("uvpq,pr,tlrq,tk->uvkl", F(a, b, c, d, e, f), R(b, c, f),
                                        F(a, c, b, d, g, f).conj(), R(a, c, g))
                              for f in outs(sym, b, c))
                    err = _max(err, lhs - rhs)
    return err


def unit_error(sym, labels):
    """The unit fuses trivially: cg(1, a, a) and cg(a, 1, a) are identities, and F with a unit
    leaf is the identity on the remaining vertex."""
    one, err = sym.unit(), 0.0
    for a in labels:
        eye = np.eye(sym.dim(a))
        err = _max(err, cg_tensor(sym, one, a, a)[0, :, :, 0] - eye)
        err = _max(err, cg_tensor(sym, a, one, a)[:, 0, :, 0] - eye)
    for b, c in itertools.product(labels, repeat=2):
        for d in outs(sym, b, c):
            eye = np.eye(sym.multiplicity(b, c, d))
            err = _max(err, f_tensor(sym, one, b, c, d, b, d)[0, :, :, 0] - eye)
            err = _max(err, f_tensor(sym, b, c, one, d, d, c)[:, 0, 0, :] - eye)
            err = _max(err, f_tensor(sym, b, one, c, d, b, c)[0, :, 0, :] - eye)
    return err


def dual_error(sym, labels):
    """Cups are unitary; kappa_a = dim a F(a, a*, a, a, 1, 1). For self-dual a, kappa_a is
    `frobenius_schur`, Z_a^T = kappa_a Z_a and kappa_a = R(a, a, 1). Otherwise kappa_a is a
    gauge-dependent phase, kappa_a = conj R(a*, a, 1), and Z_(a*) = R(a*, a, 1) Z_a^T."""
    one, err = sym.unit(), 0.0
    for a in labels:
        Z, ab = cup(sym, a), sym.dual(a)
        err = _max(err, Z @ Z.conj().T - np.eye(len(Z)))
        kappa = sym.dim(a) * f_tensor(sym, a, ab, a, a, one, one)[0, 0, 0, 0]
        if ab == a:
            err = _max(err, kappa - sym.frobenius_schur(a))
            err = _max(err, Z.T - kappa * Z)
            err = _max(err, r_matrix(sym, a, a, one)[0, 0] - kappa)
        else:
            R = r_matrix(sym, ab, a, one)[0, 0]
            err = _max(err, kappa - np.conj(R))
            err = _max(err, cup(sym, ab) - R * Z.T)
    return err


def all_errors(sym, labels, gs, pentagon_labels=None):
    """Every float check above, by name."""
    pl = labels if pentagon_labels is None else pentagon_labels
    return {"cg unitarity": cg_unitarity_error(sym, labels),
            "intertwiner": intertwiner_error(sym, labels, gs),
            "F against CG": f_against_cg_error(sym, labels),
            "R against CG": r_against_cg_error(sym, labels),
            "R symmetric": r_symmetric_error(sym, labels),
            "F unitary": f_unitarity_error(sym, labels),
            "pentagon": pentagon_error(sym, pl),
            "hexagons": hexagon_error(sym, labels),
            "unit": unit_error(sym, labels),
            "duals": dual_error(sym, labels)}
