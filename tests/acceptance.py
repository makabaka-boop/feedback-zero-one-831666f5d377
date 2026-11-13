"""Independent acceptance suite for the limiting smoother.

This file deliberately re-implements its own float-only Gaussian conditioning
reference instead of importing the production solver's math.  It checks:

  A. hand-computed exact rational cases (the spec example included);
  B. the exact limit against finite, large kappa Gaussian conditioning,
     estimating A and B from differences of finite-kappa covariances;
  C. symmetry / positive semidefiniteness of A and B and the null observation
     law Cov(X|obs) = kappa A + B + o(1) on random small models;
  D. the command line: success replaces the output, failure keeps the old
     file, and validation errors carry JSON paths with zero-based indices.

Run:  python -I tests/acceptance.py
"""

from __future__ import annotations

import json
import os
import random
import subprocess
import sys
import tempfile
from fractions import Fraction

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from smoother.core import solve, to_jsonable           # noqa: E402
from smoother.validate import ValidationError, load_input  # noqa: E402


# ===========================================================================
# independent float reference: plain lists, its own matrix routines
# ===========================================================================
def fz(r, c=None):
    return [[0.0] * (r if c is None else c) for _ in range(r)]


def fT(a):
    return [list(col) for col in zip(*a)]


def fmm(a, b):
    bt = fT(b)
    return [[sum(x * y for x, y in zip(a[i], bt[j]))
             for j in range(len(b[0]))] for i in range(len(a))]


def fmv(a, v):
    return [sum(x * y for x, y in zip(row, v)) for row in a]


def finv(a):
    n = len(a)
    aug = [row[:] + [1.0 if i == j else 0.0 for j in range(n)]
           for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(aug[r][col]))
        assert abs(aug[piv][col]) > 1e-12
        aug[col], aug[piv] = aug[piv], aug[col]
        d = aug[col][col]
        aug[col] = [x / d for x in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0.0:
                f = aug[r][col]
                aug[r] = [x - f * y for x, y in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def finite_kappa(a, P, D, F, Q, h, r, y, kappa):
    """Direct Gaussian conditioning at a finite, large kappa."""
    n = len(a)
    T = len(y)
    N = n * T
    powers = [[[1.0 if i == j else 0.0 for j in range(n)] for i in range(n)]]
    for _ in range(1, T):
        powers.append(fmm(powers[-1], F))
    mu, W = [], []
    for t in range(T):
        mu += fmv(powers[t], a)
        W += fmm(powers[t], D)  # flatten time blocks into N rows
    C0 = fz(N)
    for t in range(T):
        blk = fmm(powers[t], fmm(P, fT(powers[t])))
        for k in range(1, t + 1):
            fk = powers[t - k]
            blk = [ [blk[i][j] + fmm(fmm(fk, Q), fT(fk))[i][j]
                     for j in range(n)] for i in range(n)]
        for i in range(n):
            for j in range(n):
                C0[t * n + i][t * n + j] += blk[i][j]
        for t2 in range(t + 1, T):
            cross = fmm(powers[t2 - t], blk)
            for i in range(n):
                for j in range(n):
                    C0[t2 * n + i][t * n + j] += cross[i][j]
                    C0[t * n + j][t2 * n + i] += cross[i][j]
    Cov = [[C0[i][j] + kappa * sum(W[i][c] * W[j][c] for c in range(len(D[0])))
            for j in range(N)] for i in range(N)]
    obs = [t for t, v in enumerate(y) if v is not None]
    if not obs:
        return mu, Cov
    H = fz(len(obs), N)
    z = []
    for k, t in enumerate(obs):
        for i in range(n):
            H[k][t * n + i] = h[i]
        z.append(y[t] - sum(h[i] * mu[t * n + i] for i in range(n)))
    S = [[fmm(fmm(H, Cov), fT(H))[i][j] + (r if i == j else 0.0)
          for j in range(len(obs))] for i in range(len(obs))]
    Sinv = finv(S)
    CHt = fmm(Cov, fT(H))
    gain = fmm(CHt, Sinv)
    mean = [mu[i] + fmv(gain, z)[i] for i in range(N)]
    gH = fmm(gain, H)
    ccov = [[Cov[i][j] - fmm(gH, Cov)[i][j] for j in range(N)]
            for i in range(N)]
    return mean, ccov


# ===========================================================================
# fraction -> float and small numeric checks
# ===========================================================================
def f2f(mat_or_vec):
    if not mat_or_vec:
        return []
    if isinstance(mat_or_vec[0], Fraction):
        return [float(x) for x in mat_or_vec]
    return [[float(x) for x in row] for row in mat_or_vec]


def scale_of(*blocks):
    m = 1.0
    for blk in blocks:
        for row in blk:
            for v in row:
                m = max(m, abs(float(v)))
    return m


def assert_close(got, want, tol, label):
    if isinstance(got[0], list):
        for i, row in enumerate(got):
            for j, v in enumerate(row):
                if abs(v - want[i][j]) > tol:
                    raise AssertionError(
                        f"{label}[{i}][{j}]: {v} vs {want[i][j]}")
    else:
        for i, v in enumerate(got):
            if abs(v - want[i]) > tol:
                raise AssertionError(f"{label}[{i}]: {v} vs {want[i]}")


def assert_symmetric_psd(mat, label, tol):
    n = len(mat)
    for i in range(n):
        for j in range(i):
            if abs(mat[i][j] - mat[j][i]) > tol:
                raise AssertionError(f"{label} not symmetric at {i},{j}")
    if n == 0:
        return
    # cyclic Jacobi eigenvalues: robust for rank-deficient exact-PSD matrices
    # whose float LDL pivots would suffer catastrophic cancellation.
    work = [row[:] for row in mat]
    for _ in range(100):
        offs = [abs(work[i][j]) for i in range(n)
                for j in range(i + 1, n)]
        if not offs or max(offs) < 1e-12:
            break
        for p in range(n - 1):
            for q in range(p + 1, n):
                apq = work[p][q]
                if abs(apq) < 1e-15:
                    continue
                tau = (work[q][q] - work[p][p]) / (2 * apq)
                t = (1.0 if tau >= 0 else -1.0) / (abs(tau) + (1 + tau ** 2) ** 0.5)
                c = 1 / (1 + t * t) ** 0.5
                s = t * c
                for k in range(n):
                    akp, akq = work[k][p], work[k][q]
                    work[k][p] = c * akp - s * akq
                    work[k][q] = s * akp + c * akq
                for k in range(n):
                    apk, aqk = work[p][k], work[q][k]
                    work[p][k] = c * apk - s * aqk
                    work[q][k] = s * apk + c * aqk
    eig = [work[i][i] for i in range(n)]
    scale = max([1.0] + [abs(v) for v in eig])
    if min(eig) < -1e-9 * scale:
        raise AssertionError(f"{label} not PSD, eigenvalue {min(eig)}")


# ===========================================================================
# exact hand-checked cases
# ===========================================================================
def case_spec_example():
    inp = load_input(json.dumps({
        "a": ["0"], "P": [["0"]], "D": [["1"]], "F": [["1"]],
        "Q": [["0"]], "h": ["1"], "r": "1", "y": [None, "2"],
    }))
    out = solve(inp)
    assert [str(x) for x in out["mean"]] == ["2", "2"]
    for name in ("A", "B"):
        mat = to_jsonable(out)[name]
        assert mat == [["0", "0"], ["0", "0"]] if name == "A" else \
               mat == [["1", "1"], ["1", "1"]], mat
    print("ok  spec example: mean=[2,2], A=0, B all ones")


def case_unidentified_drift():
    # h = 0: nothing is ever observed about the drift direction.
    inp = load_input(json.dumps({
        "a": ["3"], "P": [["2"]], "D": [["5"]], "F": [["1"]],
        "Q": [["7"]], "h": ["0"], "r": "1", "y": [None, "4", None],
    }))
    out = solve(inp)
    assert out["mean"] == [Fraction(3), Fraction(3), Fraction(3)]
    assert to_jsonable(out)["A"] == [["25", "25", "25"]] * 3
    # B is the plain prior covariance C0 (identifying part):
    # C0_tt = 2 + 7t ; C0_st = 2 + 7 min(s,t)
    T = 3
    for s in range(T):
        for t in range(T):
            assert out["B"][s][t] == Fraction(2 + 7 * min(s, t))
    print("ok  unidentified drift: A = W W^T retained, B = prior C0")


def case_partially_identified_drift():
    # n = d = 2, h observes only component 0 each time; component 1 drift
    # (amplitude D col 1) is never seen and must survive in A.
    inp = load_input(json.dumps({
        "a": ["0", "0"],
        "P": [["0", "0"], ["0", "0"]],
        "D": [["1", "0"], ["0", "1"]],
        "F": [["1", "0"], ["0", "1"]],
        "Q": [["0", "0"], ["0", "0"]],
        "h": ["1", "0"], "r": "1",
        "y": ["4", None],
    }))
    out = solve(inp)
    j = to_jsonable(out)
    assert j["mean"] == ["4", "0", "4", "0"], j["mean"]
    # A keeps component 1 at both times, component 0 is fully identified.
    want_A = [[0, 0, 0, 0], [0, 1, 0, 1], [0, 0, 0, 0], [0, 1, 0, 1]]
    assert [[int(x) for x in row] for row in j["A"]] == want_A, j["A"]
    # B: diffuse-prior observation of an unknown constant with r=1 gives
    # limiting posterior variance r (=1), not the finite-prior r/2.
    one = "1"
    want_B = [[one, "0", one, "0"], ["0", "0", "0", "0"],
              [one, "0", one, "0"], ["0", "0", "0", "0"]]
    assert j["B"] == want_B, j["B"]
    print("ok  partially identified drift: unseen direction kept in A only")


def case_all_missing_is_prior():
    inp = load_input(json.dumps({
        "a": ["1", "-1"],
        "P": [["2", "1"], ["1", "2"]],
        "D": [["1", "0"], ["0", "1"]],
        "F": [["1", "0"], ["0", "1"]],
        "Q": [["1", "0"], ["0", "1"]],
        "h": ["1", "1"], "r": "2", "y": [None, None],
    }))
    out = solve(inp)
    assert out["mean"] == [Fraction(1), Fraction(-1),
                           Fraction(1), Fraction(-1)]
    # A = W W^T with W = [D; D]
    assert out["A"][0][0] == 1 and out["A"][0][2] == 1
    assert out["A"][3][3] == 1 and out["A"][1][2] == 0
    # B = C0: blocks P + t Q, cross P + min(s,t) Q
    assert out["B"][0] == [Fraction(2), Fraction(1),
                           Fraction(2), Fraction(1)]
    assert out["B"][3][3] == Fraction(3)
    assert out["B"][2][3] == Fraction(1)
    print("ok  all-missing: prior mean and A = WW^T, B = C0 returned")


def case_d_zero_remains_finite():
    inp = load_input(json.dumps({
        "a": ["0"], "P": [["1"]], "D": [[]], "F": [["1"]],
        "Q": [["1"]], "h": ["1"], "r": "1", "y": ["0", None, "3"],
    }))
    out = solve(inp)
    j = to_jsonable(out)
    assert all(x == "0" for row in j["A"] for x in row)
    # hand-checked classical Kalman smoother (P0=1, Q=1, r=1):
    # forward means 0, 0, 3/2 ; backward pass gives 3/7, 9/7, 15/7,
    # posterior variances 3/7, 6/7, 5/7 and cross-time covs 2/7, 1/7, 3/7.
    assert j["mean"] == ["3/7", "9/7", "15/7"], j["mean"]
    assert out["B"][0][0] == Fraction(3, 7)
    assert out["B"][1][1] == Fraction(6, 7)
    assert out["B"][2][2] == Fraction(5, 7)
    assert out["B"][0][1] == Fraction(2, 7)
    assert out["B"][0][2] == Fraction(1, 7)
    assert out["B"][1][2] == Fraction(3, 7)
    print("ok  d=0: A identically zero, ordinary finite smoother in B")


def case_d_zero_all_missing_shape():
    # d=0 together with zero observations: A must still be a full N x N
    # zero matrix, not a matrix of empty rows.
    inp = load_input(json.dumps({
        "a": ["2"], "P": [["3"]], "D": [[]], "F": [["1"]],
        "Q": [["5"]], "h": ["1"], "r": "1", "y": [None, None],
    }))
    out = solve(inp)
    N = 2
    assert len(out["A"]) == N and all(len(row) == N for row in out["A"])
    assert all(x == 0 for row in out["A"] for x in row)
    assert len(out["B"]) == N and all(len(row) == N for row in out["B"])
    # prior mean/covariance pass through untouched
    assert out["mean"] == [Fraction(2), Fraction(2)]
    assert out["B"][0][0] == Fraction(3)
    assert out["B"][1][1] == Fraction(8)
    print("ok  d=0 + all-missing: full-size zero A, prior B preserved")


def case_scalar_input_rejection():
    base = {
        "a": ["0"], "P": [["0"]], "D": [["1"]], "F": [["1"]],
        "Q": [["0"]], "h": ["1"], "r": "1", "y": ["1"],
    }

    def must_fail(mutate, path):
        payload = dict(base)
        mutate(payload)
        try:
            load_input(json.dumps(payload))
        except ValidationError as exc:
            assert exc.path == path, (exc.path, path)
            return
        raise AssertionError(f"expected failure at {path}")

    must_fail(lambda p: p.update(r=True), "$.r")
    must_fail(lambda p: p.update(y=[1]), "$.y[0]")
    must_fail(lambda p: p.update(a=[0]), "$.a[0]")
    must_fail(lambda p: p.update(P=[[0]]), "$.P[0][0]")
    must_fail(lambda p: p.update(r=" 1"), "$.r")
    must_fail(lambda p: p.update(r="1/ 2"), "$.r")
    must_fail(lambda p: p.update(r="1.5"), "$.r")
    must_fail(lambda p: p.update(y=[True]), "$.y[0]")
    print("ok  bare numbers, booleans and malformed strings rejected")


# ===========================================================================
# random models vs finite kappa reference
# ===========================================================================
def random_model(rng):
    n = rng.randint(1, 3)
    d = rng.randint(0, n)
    T = rng.randint(1, 4)

    def mat(rows, cols, span=2):
        return [[Fraction(rng.randint(-span, span)) for _ in range(cols)]
                for _ in range(rows)]

    F = mat(n, n, 2)
    G = mat(n, n, 2)
    P = fmm_frac(G, transpose_frac(G))
    Gq = mat(n, n, 2)
    Q = fmm_frac(Gq, transpose_frac(Gq))
    h = [Fraction(rng.randint(-2, 2)) for _ in range(n)]
    a = [Fraction(rng.randint(-3, 3)) for _ in range(n)]
    # full column rank D: distinct pivot rows with unit diagonal entries.
    D = [[Fraction(0) for _ in range(d)] for _ in range(n)]
    rows = rng.sample(range(n), d)
    for c, rr in enumerate(rows):
        D[rr][c] = Fraction(1)
    # off-pivot noise is restricted to rows that hold no pivot, so no
    # constructed column can become a linear combination of the others.
    free_rows = [i for i in range(n) if i not in rows]
    for c in range(d):
        for i in free_rows:
            D[i][c] = Fraction(rng.randint(-1, 1))
    r = Fraction(rng.randint(1, 4))
    y = [None if rng.random() < 0.35 else str(rng.randint(-4, 4))
         for _ in range(T)]
    if all(v is None for v in y):
        y[0] = "1"
    return {
        "a": [str(x) for x in a],
        "P": [[str(x) for x in row] for row in P],
        "D": [[str(x) for x in row] for row in D],
        "F": [[str(x) for x in row] for row in F],
        "Q": [[str(x) for x in row] for row in Q],
        "h": [str(x) for x in h], "r": str(r), "y": y,
    }


def transpose_frac(a):
    return [list(col) for col in zip(*a)]


def fmm_frac(a, b):
    bt = transpose_frac(b)
    return [[sum(x * y for x, y in zip(a[i], bt[j]))
             for j in range(len(b[0]))] for i in range(len(a))]


def run_random_convergence():
    rng = random.Random(20261113)
    kappas = (1000, 4000, 10000)
    for trial in range(40):
        payload = random_model(rng)
        inp = load_input(json.dumps(payload))
        exact = solve(inp)
        n, T, d = inp["n"], inp["T"], inp["d"]
        fa = f2f(inp["a"]); fP = f2f(inp["P"]); fD = f2f(inp["D"])
        fF = f2f(inp["F"]); fQ = f2f(inp["Q"]); fh = f2f(inp["h"])
        fr = float(inp["r"]); fy = [None if v is None else float(v)
                                    for v in inp["y"]]
        means, covs = [], []
        for kap in kappas:
            m, c = finite_kappa(fa, fP, fD, fF, fQ, fh, fr, fy, kap)
            means.append(m); covs.append(c)
        # mean has a finite limit; its O(1/kappa) transients are checked
        # against the expected 1/kappa decay between successive kappas.
        sc_m = max(1.0, max(abs(v) for v in means[-1]),
                   max(abs(float(v)) for v in exact["mean"]))
        # float reference at k=1e4 carries conditioning error of order 1e-2
        assert_close(means[-1], f2f(exact["mean"]),
                     2e-2 * sc_m, f"trial{trial} mean limit")
        d01 = max(abs(a - b) for a, b in zip(means[0], means[1]))
        d12 = max(abs(a - b) for a, b in zip(means[1], means[2]))
        ratio = (kappas[2] - kappas[1]) / (kappas[1] - kappas[0])
        # d12 should be roughly d01 / ratio (allow slack for float noise)
        assert d12 <= max(1e-4 * sc_m, 1.5 * d01 / ratio), \
            f"trial{trial} mean 1/kappa decay: {d01} -> {d12}"
        # A from secant slopes of Cov(kappa); B from intercept
        k1, k2, k3 = kappas
        Ahat = [[(covs[2][i][j] - covs[1][i][j]) / (k3 - k2)
                 for j in range(n * T)] for i in range(n * T)]
        Bhat2 = [[covs[2][i][j] - k3 * Ahat[i][j]
                  for j in range(n * T)] for i in range(n * T)]
        Bhat1 = [[covs[1][i][j] - k2 * Ahat[i][j]
                  for j in range(n * T)] for i in range(n * T)]
        sc_a = max([1.0] + [abs(float(v)) for row in exact["A"]
                            for v in row])
        sc_b = max([1.0] + [abs(float(v)) for row in exact["B"]
                            for v in row])
        # secant/intercept extraction loses precision through cancellation,
        # so the independent float reference uses loose relative tolerances.
        assert_close(Ahat, f2f(exact["A"]),
                     2e-2 * sc_a, f"trial{trial} A")
        assert_close(Bhat2, f2f(exact["B"]),
                     6e-2 * sc_b, f"trial{trial} B")
        assert_close(Bhat1, Bhat2, 6e-2 * sc_b,
                     f"trial{trial} B intercept")
        assert_symmetric_psd(f2f(exact["A"]), f"trial{trial} A", 1e-9)
        assert_symmetric_psd(f2f(exact["B"]), f"trial{trial} B", 1e-9)
        # residual law: Cov(k) - (k A + B) = o(1), checked relatively
        sc_c = max([1.0] + [abs(covs[2][i][j])
                            for i in range(n * T)
                            for j in range(n * T)])
        for kap, cov in ((k2, covs[1]), (k3, covs[2])):
            resid = [[cov[i][j] - kap * float(exact["A"][i][j])
                      - float(exact["B"][i][j])
                      for j in range(n * T)] for i in range(n * T)]
            assert max(abs(v) for row in resid for v in row) < 5e-2 * sc_c, \
                f"trial{trial} o(1) residual too large at kappa={kap}"
        if d == 0:
            assert all(x == 0 for row in f2f(exact["A"]) for x in row)
    print(f"ok  40 random models match finite-kappa limits "
          f"(k up to {kappas[-1]}, independent float reference)")


# ===========================================================================
# validation and CLI behaviour
# ===========================================================================
def expect_error(payload_text, path_fragment):
    try:
        load_input(payload_text)
    except ValidationError as exc:
        assert path_fragment in exc.path, (exc.path, path_fragment)
        return exc
    raise AssertionError(f"expected error mentioning {path_fragment}")


def case_validation_paths():
    base = {
        "a": ["0"], "P": [["0"]], "D": [["1"]], "F": [["1"]],
        "Q": [["0"]], "h": ["1"], "r": "1", "y": ["1"],
    }
    expect_error(json.dumps(dict(base, a=[])), "$.a")
    expect_error(json.dumps(dict(base, P=[["-1"]])), "$.P")
    expect_error(json.dumps(dict(base, P=[["0", "1"], ["1", "0"]])), "$.P")
    expect_error(json.dumps(dict(base, Q=[["1/0"]])), "$.Q[0][0]")
    expect_error(json.dumps(dict(base, r="0")), "$.r")
    expect_error(json.dumps(dict(base, r="1/-2")), "$.r")
    expect_error(json.dumps(dict(base, h=["1", "2"])), "$.h")
    expect_error(json.dumps(dict(base, F=[["1", "0"]])), "$.F")
    expect_error(json.dumps(dict(base, D=[["1", "1"]])), "$.D")
    expect_error(json.dumps(dict(base, y=[])), "$.y")
    expect_error(json.dumps(dict(base, y=[1])), "$.y[0]")
    expect_error(json.dumps(dict(base, y=["1/2", None, "x"])), "$.y[2]")
    expect_error('{"a": ["0"], "a": ["1"], "P": [["0"]], '
                 '"D": [["1"]], "F": [["1"]], "Q": [["0"]], '
                 '"h": ["1"], "r": "1", "y": ["1"]}', "$")
    expect_error(json.dumps(dict(base, zzz="1")), "$.zzz")
    expect_error("{not json", "$")
    print("ok  validation rejects bad shapes/values with indexed paths")


def case_cli_file_lifecycle():
    with tempfile.TemporaryDirectory() as td:
        in_path = os.path.join(td, "in.json")
        out_path = os.path.join(td, "nested", "out.json")
        with open(in_path, "w", encoding="utf-8") as fh:
            json.dump({
                "a": ["0"], "P": [["0"]], "D": [["1"]], "F": [["1"]],
                "Q": [["0"]], "h": ["1"], "r": "1", "y": [None, "2"],
            }, fh)
        # -I keeps the interpreter isolated; the top-level entry script
        # inserts the repo root itself, so no environment path is needed.
        entry = os.path.join(ROOT, "run_smoother.py")
        rc = subprocess.run([sys.executable, "-I", entry,
                             in_path, out_path], cwd=ROOT,
                            capture_output=True, text=True)
        assert rc.returncode == 0, rc.stderr
        with open(out_path, encoding="utf-8") as fh:
            result = json.load(fh)
        assert set(result) == {"mean", "A", "B"}
        assert result["mean"] == ["2", "2"]
        assert result["B"] == [["1", "1"], ["1", "1"]]

        # failure must preserve the previously produced file
        bad = {
            "a": [], "P": [["0"]], "D": [["1"]], "F": [["1"]],
            "Q": [["0"]], "h": ["1"], "r": "1", "y": ["1"],
        }
        bad_path = os.path.join(td, "bad.json")
        with open(bad_path, "w", encoding="utf-8") as fh:
            json.dump(bad, fh)
        before = open(out_path, encoding="utf-8").read()
        rc = subprocess.run([sys.executable, "-I", entry,
                             bad_path, out_path], cwd=ROOT,
                            capture_output=True, text=True)
        assert rc.returncode == 1
        err = json.loads(rc.stderr)
        assert err["error"]["path"] == "$.a"
        assert open(out_path, encoding="utf-8").read() == before
    print("ok  CLI: success atomically replaces, failure keeps old output")


def main():
    case_spec_example()
    case_unidentified_drift()
    case_partially_identified_drift()
    case_all_missing_is_prior()
    case_d_zero_remains_finite()
    case_d_zero_all_missing_shape()
    case_scalar_input_rejection()
    run_random_convergence()
    case_validation_paths()
    case_cli_file_lifecycle()
    print("\nALL ACCEPTANCE CHECKS PASSED")


if __name__ == "__main__":
    main()
