"""Limiting Kalman smoother for a diffuse initial state.

Model
-----
    x_0 ~ N(a, P + kappa D D^T),          kappa -> +infinity
    x_{t+1} = F x_t + w_t,  w_t ~ N(0, Q)
    y_t = h x_t + v_t,      v_t ~ N(0, r)
with all noises and the initial state mutually independent; ``null`` entries of
``y`` mark missing observations (not zero measurements).

The stacked state ``X = (x_0, ..., x_{T-1})`` is written as
``X = mu + W delta + eta`` where ``delta ~ N(0, kappa I_d)`` carries the
unknown initial drift, ``eta ~ N(0, C0)`` is the fully identified part and

    W_{t,*} = F^t D,
    mu_t    = F^t a,
    C0_{s,t} = F^s P (F^t)^T + sum_{k=1}^{min(s,t)} F^{s-k} Q (F^{t-k})^T.

Conditioning the observed coordinates ``z = H(mu + W delta + eta) + v`` gives
``E0 = C0 H^T`` and the strictly positive definite
``S0 = H C0 H^T + r I``.  Writing ``U = W^T H^T`` (d x m), an exact
Woodbury expansion of the conditional Gaussian, followed by the exact limit
kappa -> infinity, yields with
``M = U S0^{-1} U^T``, ``P0`` the projection onto range(M), ``M^+`` the
Moore-Penrose inverse of the symmetric PSD matrix M and
``K = W - E0 S0^{-1} U^T``:

    E[X | z] = mu + (E0 S0^{-1} + K M^+ U S0^{-1}) (z - H mu)
    Cov(X | z) = kappa A + B + o(1)
    A = W (I - P0) W^T
    B = C0 - E0 S0^{-1} E0^T + K M^+ K^T.

Every coefficient is an exact rational number; no finite kappa, numeric
threshold or symbolic limit engine is involved.
"""

from __future__ import annotations

from fractions import Fraction
from typing import Any, Dict, List

from . import linalg as la
from .linalg import Matrix, Vector, ZERO, ONE


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------
def _set_block(big: Matrix, block: Matrix, row0: int, col0: int) -> None:
    for i, row in enumerate(block):
        for j, value in enumerate(row):
            big[row0 + i][col0 + j] = value


def _pseudoinverse_symmetric(m: Matrix) -> Matrix:
    """Exact Moore-Penrose inverse of a symmetric PSD matrix.

    With B a full-column-rank selection of pivot columns spanning range(M),
    ``M^+ = B (B^T M B)^{-1} B^T`` (all quantities exact rationals).
    """
    d = len(m)
    cols = la.pivot_columns(m)
    if not cols:
        return la.zeros(d)
    b = [[row[c] for c in cols] for row in m]
    btm = la.matmul(la.transpose(b), la.matmul(m, b))
    return la.matmul(b, la.matmul(la.inverse(btm), la.transpose(b)))


# ---------------------------------------------------------------------------
# main entry
# ---------------------------------------------------------------------------
def solve(inp: Dict[str, Any]) -> Dict[str, Any]:
    a: Vector = inp["a"]
    P: Matrix = inp["P"]
    D: Matrix = inp["D"]
    F: Matrix = inp["F"]
    Q: Matrix = inp["Q"]
    h: Vector = inp["h"]
    r: Fraction = inp["r"]
    y: List[Fraction | None] = inp["y"]
    n: int = inp["n"]
    d: int = inp["d"]
    T: int = inp["T"]
    size = n * T

    # state transition powers F^0 .. F^{T-1} --------------------------------
    powers: List[Matrix] = [la.eye(n)]
    for _ in range(1, T):
        powers.append(la.matmul(powers[-1], F))

    # prior mean mu and drift loading W -------------------------------------
    mu: Vector = []
    w_blocks: List[Matrix] = []
    for t in range(T):
        mu.extend(la.matvec(powers[t], a))
        w_blocks.append(la.matmul(powers[t], D))
    W: Matrix = [row for block in w_blocks for row in block]  # size x d

    # prior covariance C0 of the identified component eta -------------------
    c0 = la.zeros(size)
    for t in range(T):
        ft = powers[t]
        block = la.matmul(ft, la.matmul(P, la.transpose(ft)))
        for k in range(1, t + 1):
            fk = powers[t - k]
            block = la.add(block, la.matmul(fk, la.matmul(Q, la.transpose(fk))))
        _set_block(c0, block, t * n, t * n)
        for t2 in range(t + 1, T):
            cross = la.matmul(powers[t2 - t], block)
            _set_block(c0, cross, t2 * n, t * n)
            _set_block(c0, la.transpose(cross), t * n, t2 * n)

    # observation selection H, residual z = y - H mu ------------------------
    observed = [t for t, value in enumerate(y) if value is not None]
    m = len(observed)
    H = la.zeros(m, size)
    z: Vector = []
    for k, t in enumerate(observed):
        for i in range(n):
            H[k][t * n + i] = h[i]
        fitted = sum((h[i] * mu[t * n + i] for i in range(n)), ZERO)
        z.append(y[t] - fitted)  # type: ignore[operator]

    # E0 = C0 H^T, U = (H W)^T, S0 = H C0 H^T + r I -------------------------
    ht = la.transpose(H, size, m)
    e0 = la.matmul(c0, ht)
    hw = la.matmul(H, W)                      # m x d
    U = la.transpose(hw, d, m)                # d x m
    s0 = la.add(la.matmul(H, e0), la.scale(la.eye(m), r))

    if m == 0:
        # pure prior: every drift direction remains unidentified
        if d == 0:
            A = la.zeros(size)
        else:
            A = la.matmul(W, la.transpose(W, d, size))
        return {"mean": mu, "A": A, "B": [row[:] for row in c0]}

    s0_inv = la.inverse(s0)                   # strictly positive definite

    if d == 0:
        # No diffuse direction: the limit is the ordinary finite smoother,
        # A vanishes identically and only the identified part remains.
        e0_sinv = la.matmul(e0, s0_inv)
        gain = la.matvec(e0_sinv, z)
        mean = [mu[i] + gain[i] for i in range(size)]
        identified = la.matmul(e0_sinv, la.transpose(e0, m, size))
        B = la.add(c0, identified, -ONE)
        return {"mean": mean, "A": la.zeros(size), "B": B}

    M = la.matmul(U, la.matmul(s0_inv, la.transpose(U, m, d)))  # d x d, PSD
    p0 = la.column_space_projection(M)
    m_plus = _pseudoinverse_symmetric(M)
    q0 = la.add(la.eye(d), p0, -ONE)

    e0_sinv = la.matmul(e0, s0_inv)           # size x m
    k_tilde = la.add(W, la.matmul(e0_sinv, hw), -ONE)

    # limiting conditional mean ---------------------------------------------
    sinv_z = la.matvec(s0_inv, z)
    gain1 = la.matvec(e0_sinv, z)
    gain2 = la.matvec(
        k_tilde,
        la.matvec(m_plus, la.matvec(U, sinv_z)))
    mean = [mu[i] + gain1[i] + gain2[i] for i in range(size)]

    # limiting covariance coefficients --------------------------------------
    A = la.matmul(W, la.matmul(q0, la.transpose(W, d, size)))
    identified = la.matmul(e0_sinv, la.transpose(e0, m, size))
    drift_back = la.matmul(k_tilde,
                           la.matmul(m_plus,
                                     la.transpose(k_tilde, d, size)))
    B = la.add(la.add(c0, identified, -ONE), drift_back)

    return {"mean": mean, "A": A, "B": B}


# ---------------------------------------------------------------------------
# exact serialization
# ---------------------------------------------------------------------------
def _num(value: Fraction) -> str:
    value = Fraction(value)  # guarantees reduced form, positive denominator
    if value.denominator == 1:
        return str(value.numerator)
    return f"{value.numerator}/{value.denominator}"


def to_jsonable(result: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "mean": [_num(v) for v in result["mean"]],
        "A": [[_num(v) for v in row] for row in result["A"]],
        "B": [[_num(v) for v in row] for row in result["B"]],
    }
