"""Exact rational linear algebra for the limiting Gaussian smoother.

Only :class:`fractions.Fraction` and plain lists are used.  Every routine is
written from scratch: no numeric thresholds, no square roots, no symbolic
engines, no third-party packages.
"""

from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

Matrix = List[List[Fraction]]
Vector = List[Fraction]

ZERO = Fraction(0)
ONE = Fraction(1)


# ---------------------------------------------------------------------------
# basic constructors
# ---------------------------------------------------------------------------
def zeros(rows: int, cols: Optional[int] = None) -> Matrix:
    if cols is None:
        cols = rows
    return [[ZERO for _ in range(cols)] for _ in range(rows)]


def eye(size: int) -> Matrix:
    out = zeros(size)
    for i in range(size):
        out[i][i] = ONE
    return out


def transpose(a: Sequence[Sequence[Fraction]],
              out_rows: Optional[int] = None,
              out_cols: Optional[int] = None) -> Matrix:
    """Transpose of ``a``.

    Empty matrices carry no dimension in their row lists, so when needed the
    result shape (out_rows x out_cols) may be supplied explicitly.
    """
    rows = len(a)
    cols = len(a[0]) if rows else 0
    if out_rows is None:
        out_rows, out_cols = cols, rows
    elif out_cols is None:
        out_cols = rows
    if rows == 0 or cols == 0:
        return zeros(out_rows, out_cols)
    return [[a[i][j] for i in range(rows)] for j in range(cols)]


def matmul(a: Sequence[Sequence[Fraction]],
           b: Sequence[Sequence[Fraction]]) -> Matrix:
    """Exact matrix product a @ b (zero inner dimensions yield a zero).

    Shapes follow the rule ``(r x k) @ (k x c) -> (r x c)``; a zero-width
    matrix has empty rows, a zero-height matrix is an empty outer list.
    """
    rows = len(a)
    inner = len(b)
    if rows == 0:
        cols = len(b[0]) if inner else 0
        return zeros(0, cols)
    if len(a[0]) != inner:
        raise ValueError("matmul: inner dimensions do not match")
    cols = len(b[0]) if inner else len(a[0])
    if inner == 0:
        return zeros(rows, cols)
    bt = transpose(b)
    return [[sum((a[i][k] * bt[j][k] for k in range(inner)), ZERO)
             for j in range(cols)]
            for i in range(rows)]


def matvec(a: Sequence[Sequence[Fraction]],
           v: Sequence[Fraction]) -> Vector:
    if not a:
        return []
    if len(a[0]) != len(v):
        raise ValueError("matvec: dimensions do not match")
    return [sum((a[i][k] * v[k] for k in range(len(v))), ZERO)
            for i in range(len(a))]


def add(a: Matrix, b: Matrix, sign: Fraction = ONE) -> Matrix:
    return [[a[i][j] + sign * b[i][j]
             for j in range(len(a[0]))] for i in range(len(a))]


def scale(a: Matrix, s: Fraction) -> Matrix:
    return [[s * a[i][j] for j in range(len(a[0]))]
            for i in range(len(a))]


# ---------------------------------------------------------------------------
# exact inversion via Gauss-Jordan elimination
# ---------------------------------------------------------------------------
def inverse(a: Sequence[Sequence[Fraction]]) -> Matrix:
    """Return the exact inverse of the square matrix ``a``.

    Raises ``ZeroDivisionError`` if the matrix is singular.
    """
    n = len(a)
    aug = [row[:] + [ONE if i == j else ZERO for j in range(n)]
           for i, row in enumerate(a)]
    for col in range(n):
        pivot = next((r for r in range(col, n) if aug[r][col] != ZERO), -1)
        if pivot < 0:
            raise ZeroDivisionError("matrix is singular")
        if pivot != col:
            aug[col], aug[pivot] = aug[pivot], aug[col]
        piv = aug[col][col]
        aug[col] = [x / piv for x in aug[col]]
        for r in range(n):
            if r == col:
                continue
            factor = aug[r][col]
            if factor != ZERO:
                aug[r] = [x - factor * y for x, y in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


def pivot_columns(a: Sequence[Sequence[Fraction]]) -> List[int]:
    """Indices of linearly independent columns of ``a`` (row-echelon form).

    Exact rational elimination; the selected columns span range(a).
    """
    if not a:
        return []
    rows = len(a)
    cols = len(a[0])
    work = [row[:] for row in a]
    chosen: List[int] = []
    row = 0
    for col in range(cols):
        pivot = next((r for r in range(row, rows) if work[r][col] != ZERO),
                     -1)
        if pivot < 0:
            continue
        work[row], work[pivot] = work[pivot], work[row]
        piv = work[row][col]
        for r in range(rows):
            if r == row or work[r][col] == ZERO:
                continue
            factor = work[r][col] / piv
            work[r] = [x - factor * y for x, y in zip(work[r], work[row])]
        chosen.append(col)
        row += 1
        if row == rows:
            break
    return chosen


def rank(a: Sequence[Sequence[Fraction]]) -> int:
    return len(pivot_columns(a))


def column_space_projection(a: Sequence[Sequence[Fraction]]) -> Matrix:
    """Exact orthogonal projection onto range(a): ``B (BᵀB)⁻¹ Bᵀ``.

    ``B`` holds an exact column basis of ``a`` (a subset of its columns), so
    ``BᵀB`` is invertible whenever B is full-column-rank.  No square root is
    needed because the ordinary inverse already gives the Euclidean
    projection in the standard basis.
    """
    cols = pivot_columns(a)
    dim = len(a[0]) if a else 0
    if not cols:
        return zeros(dim)
    b = [[row[c] for c in cols] for row in a]
    btb = matmul(transpose(b), b)
    return matmul(b, matmul(inverse(btb), transpose(b)))


# ---------------------------------------------------------------------------
# exact positive-semidefinite test for a symmetric matrix
# ---------------------------------------------------------------------------
def is_symmetric(a: Sequence[Sequence[Fraction]]) -> bool:
    n = len(a)
    return all(len(row) == n for row in a) and all(
        a[i][j] == a[j][i] for i in range(n) for j in range(i + 1, n))


def is_symmetric_psd(a: Sequence[Sequence[Fraction]]) -> bool:
    """Exact symmetric PSD test through symmetric pivot elimination.

    For a symmetric PSD matrix a zero pivot forces the whole pivot row and
    column to vanish, so the elimination simply continues on the trailing
    principal block.  A non-zero pivot performs the ordinary Schur update;
    any negative pivot or negative diagonal of a remaining block rejects the
    matrix.  All comparisons are exact.
    """
    n = len(a)
    if any(len(row) != n for row in a):
        return False
    work = [row[:] for row in a]
    k = 0
    while k < n:
        # symmetric row/column swap to bring a non-zero entry to (k, k)
        pivot_row = next((i for i in range(k, n) if work[i][k] != ZERO), -1)
        if pivot_row < 0:
            # column (and, by symmetry, row) k of the residual is all zero:
            # PSD can only hold if every remaining diagonal is non-negative
            return all(work[i][i] >= ZERO for i in range(k, n))
        if pivot_row != k:
            work[k], work[pivot_row] = work[pivot_row], work[k]
            for i in range(n):
                work[i][k], work[i][pivot_row] = (work[i][pivot_row],
                                                  work[i][k])
        piv = work[k][k]
        if piv < ZERO:
            return False
        if piv == ZERO:
            # PSD with zero (k,k) forces the whole pivot row to vanish
            if any(work[k][j] != ZERO for j in range(k, n)):
                return False
            k += 1
            continue
        inv = ONE / piv
        for i in range(k + 1, n):
            aik = work[i][k]
            if aik == ZERO:
                continue
            for j in range(k + 1, n):
                work[i][j] -= aik * work[k][j] * inv
        k += 1
    return True
