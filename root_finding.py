"""Root finding for ``det[1 - G(E, L) lambda] = 0`` with exact root counting.

The finite-volume function of every sector is a sum of simple poles with
non-negative residues (``gfunctions.py``),

    G_NN(E) = sum_k w_k r_k / (E - p_k) ,

where ``w_k = w(E^(0)) >= 0`` is the regulator weight of the level, a constant
for both the hard and the smooth cutoff.  The weights are folded into the
residues below; ``Secular.check_pole_sum`` verifies the form at run time.

so the quantization condition is a *secular equation*: its solutions are the
eigenvalues of the diagonal-plus-low-rank matrix

    H = D + W lambda W^T ,   D = diag(p_k) ,   W_{kN} = sqrt(r_k) [k in sector N] ,

which is the truncated Hamiltonian of ``analytic_roots.py`` written in the basis
that diagonalizes ``H^(0) + V^d``.  Two consequences drive the methods here.

1.  Root counting.  Write ``lambda = Q Lambda Q^T`` on its non-zero
    eigenvalues.  Haynsworth inertia additivity applied to
    ``[[E - D, W Q], [Q^T W^T, Lambda^{-1}]]`` gives

        #{roots > E} = #{poles > E} + n_-( Lambda^{-1} - Q^T G(E) Q ) - const ,

    so on an interval free of poles the number of roots is the drop of
    ``m(E) = n_-(Lambda^{-1} - Q^T G(E) Q)``, a ``rank(lambda) x rank(lambda)``
    inertia that costs one evaluation of ``G``.  No root can be missed, however
    close two roots or a root and a pole are, and at most ``rank(lambda)`` roots
    sit between two consecutive poles.

2.  Relative accuracy.  A root next to a pole sits at ``E = p + t`` with ``t``
    as small as ``r lambda``, ``~1e-10`` for the four-particle levels.  Every
    evaluation is therefore done in shifted coordinates, ``E - p_j = (p - p_j) +
    t`` with ``t`` stored separately, so ``t`` is found to relative accuracy
    instead of to ``eps |E|`` (cf. LAPACK ``dlaed4``).

Before solving, the pole list is *deflated*: poles of one sector that coincide
within rounding (degenerate ``V^d`` eigenvalues) are merged with their residues
summed, and poles whose residue is zero up to the rounding of the ``V^d``
eigenvectors are dropped, since they decouple from the connected vertex and are
not solutions of the quantization condition.

Methods
-------
``baseline``  the grid + ``brentq`` search of ``quantisation_condition.py``.
``eig``       dense ``eigvalsh`` of the deflated ``D + W lambda W^T``.
``brent``     root counting to isolate, scalar ``brentq`` in shifted coordinates.
``newton``    root counting to isolate, vectorized safeguarded Newton on the
              pole-free function ``h(t) = t (T - t) f(t)`` in shifted
              coordinates, all roots of a window at once.  The default.

Run ``python root_finding.py`` for the benchmark.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Sequence

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import brentq

from gfunctions import Couplings
from quantisation_condition import QC, energy_grid, find_roots

Array = NDArray[Any]

EPS = np.finfo(float).eps
# Offset from a pole at which the limit E -> p^{+-} is taken.  Small enough to
# sit below any resolvable root shift, large enough that r / t and the product
# of two such terms stay finite.
T_EDGE = 1e-120


# ---------------------------------------------------------------------------
# the secular problem
# ---------------------------------------------------------------------------


@dataclass
class Secular:
    """Deflated poles and residues of ``G`` together with the couplings.

    Attributes
    ----------
    poles, residues:
        All retained poles of every sector, sorted, and their residues.
    onehot:
        ``(n_poles, n_sectors)`` matrix, ``residue`` in the column of the
        pole's sector, so that ``G(E) = (1 / (E - poles)) @ onehot``.
    lam:
        Coupling matrix in the sector basis.
    """

    poles: Array
    residues: Array
    sector: Array
    lam: Array
    n_dropped: int = 0
    n_merged: int = 0

    def __post_init__(self) -> None:
        n_sectors = self.lam.shape[0]
        if n_sectors > 2:
            raise NotImplementedError("only one or two sectors are supported")
        self.onehot = np.zeros((len(self.poles), n_sectors))
        self.onehot[np.arange(len(self.poles)), self.sector] = self.residues

        values, vectors = np.linalg.eigh(self.lam)
        scale = max(np.max(np.abs(values)), 1e-300)
        keep = np.abs(values) > 1e-13 * scale
        self.rank = int(keep.sum())
        self.lam_values = values[keep]
        self.lam_vectors = vectors[:, keep]
        if self.rank == 2:
            inverse = np.linalg.inv(self.lam)
            self.inv_det_sign = np.sign(np.linalg.det(inverse))
            self.inv_trace = np.trace(inverse)

    # -- construction -------------------------------------------------------

    @classmethod
    def from_qc(cls, qc: QC, merge_tol: float = 64 * EPS) -> "Secular":
        """Deflated poles and residues of every sector of ``qc``.

        The residues include the regulator weight ``w`` of each level
        (``GFunction.level_weights``).  ``check_pole_sum`` then verifies that
        ``G`` really is this pole sum, so that a regulator which made ``G``
        depend on ``E`` other than through its poles fails loudly here instead
        of silently invalidating the root count.
        """
        poles: list[Array] = []
        residues: list[Array] = []
        sectors: list[Array] = []
        dropped = merged = 0
        for position, N in enumerate(qc.sectors):
            function = qc.G.sectors[N]
            level_weights = getattr(function, "level_weights", None)
            p_sector: list[float] = []
            r_sector: list[float] = []
            for level, members in enumerate(function.level_members):
                u = function.u[members]
                w = 1.0 if level_weights is None else float(level_weights[level])
                shifts = function.level_shifts[level]
                overlaps = function.level_vectors[level].T @ u
                weight = w * overlaps**2
                # Rounding of eigh leaves |overlap| ~ eps |u| on decoupled
                # states; a level the regulator switches off (w = 0) drops out.
                zero = weight <= (64 * EPS) ** 2 * w * float(u @ u)
                dropped += int(zero.sum())
                p_level = function.level_energies[level] + shifts[~zero]
                r_level = weight[~zero]
                p_sector.extend(p_level.tolist())
                r_sector.extend(r_level.tolist())
            p, r, n = _merge(np.array(p_sector), np.array(r_sector), merge_tol)
            merged += n
            poles.append(p)
            residues.append(r)
            sectors.append(np.full(len(p), position))
        p = np.concatenate(poles)
        order = np.argsort(p, kind="stable")
        problem = cls(
            poles=p[order],
            residues=np.concatenate(residues)[order],
            sector=np.concatenate(sectors)[order],
            lam=qc.lam.copy(),
            n_dropped=dropped,
            n_merged=merged,
        )
        problem.check_pole_sum(qc)
        return problem

    def check_pole_sum(self, qc: QC, n_probe: int = 8, rtol: float = 1e-9) -> None:
        """Require ``G`` of ``qc`` to equal ``sum_k r_k / (E - p_k)``, ``r_k >= 0``.

        This is the property the root count rests on: with non-negative
        residues ``G'`` is negative definite between poles, so the eigenvalues
        of ``lambda^{-1} - G(E)`` rise monotonically and cross zero at most once
        each.  An ``E``-dependent regulator (for instance a damping
        ``exp(alpha (E - E_pole))``) breaks it, and is rejected here.
        """
        if np.any(self.residues < 0):
            raise ValueError("G has a negative residue; root counting is not valid")
        for position, N in enumerate(qc.sectors):
            function = qc.G.sectors[N]
            own = function.poles
            if len(own) < 2:
                continue
            # Probe midway between poles, where both sums are well conditioned.
            gaps = np.diff(own)
            widest = np.sort(np.argsort(gaps)[-n_probe:])
            probes = own[widest] + 0.5 * gaps[widest]
            with np.errstate(all="ignore"):  # spurious matmul flags, see ``g``
                expected = np.asarray(function(probes), dtype=float)
                ours = self.g(probes, np.zeros_like(probes))[:, position]
                scale = np.abs(1.0 / (probes[:, None] - own[None, :])) @ np.abs(function.residues)
            error = np.max(np.abs(ours - expected) / np.maximum(scale, 1e-300))
            if error > rtol:
                raise ValueError(
                    f"G_{N}{N} is not the pole sum root counting assumes "
                    f"(relative mismatch {error:.1e})"
                )

    # -- evaluation in shifted coordinates ------------------------------------

    def g(self, anchor: Array, t: Array, derivative: bool = False) -> Any:
        """Diagonal of ``G`` at ``E = anchor + t``; shapes ``(n,)`` -> ``(n, n_s)``."""
        gaps = (np.asarray(anchor)[:, None] - self.poles[None, :]) + np.asarray(t)[:, None]
        # numpy's matmul on macOS (Accelerate) raises divide/overflow/invalid
        # flags on finite inputs; the values are checked by the callers.
        with np.errstate(all="ignore"):
            inverse = 1.0 / gaps
            value = inverse @ self.onehot
            if not derivative:
                return value
            return value, -(inverse**2) @ self.onehot

    def f(self, anchor: Array, t: Array, derivative: bool = False) -> Any:
        """``det[1 - G lambda]`` (and ``d/dE``) at ``E = anchor + t``."""
        lam = self.lam
        if derivative:
            g, dg = self.g(anchor, t, derivative=True)
        else:
            g = self.g(anchor, t)
        if lam.shape[0] == 1:
            value = 1.0 - g[:, 0] * lam[0, 0]
            return (value, -dg[:, 0] * lam[0, 0]) if derivative else value
        det = lam[0, 0] * lam[1, 1] - lam[0, 1] ** 2
        g2, g4 = g[:, 0], g[:, 1]
        value = 1.0 - g2 * lam[0, 0] - g4 * lam[1, 1] + g2 * g4 * det
        if not derivative:
            return value
        d2, d4 = dg[:, 0], dg[:, 1]
        slope = -d2 * lam[0, 0] - d4 * lam[1, 1] + (d2 * g4 + g2 * d4) * det
        return value, slope

    def count(self, anchor: Array, t: Array) -> Array:
        """``m(E) = n_-(Lambda^{-1} - Q^T G Q)``; its drop counts the roots."""
        anchor = np.atleast_1d(np.asarray(anchor, dtype=float))
        t = np.atleast_1d(np.asarray(t, dtype=float))
        if self.rank == 0:
            return np.zeros(len(t), dtype=int)
        g = self.g(anchor, t)
        if self.rank == 1:
            q = self.lam_vectors[:, 0]
            value = 1.0 / self.lam_values[0] - g @ (q**2)
            return (value < 0).astype(int)
        # Rank two: M = lambda^{-1} - diag(g).  det M = det(lambda^{-1}) f, which
        # is evaluated without cancellation even when one g is huge.
        det_sign = self.inv_det_sign * np.sign(self.f(anchor, t))
        trace = self.inv_trace - g.sum(axis=1)
        return np.where(det_sign < 0, 1, np.where(trace > 0, 0, 2))


def _merge(p: Array, r: Array, tol: float) -> tuple[Array, Array, int]:
    """Merge poles of one sector that coincide to rounding, summing residues."""
    if len(p) == 0:
        return p, r, 0
    order = np.argsort(p)
    p, r = p[order], r[order]
    new_group = np.diff(p) > tol * np.maximum(np.abs(p[1:]), 1.0)
    labels = np.concatenate([[0], np.cumsum(new_group)])
    n_groups = labels[-1] + 1
    weights = np.bincount(labels, weights=r, minlength=n_groups)
    centres = np.bincount(labels, weights=p * r, minlength=n_groups) / weights
    return centres, weights, len(p) - n_groups


# ---------------------------------------------------------------------------
# isolation by counting
# ---------------------------------------------------------------------------


@dataclass
class Bracket:
    """One isolated root: ``E = anchor + t`` with ``t`` between ``lo`` and ``hi``."""

    anchor: float
    lo: float
    hi: float
    other: float  # offset of the other end pole from the anchor (or inf)


def isolate(problem: Secular, emin: float, emax: float, max_depth: int = 80) -> list[Bracket]:
    """Brackets around each root in ``(emin, emax)``, one root per bracket."""
    poles = problem.poles
    inside = poles[(poles > emin) & (poles < emax)]
    # Pole-free intervals (a, b) with the poles bounding them (None at a window edge).
    edges = np.concatenate([[emin], inside, [emax]])
    brackets: list[Bracket] = []
    for left, right in zip(edges[:-1], edges[1:]):
        left_pole = left if left != emin else None
        right_pole = right if right != emax else None
        _split(problem, left, right, left_pole, right_pole, brackets, max_depth)
    return brackets


def _point(E: float, left_pole: float | None, right_pole: float | None, side: int) -> tuple[float, float]:
    """``(anchor, t)`` for an interval end; ``side`` is +1 at the left end."""
    pole = left_pole if side > 0 else right_pole
    if pole is not None and E == pole:
        return pole, side * T_EDGE
    return E, 0.0


def _split(
    problem: Secular,
    a: float,
    b: float,
    left_pole: float | None,
    right_pole: float | None,
    out: list[Bracket],
    depth: int,
) -> None:
    """Bisect ``(a, b)`` with ``count`` until each piece holds a single root."""
    ends = [_point(a, left_pole, right_pole, +1), _point(b, left_pole, right_pole, -1)]
    m = problem.count(np.array([ends[0][0], ends[1][0]]), np.array([ends[0][1], ends[1][1]]))
    n_roots = int(m[0] - m[1])
    if n_roots < 0:
        raise RuntimeError(f"negative root count on ({a!r}, {b!r})")
    if n_roots == 0:
        return

    mid = 0.5 * (a + b)
    if n_roots == 1 or depth == 0 or not (a < mid < b):
        out.extend(_anchor(problem, a, b, mid, left_pole, right_pole, n_roots))
        return
    # Keep the true poles as the edge markers only where they bound the piece.
    _split(problem, a, mid, left_pole, None, out, depth - 1)
    _split(problem, mid, b, None, right_pole, out, depth - 1)


def _anchor(
    problem: Secular,
    a: float,
    b: float,
    mid: float,
    left_pole: float | None,
    right_pole: float | None,
    n_roots: int,
) -> list[Bracket]:
    """Express a one-root bracket relative to the nearer bounding pole.

    One extra count at the midpoint says which half holds the root; the
    bracket is then written as offsets from the pole at that end, so that a
    root hugging the pole keeps all its significant digits.
    """
    if n_roots > 1:  # unresolvable at double precision: a degenerate root
        return [Bracket(mid, 0.0, 0.0, np.inf)] * n_roots
    if left_pole is not None and right_pole is not None:
        m = problem.count(np.array([mid, right_pole]), np.array([0.0, -T_EDGE]))
        in_left = m[0] == m[1]  # no root in (mid, b)
        if in_left:
            return [Bracket(left_pole, T_EDGE, mid - left_pole, right_pole - left_pole)]
        return [Bracket(right_pole, mid - right_pole, -T_EDGE, left_pole - right_pole)]
    if left_pole is not None:
        return [Bracket(left_pole, T_EDGE, b - left_pole, np.inf)]
    if right_pole is not None:
        return [Bracket(right_pole, a - right_pole, -T_EDGE, np.inf)]
    return [Bracket(a, 0.0, b - a, np.inf)]


# ---------------------------------------------------------------------------
# solvers on isolated brackets
# ---------------------------------------------------------------------------


def solve_brent(problem: Secular, brackets: Sequence[Bracket]) -> tuple[Array, Array]:
    """Scalar ``brentq`` on ``f`` in shifted coordinates, one bracket at a time.

    Returns the anchors and offsets ``t`` of the roots, ``E = anchor + t``.
    """
    anchors, offsets = [], []
    for bracket in brackets:
        anchors.append(bracket.anchor)
        if bracket.lo == bracket.hi:
            offsets.append(bracket.lo)
            continue
        anchor = np.array([bracket.anchor])

        def func(t: float) -> float:
            return float(problem.f(anchor, np.array([t]))[0])

        offsets.append(brentq(func, bracket.lo, bracket.hi, xtol=1e-300, rtol=4 * EPS, maxiter=200))
    return _sorted(np.array(anchors, dtype=float), np.array(offsets, dtype=float))


def _sorted(anchors: Array, offsets: Array) -> tuple[Array, Array]:
    order = np.argsort(anchors + offsets, kind="stable")
    return anchors[order], offsets[order]


def solve_newton(
    problem: Secular, brackets: Sequence[Bracket], max_iter: int = 100
) -> tuple[Array, Array, int]:
    """Vectorized safeguarded Newton on ``h(t) = t (T - t) f(t)``.

    Multiplying by ``t`` removes the pole at the anchor and by ``T - t`` the
    pole at the far end, so near a pole ``h`` is close to linear in ``t`` and
    Newton converges in a few steps even when the root is ``1e-12`` from it.
    A step that leaves the bracket is replaced by a bisection, geometric when
    the bracket spans many decades.  Returns anchors, offsets and iterations.
    """
    if not brackets:
        return np.zeros(0), np.zeros(0), 0
    anchor = np.array([b.anchor for b in brackets])
    lo = np.array([b.lo for b in brackets])
    hi = np.array([b.hi for b in brackets])
    far = np.array([b.other for b in brackets])
    far = np.where(np.isfinite(far), far, np.inf)

    def h(t: Array, idx: Array) -> tuple[Array, Array]:
        value, slope = problem.f(anchor[idx], t, derivative=True)
        a = anchor[idx]
        anchored = np.isin(a, problem.poles)
        # The prefactor only multiplies by the poles that are actually present.
        s = np.where(anchored, t, 1.0)
        ds = np.where(anchored, 1.0, 0.0)
        T = far[idx]
        u = np.where(np.isfinite(T), T - t, 1.0)
        du = np.where(np.isfinite(T), -1.0, 0.0)
        pre = s * u
        dpre = ds * u + s * du
        return pre * value, dpre * value + pre * slope

    sign_lo = np.sign(problem.f(anchor, lo))
    t = 0.5 * (lo + hi)
    active = lo != hi
    t[~active] = lo[~active]
    iterations = 0
    while np.any(active) and iterations < max_iter:
        iterations += 1
        idx = np.flatnonzero(active)
        value, slope = h(t[idx], idx)
        # Shrink the bracket using the sign of f (same as the sign of h up to
        # the prefactor, whose sign is fixed on the bracket).
        f_sign = np.sign(problem.f(anchor[idx], t[idx]))
        left = f_sign == sign_lo[idx]
        lo[idx] = np.where(left, t[idx], lo[idx])
        hi[idx] = np.where(left, hi[idx], t[idx])

        with np.errstate(divide="ignore", invalid="ignore"):
            correction = value / slope
        # Converged on the Newton correction itself, before any safeguarding:
        # a converged step can land a rounding error outside the bracket.
        converged = (value == 0) | (np.abs(correction) <= 2 * EPS * np.abs(t[idx]))
        step = t[idx] - correction
        lower = np.minimum(lo[idx], hi[idx])
        upper = np.maximum(lo[idx], hi[idx])
        ok = np.isfinite(step) & (step > lower) & (step < upper)
        small = np.minimum(np.abs(lower), np.abs(upper))
        large = np.maximum(np.abs(lower), np.abs(upper))
        geometric = (np.sign(lower) == np.sign(upper)) & (large > 1e3 * small)
        bisect = np.where(
            geometric,
            np.sign(upper) * np.sqrt(small * large),
            0.5 * (lower + upper),
        )
        collapsed = upper - lower <= 4 * EPS * large
        t[idx] = np.where(converged | collapsed, t[idx], np.where(ok, step, bisect))
        active[idx[converged | collapsed]] = False
    return (*_sorted(anchor, t), iterations)


# ---------------------------------------------------------------------------
# front ends
# ---------------------------------------------------------------------------


def roots_baseline(qc: QC, emin: float, emax: float, n_coarse: int = 2000, delta: float = 1e-11) -> Array:
    grid = energy_grid(emin, emax, qc.poles, n_coarse=n_coarse, delta=delta)
    return find_roots(qc, grid, qc.poles)


def roots_eig(qc: QC, emin: float, emax: float) -> Array:
    problem = Secular.from_qc(qc)
    w = np.sqrt(problem.residues)[:, None] * (problem.onehot != 0)
    with np.errstate(all="ignore"):  # spurious matmul flags, see ``Secular.g``
        h = np.diag(problem.poles) + w @ problem.lam @ w.T
    values = np.linalg.eigvalsh(h)
    return values[(values > emin) & (values < emax)]


def roots_brent(qc: QC, emin: float, emax: float) -> Array:
    problem = Secular.from_qc(qc)
    anchors, offsets = solve_brent(problem, isolate(problem, emin, emax))
    return anchors + offsets


def roots_newton(qc: QC, emin: float, emax: float) -> Array:
    problem = Secular.from_qc(qc)
    anchors, offsets, _ = solve_newton(problem, isolate(problem, emin, emax))
    return anchors + offsets


def shifts(qc: QC, emin: float, emax: float) -> tuple[Array, Array, Array]:
    """Roots as ``(E, pole, E - pole)`` with the offset to full relative precision.

    ``pole`` is the pole of ``G`` the root was solved relative to (the window
    edge for a root far from any pole).  Use the offset, not ``E - pole``
    recomputed from ``E``, when the interaction shift itself is the observable.
    """
    problem = Secular.from_qc(qc)
    anchors, offsets, _ = solve_newton(problem, isolate(problem, emin, emax))
    return anchors + offsets, anchors, offsets


METHODS: dict[str, Callable[[QC, float, float], Array]] = {
    "baseline": roots_baseline,
    "eig": roots_eig,
    "brent": roots_brent,
    "newton": roots_newton,
}


# ---------------------------------------------------------------------------
# reference roots at extended precision
# ---------------------------------------------------------------------------


def polish_mp(
    problem: Secular, anchors: Array, offsets: Array, digits: int = 40
) -> tuple[Array, Array]:
    """Refine roots in ``mpmath`` on the same (double-precision) poles.

    The offsets returned are the exact solutions for the given input data,
    rounded to double, and serve as the truth in the benchmark.
    """
    import mpmath

    mpmath.mp.dps = digits
    poles = [mpmath.mpf(float(p)) for p in problem.poles]
    residues = [mpmath.mpf(float(r)) for r in problem.residues]
    sector = problem.sector.tolist()
    lam = [[mpmath.mpf(float(x)) for x in row] for row in problem.lam]
    n_s = len(lam)

    out = []
    for anchor_value, offset in zip(anchors, offsets):
        anchor = mpmath.mpf(float(anchor_value))
        on_pole = bool(np.isin(anchor_value, problem.poles))

        def f(t: Any) -> Any:
            g = [mpmath.mpf(0)] * n_s
            for p, r, s in zip(poles, residues, sector):
                g[s] += r / ((anchor - p) + t)
            if n_s == 1:
                return 1 - g[0] * lam[0][0]
            det = lam[0][0] * lam[1][1] - lam[0][1] ** 2
            return 1 - g[0] * lam[0][0] - g[1] * lam[1][1] + g[0] * g[1] * det

        def h(t: Any) -> Any:
            return t * f(t) if on_pole else f(t)

        t0 = mpmath.mpf(float(offset))
        try:
            t = mpmath.findroot(h, t0, tol=(abs(t0) * mpmath.mpf(10) ** (-digits + 5)) ** 2)
        except (ValueError, ZeroDivisionError):
            t = t0
        out.append(float(t))
    return np.asarray(anchors, dtype=float), np.array(out)


# ---------------------------------------------------------------------------
# benchmark
# ---------------------------------------------------------------------------


@dataclass
class Case:
    name: str
    couplings: Couplings
    mLs: Array
    emin: float = 1.9
    emax: float = 6.0


def compare(
    found: tuple[Array, Array], truth: tuple[Array, Array]
) -> dict[str, float]:
    """Missed/spurious roots, ``max |dE|`` and the relative error of the shift.

    Roots are given as ``(anchor, offset)``.  The error of the shift is measured
    on ``offset`` re-expressed relative to the truth's anchor, so a method that
    only returns ``E`` (anchor ``E``, offset ``0``) is charged for the digits of
    ``E - pole`` it lost.
    """
    f_anchor, f_offset = found
    t_anchor, t_offset = truth
    f_energy = f_anchor + f_offset
    t_energy = t_anchor + t_offset
    used = np.zeros(len(f_energy), dtype=bool)
    missing = 0
    abs_err: list[float] = []
    rel_err: list[float] = []
    for anchor, offset, energy in zip(t_anchor, t_offset, t_energy):
        candidates = np.flatnonzero(~used)
        if len(candidates) == 0:
            missing += 1
            continue
        j = candidates[np.argmin(np.abs(f_energy[candidates] - energy))]
        tolerance = max(0.5 * abs(offset), 1e-9 * abs(energy))
        if abs(f_energy[j] - energy) > tolerance:
            missing += 1
            continue
        used[j] = True
        shift = (f_anchor[j] - anchor) + f_offset[j]
        abs_err.append(abs(f_energy[j] - energy))
        rel_err.append(abs(shift - offset) / max(abs(offset), 1e-300))
    return {
        "missing": missing,
        "spurious": int((~used).sum()),
        "abs": max(abs_err, default=0.0),
        "rel": max(rel_err, default=0.0),
    }


def _energies(method: Callable[..., Array]) -> Callable[[QC, float, float], tuple[Array, Array]]:
    def wrapped(qc: QC, emin: float, emax: float) -> tuple[Array, Array]:
        energies = np.asarray(method(qc, emin, emax), dtype=float)
        return energies, np.zeros_like(energies)

    return wrapped


def _offsets(solver: str) -> Callable[[QC, float, float], tuple[Array, Array]]:
    def wrapped(qc: QC, emin: float, emax: float) -> tuple[Array, Array]:
        problem = Secular.from_qc(qc)
        brackets = isolate(problem, emin, emax)
        if solver == "brent":
            return solve_brent(problem, brackets)
        return solve_newton(problem, brackets)[:2]

    return wrapped


VARIANTS: dict[str, Callable[[QC, float, float], tuple[Array, Array]]] = {
    "baseline n=2000": _energies(roots_baseline),
    "baseline n=20000": _energies(lambda qc, a, b: roots_baseline(qc, a, b, n_coarse=20000)),
    "eig": _energies(roots_eig),
    "count + brent": _offsets("brent"),
    "count + newton": _offsets("newton"),
}


def run_case(
    case: Case, tables: dict[int, Any], cutoff: float = 10.0, repeat: int = 3
) -> dict[str, dict[str, float]]:
    print(f"\n== {case.name}: {case.couplings.label()}, mL in "
          f"[{case.mLs[0]:g}, {case.mLs[-1]:g}] ({len(case.mLs)} volumes)")
    totals = {name: {"missing": 0, "spurious": 0, "abs": 0.0, "rel": 0.0, "time": 0.0}
              for name in VARIANTS}
    n_truth = n_double = n_iter = n_dropped = n_merged = 0
    build = 0.0
    for mL in case.mLs:
        start = time.perf_counter()
        qc = QC(case.couplings, mL, cutoff=cutoff, mL_ref=float(case.mLs.max()), tables=tables)
        build += time.perf_counter() - start
        problem = Secular.from_qc(qc)
        n_dropped += problem.n_dropped
        n_merged += problem.n_merged
        brackets = isolate(problem, case.emin, case.emax)
        n_double += _count_shared(brackets)
        anchors, offsets, iterations = solve_newton(problem, brackets)
        n_iter = max(n_iter, iterations)
        truth = polish_mp(problem, anchors, offsets)
        n_truth += len(truth[0])
        for name, method in VARIANTS.items():
            start = time.perf_counter()
            for _ in range(repeat):
                found = method(qc, case.emin, case.emax)
            totals[name]["time"] += (time.perf_counter() - start) / repeat
            stats = compare(found, truth)
            entry = totals[name]
            entry["missing"] += stats["missing"]
            entry["spurious"] += stats["spurious"]
            entry["abs"] = max(entry["abs"], stats["abs"])
            entry["rel"] = max(entry["rel"], stats["rel"])
    n_vol = len(case.mLs)
    print(f"   {n_truth} roots; {n_double} pole-free intervals with two roots; "
          f"deflation dropped {n_dropped} / merged {n_merged} poles; "
          f"newton <= {n_iter} sweeps; QC build {1e3 * build / n_vol:.2f} ms/vol")
    print(f"   {'method':<18}{'missed':>8}{'spurious':>10}{'max |dE|':>11}"
          f"{'max shift err':>15}{'time/vol':>11}")
    for name, entry in totals.items():
        print(f"   {name:<18}{entry['missing']:>8d}{entry['spurious']:>10d}"
              f"{entry['abs']:>11.1e}{entry['rel']:>15.1e}"
              f"{1e3 * entry['time'] / n_vol:>9.2f}ms")
    return totals


def _count_shared(brackets: Sequence[Bracket]) -> int:
    """Number of pole-free intervals that contained two roots."""
    anchors = [(b.anchor, b.other) for b in brackets]
    keys = [tuple(sorted((a, a + o))) if np.isfinite(o) else (a,) for a, o in anchors]
    return sum(1 for key in set(keys) if keys.count(key) > 1)


def main() -> None:
    from orbits import load_orbits

    tables = {N: load_orbits(N, emax=10.0, mL=7.0, data_dir="data") for N in (2, 4)}
    mLs = np.linspace(4.0, 7.0, 31)
    cases = [
        Case("production couplings", Couplings.two_four(20.0, 5000.0, 1.0), mLs),
        Case("with lambda_33", Couplings.two_four(20.0, 5000.0, 50.0, 10.0), mLs),
        Case("weak mixing", Couplings.two_four(20.0, 1.0, 1.0), mLs),
        Case("weak couplings (shifts below delta)", Couplings.two_four(1.0, 0.01, 0.01), mLs),
        Case("strong mixing", Couplings.two_four(20.0, 1e5, 1e3), mLs),
    ]
    for case in cases:
        run_case(case, tables)


if __name__ == "__main__":
    main()
