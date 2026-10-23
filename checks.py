"""Analytic checks of the orbit construction and of the quantization condition.

Run with ``python checks.py``.  Each check compares the code against a result
that is known independently, either from the draft or from a brute-force
calculation:

1.  orbit sums:  ``sum_mu (d_mu / N_mu) f_mu = (1/N!) sum_{k_1..k_{N-1}} f``,
    eq. (2.17), checked against an explicit sum over momenta,
2.  ``G_22``:    against the direct two-particle sum of section 3.1,
3.  thresholds:  the two- and four-particle threshold expansions of section 3.1,
    including the ``lambda_{2<-2}``, ``lambda_{3<-3}`` and ``lambda_{4<-4}``
    terms,
4.  ``V^d``:     symmetry of the disconnected blocks in every degenerate level,
5.  the roots:   against the eigenvalues of the truncated Hamiltonian, which
    the matrix determinant lemma says must agree.
"""

from __future__ import annotations


import math
from typing import Any, Callable

import numpy as np
from numpy.typing import NDArray

import orbits
from analytic_roots import levels as diagonalised_levels
from gfunctions import Couplings, GFunction
from quantisation_condition import QC

Array = NDArray[Any]

FAILURES: list[str] = []


def report(name: str, got: float, expected: float, tol: float = 1e-10) -> None:
    """Compare against a known value, relatively unless the value vanishes."""
    error = abs(got - expected)
    if expected != 0.0:
        error /= abs(expected)
    status = "ok " if error < tol else "FAIL"
    if error >= tol:
        FAILURES.append(name)
    print(f"  [{status}] {name}: got {got:.12g}, expected {expected:.12g} "
          f"(rel. {error:.2e})")


def check_orbit_sum(mL: float = 5.0, cutoff: float = 7.0) -> None:
    """Eq. (2.17) for ``f = 1 / Omega``, i.e. the sum rule obeyed by ``u_mu``."""
    print("orbit sum rule, N = 4")
    table = orbits.generate(N=4, emax=cutoff, mL=mL, disconnected=False, verbose=False)
    function = GFunction(table, mL=mL, emax=cutoff)
    from_orbits = float(np.sum(function.u**2) * mL**9)

    tpiL = 2 * np.pi / mL
    # With three companions at rest, one particle can carry at most cutoff - 3.
    radius = int(math.isqrt(max(0, int(((cutoff - 3) ** 2 - 1) / tpiL**2)))) + 1
    axis = np.arange(-radius, radius + 1)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    grid = grid.reshape(-1, 3)
    energies = np.sqrt(1.0 + np.einsum("ij,ij->i", grid, grid) * tpiL**2)

    total = 0.0
    for first in range(len(grid)):
        for second in range(len(grid)):
            fourth = -(grid[first] + grid[second] + grid)
            energy_fourth = np.sqrt(
                1.0 + np.einsum("ij,ij->i", fourth, fourth) * tpiL**2
            )
            energy = energies[first] + energies[second] + energies + energy_fourth
            keep = energy <= cutoff
            if not np.any(keep):
                continue
            product = (
                16.0
                * energies[first]
                * energies[second]
                * energies[keep]
                * energy_fourth[keep]
            )
            total += float(np.sum(1.0 / product))
    report("sum_mu d_mu u_mu^2 L^9 == (1/4!) sum_k 1/Omega", from_orbits, total / 24)


def check_g22(mL: float = 6.0, cutoff: float = 8.0, E: float = 3.3) -> None:
    """``G_22`` against the explicit two-particle sum of section 3.1."""
    print("G_22 against the direct momentum sum")
    table = orbits.generate(N=2, emax=cutoff, mL=mL, disconnected=False, verbose=False)
    function = GFunction(table, mL=mL, emax=cutoff)

    tpiL = 2 * np.pi / mL
    radius = int(math.isqrt(max(0, int(((cutoff / 2) ** 2 - 1) / tpiL**2)))) + 1
    axis = np.arange(-radius, radius + 1)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    grid = grid.reshape(-1, 3)
    single = np.sqrt(1.0 + np.einsum("ij,ij->i", grid, grid) * tpiL**2)
    pair = 2 * single
    keep = pair <= cutoff
    direct = float(
        np.sum(1.0 / (2 * single[keep]) ** 2 / (E - pair[keep])) / (2 * mL**3)
    )
    report("G_22(E)", float(function(E)), direct)


def check_thresholds(mL: float = 6.0, cutoff: float = 8.0) -> None:
    """The threshold expansions quoted in section 3.1."""
    print("threshold expansions")
    l22, l33, l44 = 2.0, 3.0, 5.0
    couplings = Couplings.two_four(l22=l22, l24=0.0, l44=l44, l33=l33)

    table2 = orbits.generate(N=2, emax=cutoff, mL=mL, verbose=False)
    table4 = orbits.generate(N=4, emax=cutoff, mL=mL, verbose=False)
    qc = QC(couplings, mL=mL, cutoff=cutoff, tables={2: table2, 4: table4})

    # Poles of G: the free levels shifted by the disconnected interactions.
    four = qc.G.sectors[4]
    report(
        "4-particle level, V^d shift",
        float(four.poles[0]),
        4 + 6 * l22 / (8 * mL**3) + 4 * l33 / (48 * mL**6),
    )
    report("4-particle residue u_mu^2", float(four.residues[0]), 1 / (384 * mL**9))

    # Roots just above threshold, with the connected couplings switched on one
    # sector at a time so that the leading terms can be read off.
    two_only = Couplings.two_four(l22=l22, l24=0.0, l44=0.0)
    qc.set_couplings(two_only)
    roots = qc.roots(1.5, 2.5)
    report(
        "2-particle threshold level",
        float(roots[0]),
        2 + l22 / (8 * mL**3),
        tol=1e-5,
    )

    qc.set_couplings(couplings)
    roots = qc.roots(3.9, 4.3)
    report(
        "4-particle threshold level",
        float(roots[0]),
        4 + 6 * l22 / (8 * mL**3) + 4 * l33 / (48 * mL**6) + l44 / (384 * mL**9),
        tol=1e-6,
    )


def check_vd_symmetry(mL: float = 6.0, cutoff: float = 9.0) -> None:
    """``V^d`` must be symmetric inside every degenerate level."""
    print("V^d block symmetry")
    table = orbits.generate(N=4, emax=cutoff, mL=mL, verbose=False)
    function = GFunction(table, mL=mL, emax=cutoff)
    worst = 0.0
    mixed = 0
    for level, blocks in enumerate(function.level_blocks):
        if len(function.level_members[level]) > 1 and blocks:
            mixed += 1
        for block in blocks.values():
            scale = max(float(np.max(np.abs(block))), 1e-300)
            worst = max(worst, float(np.max(np.abs(block - block.T))) / scale)
    print(f"  [info] {mixed} levels contain more than one orbit")
    report("max relative asymmetry", worst, 0.0, tol=1e-10)


def check_against_diagonalisation(mL: float = 5.5, cutoff: float = 8.0) -> None:
    """The roots must be the eigenvalues of the truncated Hamiltonian."""
    print("roots against direct diagonalization")
    couplings = Couplings.two_four(l22=20.0, l24=2000.0, l44=50.0, l33=10.0)
    table2 = orbits.generate(N=2, emax=cutoff, mL=mL, verbose=False)
    table4 = orbits.generate(N=4, emax=cutoff, mL=mL, verbose=False)
    qc = QC(couplings, mL=mL, cutoff=cutoff, tables={2: table2, 4: table4})

    emin, emax = 2.5, 5.5
    roots = qc.roots(emin, emax)
    eigenvalues, _ = diagonalised_levels(qc, emin, emax)
    print(f"  [info] {len(roots)} roots, {len(eigenvalues)} eigenvalues in the window")
    if len(roots) != len(eigenvalues):
        FAILURES.append("root count")
        print("  [FAIL] different number of levels")
        return
    worst = float(np.max(np.abs(roots - eigenvalues)))
    report("largest deviation", worst, 0.0, tol=1e-8)


def main() -> int:
    checks: list[Callable[[], None]] = [
        check_g22,
        check_orbit_sum,
        check_thresholds,
        check_vd_symmetry,
        check_against_diagonalisation,
    ]
    for check in checks:
        check()
        print()
    if FAILURES:
        print(f"[fail] {len(FAILURES)} check(s) failed: {', '.join(FAILURES)}")
        return 1
    print("[ok] all checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
