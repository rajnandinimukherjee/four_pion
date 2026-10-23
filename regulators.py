"""Fit couplings across regulator choices and compare finite-volume roots."""

from __future__ import annotations

import pdb
import warnings
from typing import Any, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import minimize

from quantisation_condition import B, QC, find_roots, fine_grid

Array = NDArray[Any]

mL_init: float = 10
mL_comp: float = 5
EMAX: int = 20
alpha1: float = 0.1
alpha2: float = 0.2
gs1: Array = np.array([20, 5000, 1])
qc1: QC | None = None
qc2: QC | None = None
E1: Array | None = None


def get_roots(
    qc: QC,
    Emin: float = 1.98,
    Emax: float = 4.0,
    N_EC: int = 1000,
    N_EF: int = 200,
    tol: float = 1e-10,
    verbose: bool = False,
) -> Array:
    """Search coarse and noninteracting-level-centered grids for roots."""
    noninteracting = np.array(qc.NI)
    noninteracting = noninteracting[(noninteracting > Emin) & (noninteracting < Emax)]

    coarse_grid = np.linspace(Emin, Emax, N_EC)
    scale = 10 * qc.B[1, 1] / (qc.mL**9)
    fine_energy_grid = fine_grid(noninteracting, scale, N_E=N_EF)
    fine_energy_grid = fine_energy_grid[
        (fine_energy_grid > Emin) & (fine_energy_grid < Emax)
    ]

    if verbose:
        print(f"[info] mL={qc.mL}, EMAX={qc.EMAX}, alpha={qc.alpha}")
        print(f"[info] g22={qc.B[0, 0]}, g24={qc.B[0, 1]}, g44={qc.B[1, 1]}")
        print(
            f"[info] {len(qc.I2.levels)} levels in S2, "
            f"{len(qc.I4.levels)} levels in S4"
        )
        print(f"[info] expecting {len(noninteracting)} roots between {Emin} and {Emax}")
        print(
            f"[info] using Egrid with {len(coarse_grid)} coarse and "
            f"{len(fine_energy_grid)} fine points"
        )

    coarse_roots = find_roots(coarse_grid, qc, noninteracting, tol=tol, verbose=verbose)
    fine_roots = find_roots(
        fine_energy_grid, qc, noninteracting, tol=tol, verbose=verbose
    )
    roots = list(sorted(set(coarse_roots) | set(fine_roots)))
    if len(roots) < len(noninteracting):
        print("not all roots founds")
        pdb.set_trace()
    elif len(roots) > len(noninteracting):
        print("too many roots found")
        pdb.set_trace()
    return np.array(roots)


def det(qc: QC, gs: Sequence[float], Es: ArrayLike) -> Any:
    """Evaluate the two-channel determinant used in the regulator fit."""
    g22, g24, g44 = gs
    qc.set_B(B(gs))
    S2, S4 = qc.I2(Es), qc2.I4(Es)
    return (1 - S2 * g22) * (1 - S4 * g44) - S2 * S4 * g24**2


def loss(
    gs2: ArrayLike,
    alpha_root: float = 10.0,
    beta_mid: float = 1.0,
    kappa: float = 10.0,
) -> Any:
    """Objective matching roots and determinant signs between regulators."""
    if qc1 is None or qc2 is None or E1 is None:
        raise RuntimeError("Run the regulator comparison before evaluating loss().")

    root_loss = np.sum(det(qc2, gs2, E1) ** 2)

    energy_mids = 0.5 * (E1[:-1] + E1[1:])
    mids1 = det(qc1, gs1, energy_mids)
    mids2 = det(qc2, gs2, energy_mids)
    sign1 = np.tanh(kappa * mids1)
    sign2 = np.tanh(kappa * mids2)
    mid_loss = np.sum((sign1 - sign2) ** 2)

    return alpha_root * root_loss + beta_mid * mid_loss


def main() -> tuple[Any, Array, Array, Array, Array]:
    global E1, qc1, qc2

    warnings.filterwarnings("ignore", category=RuntimeWarning)

    reference_qc = QC(B(gs1), mL_init, alpha=alpha1, EMAX=EMAX)
    qc1 = reference_qc
    E1 = get_roots(reference_qc, verbose=True)
    qc2 = QC(B(gs1), mL=mL_init, alpha=alpha2, EMAX=EMAX)

    result = minimize(loss, gs1, method="L-BFGS-B", tol=1e-12)
    print(result)
    gs2 = result.x
    qc2.set_B(B(gs2))
    E2 = get_roots(qc2, verbose=True)

    print("\n===================================================")
    print(
        f"at mL={mL_init}:\n"
        f"E(alpha1={alpha1}, g1={', '.join(f'{g:.6g}' for g in gs1)}) = "
        f"E(alpha2={alpha2}, g2={', '.join(f'{g:.6g}' for g in gs2)})"
    )
    for root_index in range(len(E1)):
        print(f"{E1[root_index]:.10g}\t{E2[root_index]:.10g}")

    comparison_qc_1 = QC(B(gs1), mL_comp, alpha=alpha1, EMAX=EMAX)
    comparison_roots_1 = get_roots(comparison_qc_1, Emin=1.98, Emax=6.0, verbose=False)
    comparison_qc_2 = QC(B(gs2), mL_comp, alpha=alpha2, EMAX=EMAX)
    comparison_roots_2 = get_roots(comparison_qc_2, Emin=1.98, Emax=6.0, verbose=False)
    print(f"at mL={mL_comp}:\n" "E(alpha1, g1)\tE(alpha2, g2)\t % change")
    percentages = (
        np.abs(comparison_roots_1 - comparison_roots_2) * 100 / comparison_roots_1
    )
    for root_index in range(len(comparison_roots_1)):
        print(
            f"{comparison_roots_1[root_index]:.10g}\t"
            f"{comparison_roots_2[root_index]:.10g}\t"
            f"{percentages[root_index]:.4g}%"
        )

    return result, E1, E2, comparison_roots_1, comparison_roots_2


if __name__ == "__main__":
    main()
