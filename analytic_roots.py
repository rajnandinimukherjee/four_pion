"""Finite-volume levels by direct diagonalization, as a check on the roots.

At leading order the quantization condition is equivalent to diagonalizing the
truncated Hamiltonian in the ``A_1^+`` orbit basis,

    H_{mu' mu} = E_mu delta_{mu' mu} + V^d_{mu' mu} + u_mu' lambda_{N'N} u_mu ,

with the same ingredients that build ``G``: the non-interacting energies, the
disconnected blocks inside each degenerate level, and the rank-one connected
vertex of eq. (2.11).  The matrix determinant lemma turns ``det[E - H] = 0``
into

    det[E - H^(0) - V^d] det[1 - G(E, L) lambda] = 0 ,

so the eigenvalues of ``H`` away from the levels of ``Ghat`` are exactly the
solutions of ``det[G^{-1} - lambda] = 0``.  Comparing the two is therefore a
check of the root search, and the eigenvectors give the same sector content as
``QC.content`` without any root finding.
"""

from __future__ import annotations

from typing import Any, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from tqdm import tqdm

from gfunctions import Couplings
from quantisation_condition import QC, SECTORS, pad_levels
from utils import call_PDF

Array = NDArray[Any]


def hamiltonian(qc: QC) -> tuple[Array, Array]:
    """Build ``H`` in the orbit basis, and the sector label of each orbit."""
    sectors = qc.sectors
    functions = [qc.G.sectors[N] for N in sectors]
    sizes = [len(function.u) for function in functions]
    offsets = np.cumsum([0] + sizes)
    total = int(offsets[-1])

    matrix = np.zeros((total, total), dtype=float)
    labels = np.zeros(total, dtype=int)

    for position, (N, function) in enumerate(zip(sectors, functions)):
        start = offsets[position]
        labels[start : offsets[position + 1]] = N
        index = np.arange(start, offsets[position + 1])
        matrix[index, index] = function.energies

        for level, members in enumerate(function.level_members):
            blocks = function.level_blocks[level]
            if not blocks:
                continue
            potential = sum(
                qc.couplings[(r, r)] * block for r, block in blocks.items()
            )
            rows = start + members
            matrix[np.ix_(rows, rows)] += 0.5 * (potential + potential.T)

    for row, (N_out, out) in enumerate(zip(sectors, functions)):
        for column, (N_in, inside) in enumerate(zip(sectors, functions)):
            coupling = qc.couplings[(N_out, N_in)]
            if coupling == 0.0:
                continue
            block = coupling * np.outer(out.u, inside.u)
            matrix[
                offsets[row] : offsets[row + 1], offsets[column] : offsets[column + 1]
            ] += block

    return matrix, labels


def levels(qc: QC, emin: float, emax: float) -> tuple[Array, Array]:
    """Eigenvalues of ``H`` in a window, with the four-particle content of each."""
    matrix, labels = hamiltonian(qc)
    eigenvalues, eigenvectors = np.linalg.eigh(matrix)
    inside = (eigenvalues > emin) & (eigenvalues < emax)
    weights = eigenvectors[:, inside] ** 2
    content = np.array(
        [weights[labels == N].sum(axis=0) for N in qc.sectors], dtype=float
    )
    return eigenvalues[inside], content.T


def scan_volumes(
    mLs: ArrayLike,
    couplings: Couplings,
    emin: float = 1.9,
    emax: float = 6.0,
    cutoff: float = 10.0,
    sectors: Sequence[int] = SECTORS,
    alpha: float = 0.0,
    data_dir: str = "data",
    verbose: bool = True,
) -> dict[str, Any]:
    """Diagonalize at every volume of a scan."""
    mLs = np.asarray(mLs, dtype=float)
    per_volume: list[Array] = []
    contents: list[Array] = []
    tables = None

    volumes = tqdm(mLs, desc="mL", leave=False) if verbose else mLs
    for mL in volumes:
        qc = QC(
            couplings,
            mL,
            sectors=sectors,
            cutoff=cutoff,
            mL_ref=float(mLs.max()),
            alpha=alpha,
            data_dir=data_dir,
            tables=tables,
        )
        tables = qc.tables
        energies, content = levels(qc, emin, emax)
        per_volume.append(energies)
        contents.append(content)

    return {
        "mLs": mLs,
        "levels": per_volume,
        "content": contents,
        "sectors": np.array(sectors),
        "couplings": couplings,
        "emin": emin,
        "emax": emax,
    }


def plot_levels(
    result: dict[str, Any],
    filename: str = "plots/levels_diagonalised.pdf",
    figsize: tuple[float, float] = (3.2, 6.0),
    open_pdf: bool = True,
) -> None:
    """Plot the diagonalized levels, coloured by their four-particle content."""
    mLs = np.asarray(result["mLs"], dtype=float)
    energies = pad_levels(result["levels"])
    content = pad_levels([row[:, -1] for row in result["content"]])

    _, ax = plt.subplots(1, 1, figsize=figsize)
    for column in range(energies.shape[1]):
        ax.plot(mLs, energies[:, column], "k-", lw=0.6, alpha=0.4)
    scatter = ax.scatter(
        np.repeat(mLs[:, None], energies.shape[1], axis=1),
        energies,
        c=content,
        s=3,
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
    )
    plt.colorbar(scatter, ax=ax, label="four-particle content")

    ax.set_xlabel(r"$mL$", size=12)
    ax.set_ylabel(r"$E/m$", size=12)
    ax.set_xlim([mLs[0], mLs[-1]])
    ax.set_ylim([result["emin"], result["emax"]])
    call_PDF(filename, open=open_pdf)


def main() -> None:
    mLs = np.linspace(4.0, 7.0, 31)
    couplings = Couplings.two_four(l22=20.0, l24=5000.0, l44=1.0, l33=0.0)
    result = scan_volumes(mLs, couplings, emin=1.9, emax=6.0, cutoff=10.0)
    plot_levels(result, open_pdf=False)


if __name__ == "__main__":
    main()
