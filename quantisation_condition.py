"""Solutions of ``det[G(E, L)^{-1} - lambda] = 0`` for two and four scalars.

This is the analysis script for section 3.1 of the draft.  It takes the orbit
tables written by ``orbits.py``, builds the finite-volume functions ``G_{NN}``
of ``gfunctions.py``, and solves the quantization condition

    det[ G(E, L)^{-1} - lambda ] = 0 ,
    lambda = [[lambda_{2<-2}, lambda_{2<-4}], [lambda_{4<-2}, lambda_{4<-4}]] ,

for the finite-volume spectrum in the window ``2m < E < 6m``, where only the
two- and four-particle sectors contribute.  The disconnected ``2 -> 2`` and
``3 -> 3`` sub-processes enter through ``G_{44}``, driven by the same
``lambda_{2<-2}`` and by ``lambda_{3<-3}``.

The root search uses the pole-free form of the condition,

    f(E) = det[ 1 - G(E, L) lambda ] ,

which has the same zeros wherever ``G`` is invertible.  Because the poles of
``G`` are known exactly -- they are the non-interacting levels shifted and split
by ``V^d`` -- the search grid is built around them and the brackets that
straddle a pole are discarded, so no spurious roots are produced.

Usage
-----
    python orbits.py --N 2 --emax 10 --mL 7
    python orbits.py --N 4 --emax 10 --mL 7
    python quantisation_condition.py
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import root_scalar
from tqdm import tqdm

from gfunctions import Couplings, GFunction, GMatrix
from orbits import OrbitTable, load_orbits
from plot_settings import plotparams
from utils import call_PDF

Array = NDArray[Any]

plt.rcParams.update(plotparams)

SECTORS: tuple[int, ...] = (2, 4)


class QC:
    """The quantization condition at one volume, for a set of sectors.

    Parameters
    ----------
    couplings:
        The effective couplings ``lambda_{N'<-N}``.
    mL:
        Box size in units of the mass.
    sectors:
        Particle numbers retained; ``(2, 4)`` for section 3.1.
    cutoff:
        Energy cutoff ``E_max`` on the sums over levels inside ``G``.  This is a
        regulator: the states above it are smooth and are absorbed into the
        couplings, so the couplings depend on it (see ``regulators.py``).
    mL_ref:
        Reference ``mL`` of the saved orbit tables.  It must be at least the
        largest ``mL`` used, so that the tables cover every volume in a scan.
    alpha:
        Optional smooth damping of the cutoff, ``exp(alpha (E - E_pole))``.
    """

    def __init__(
        self,
        couplings: Couplings,
        mL: float,
        sectors: Sequence[int] = SECTORS,
        cutoff: float = 10.0,
        mL_ref: float | None = None,
        alpha: float = 0.0,
        data_dir: str | Path = "data",
        tables: Mapping[int, OrbitTable] | None = None,
    ) -> None:
        self.mL = float(mL)
        self.sectors = tuple(sectors)
        self.cutoff = float(cutoff)
        self.alpha = float(alpha)
        self.mL_ref = float(mL_ref) if mL_ref is not None else float(mL)

        if tables is None:
            tables = {
                N: load_orbits(N, emax=cutoff, mL=self.mL_ref, data_dir=data_dir)
                for N in self.sectors
            }
        self.tables: dict[int, OrbitTable] = dict(tables)
        self.G = GMatrix(
            {
                N: GFunction(self.tables[N], mL=self.mL, emax=cutoff, alpha=alpha)
                for N in self.sectors
            }
        )
        self.set_couplings(couplings)

    # -- the condition itself ----------------------------------------------

    def set_couplings(self, couplings: Couplings) -> None:
        self.couplings = couplings
        self.G.set_couplings(couplings)
        self.lam: Array = couplings.matrix(self.sectors)

    def gmatrix(self, E: ArrayLike) -> Array:
        return self.G(E)

    def __call__(self, E: ArrayLike) -> Any:
        """``det[1 - G lambda]``, the pole-free form of the condition."""
        g = self.G(E)
        identity = np.eye(len(self.sectors))
        if g.ndim == 3:
            return np.linalg.det(identity[None, :, :] - g @ self.lam)
        return np.linalg.det(identity - g @ self.lam)

    def determinant(self, E: ArrayLike) -> Any:
        """``det[G^{-1} - lambda]``, as written in eq. (2.14)."""
        g = self.G(E)
        inverse = np.linalg.inv(g)
        lam = self.lam[None, :, :] if g.ndim == 3 else self.lam
        return np.linalg.det(inverse - lam)

    def null_vector(self, E: float) -> Array:
        """The vector ``w`` with ``(1 - lambda G) w = 0`` at a solution.

        If ``v`` spans the kernel of ``G^{-1} - lambda``, then ``w = lambda v =
        G^{-1} v`` spans the kernel of ``1 - lambda G``.  It is ``w``, not ``v``,
        that multiplies the states ``Ghat |u_N>`` in the residue of the resolvent,
        and the pole-free form used here stays accurate next to the poles of
        ``G``, where ``G^{-1}`` is numerically ill-conditioned.
        """
        matrix = np.eye(len(self.sectors)) - self.lam @ self.G(float(E))
        _, _, right = np.linalg.svd(matrix)
        return right[-1]

    def content(self, E: float) -> Array:
        """Relative particle-number content of the finite-volume state at ``E``.

        The pole part of the resummed series of section 2.1 is

            Ghat |u_N'> [lambda (1 - G lambda)^{-1}]_{N'N} <u_N| Ghat ,

        so at a solution the state is ``|psi> ~ sum_N w_N Ghat |u_N>`` with ``w``
        the kernel vector above.  Since ``Ghat`` does not mix particle numbers
        here, the overlap with the ``N``-particle sector is

            <psi| Pi_N |psi> = w_N^2 <u_N| Ghat^2 |u_N> = -w_N^2 dG_{NN}/dE ,

        which gives the four- and two-particle contents of figure 2 of the
        draft.  Section 2.3 is still a placeholder there, so this normalization
        should be revisited once the residues are written out.
        """
        vector = self.null_vector(E)
        weights = np.array(
            [
                vector[position] ** 2 * (-self.G.sectors[N].derivative(float(E)))
                for position, N in enumerate(self.sectors)
            ],
            dtype=float,
        )
        total = weights.sum()
        return weights / total if total != 0 else weights

    # -- levels -------------------------------------------------------------

    @property
    def poles(self) -> Array:
        """Poles of ``G``: the spectrum of ``Ghat``, with disconnected shifts."""
        return self.G.poles

    def free_levels(self, N: int, emin: float, emax: float) -> Array:
        """Non-interacting levels of a sector, without the disconnected shifts."""
        return self.G.sectors[N].unshifted_levels(emin, emax)

    def shifted_levels(self, N: int, emin: float, emax: float) -> Array:
        """Levels of a sector including the disconnected shifts and splittings."""
        return self.G.sectors[N].levels_in(emin, emax)

    def roots(
        self,
        emin: float,
        emax: float,
        n_coarse: int = 2000,
        delta: float = 1e-11,
        verbose: bool = False,
    ) -> Array:
        """Solve the quantization condition in a window."""
        grid = energy_grid(emin, emax, self.poles, n_coarse=n_coarse, delta=delta)
        return find_roots(self, grid, self.poles, verbose=verbose)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"QC(mL={self.mL:g}, sectors={self.sectors}, cutoff={self.cutoff:g}, "
            f"alpha={self.alpha:g}, {self.couplings.label()})"
        )


# ---------------------------------------------------------------------------
# root finding
# ---------------------------------------------------------------------------


def energy_grid(
    emin: float,
    emax: float,
    poles: ArrayLike,
    n_coarse: int = 2000,
    delta: float = 1e-11,
) -> Array:
    """A search grid that isolates every pole inside a narrow bracket.

    Each pole is surrounded by the two points ``pole +- delta``, so that the only
    intervals containing a pole are the tiny ones; every other interval is free
    of poles and a sign change in it is a genuine root.  ``delta`` therefore has
    to be smaller than the smallest level shift one wants to resolve, which for
    a four-particle level is of order ``lambda_{4<-4} / (384 (mL)^9)``.
    """
    poles = np.asarray(poles, dtype=float)
    inside = poles[(poles > emin) & (poles < emax)]
    points = [np.linspace(emin, emax, n_coarse), inside - delta, inside + delta]
    grid = np.unique(np.concatenate(points))
    return grid[(grid >= emin) & (grid <= emax)]


def find_roots(
    func: Callable[[Any], Any],
    grid: ArrayLike,
    poles: ArrayLike | None = None,
    verbose: bool = False,
) -> Array:
    """Bracket and solve for sign changes, skipping brackets that hold a pole."""
    grid = np.asarray(grid, dtype=float)
    values = np.asarray(func(grid), dtype=float)
    poles = np.sort(np.asarray(poles, dtype=float)) if poles is not None else None

    candidates = np.flatnonzero(values[:-1] * values[1:] < 0)
    roots: list[float] = []
    skipped = 0
    iterator = tqdm(candidates, leave=False, desc="roots") if verbose else candidates
    for index in iterator:
        low, high = grid[index], grid[index + 1]
        if poles is not None and poles.size:
            first = np.searchsorted(poles, low, side="right")
            if first < len(poles) and poles[first] < high:
                skipped += 1
                continue
        solution = root_scalar(func, bracket=[low, high], method="brentq")
        roots.append(float(solution.root))
        if verbose:
            iterator.set_description(f"{len(roots)} root(s)")

    if verbose:
        print(f"[info] {len(roots)} roots, {skipped} pole crossings skipped")
    return np.array(sorted(roots), dtype=float)


# ---------------------------------------------------------------------------
# volume scans
# ---------------------------------------------------------------------------


def scan_filename(
    couplings: Couplings,
    mLs: ArrayLike,
    emin: float,
    emax: float,
    cutoff: float,
    alpha: float,
    data_dir: str | os.PathLike[str] = "data",
) -> Path:
    """Stable cache path of a volume scan."""
    mLs = np.asarray(mLs, dtype=float)
    stem = (
        f"spectrum_{couplings.label()}"
        f"_mL{mLs[0]:g}-{mLs[-1]:g}_NL{len(mLs)}"
        f"_E{emin:g}-{emax:g}_EMAX{cutoff:g}_alpha{alpha:g}"
    )
    return Path(data_dir) / f"{stem}.npz"


def scan_volumes(
    mLs: ArrayLike,
    couplings: Couplings,
    emin: float = 1.9,
    emax: float = 6.0,
    cutoff: float = 10.0,
    alpha: float = 0.0,
    sectors: Sequence[int] = SECTORS,
    n_coarse: int = 2000,
    delta: float = 1e-11,
    save: bool = True,
    load: bool = True,
    data_dir: str | os.PathLike[str] = "data",
    verbose: bool = True,
) -> dict[str, Any]:
    """Solve the quantization condition over a range of volumes.

    Returns a dictionary with the roots, the free levels of each sector and the
    levels including the disconnected shifts, one entry per volume.
    """
    mLs = np.asarray(mLs, dtype=float)
    path = scan_filename(couplings, mLs, emin, emax, cutoff, alpha, data_dir)
    if load and path.exists():
        if verbose:
            print(f"[info] loading {path}")
        return load_scan(path)

    tables = {
        N: load_orbits(N, emax=cutoff, mL=float(mLs.max()), data_dir=data_dir)
        for N in sectors
    }

    roots = np.empty(len(mLs), dtype=object)
    free: dict[int, Array] = {N: np.empty(len(mLs), dtype=object) for N in sectors}
    shifted: dict[int, Array] = {N: np.empty(len(mLs), dtype=object) for N in sectors}

    volumes = tqdm(mLs, desc="mL", leave=False) if verbose else mLs
    for index, mL in enumerate(volumes):
        qc = QC(
            couplings,
            mL,
            sectors=sectors,
            cutoff=cutoff,
            mL_ref=float(mLs.max()),
            alpha=alpha,
            tables=tables,
        )
        roots[index] = qc.roots(emin, emax, n_coarse=n_coarse, delta=delta)
        for N in sectors:
            free[N][index] = qc.free_levels(N, emin, emax)
            shifted[N][index] = qc.shifted_levels(N, emin, emax)
        if verbose:
            volumes.set_description(f"mL={mL:.3f}, {len(roots[index])} roots")

    result: dict[str, Any] = {
        "mLs": mLs,
        "roots": roots,
        "sectors": np.array(sectors),
        "emin": emin,
        "emax": emax,
        "cutoff": cutoff,
        "alpha": alpha,
        "couplings": couplings,
    }
    for N in sectors:
        result[f"free{N}"] = free[N]
        result[f"shifted{N}"] = shifted[N]

    if save:
        save_scan(path, result)
        if verbose:
            print(f"[info] saved {path}")
    return result


def save_scan(path: str | os.PathLike[str], result: Mapping[str, Any]) -> Path:
    """Write a scan; the per-volume level lists are ragged, hence object arrays."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = dict(result)
    payload["couplings"] = repr(dict(result["couplings"].values))
    np.savez(path, **payload)
    return path


def load_scan(path: str | os.PathLike[str]) -> dict[str, Any]:
    with np.load(Path(path), allow_pickle=True) as data:
        result = {key: data[key] for key in data.files}
    for key in ("emin", "emax", "cutoff", "alpha"):
        result[key] = float(result[key])
    result["couplings"] = Couplings(eval(str(result["couplings"])))
    return result


# ---------------------------------------------------------------------------
# plotting
# ---------------------------------------------------------------------------


def plot_spectrum(
    result: Mapping[str, Any],
    filename: str | os.PathLike[str] | None = None,
    emin: float | None = None,
    emax: float | None = None,
    figsize: tuple[float, float] = (3.2, 6.0),
    show_free: bool = True,
    legend: bool = True,
    open_pdf: bool = True,
) -> Path:
    """Plot a volume scan: the spectrum of figure 1 of the draft."""
    mLs = np.asarray(result["mLs"], dtype=float)
    sectors = [int(N) for N in np.asarray(result["sectors"]).tolist()]
    emin = float(result["emin"]) if emin is None else emin
    emax = float(result["emax"]) if emax is None else emax

    colours = {2: "tab:red", 4: "tab:blue", 6: "tab:green"}
    _, ax = plt.subplots(1, 1, figsize=figsize)

    for N in sectors:
        colour = colours.get(N, "grey")
        if show_free:
            _plot_level_set(
                ax,
                mLs,
                result[f"free{N}"],
                colour=colour,
                alpha=0.25,
                lw=0.8,
                label=rf"${N}$ particles, free",
            )
        _plot_level_set(
            ax,
            mLs,
            result[f"shifted{N}"],
            colour=colour,
            alpha=0.7,
            lw=0.8,
            linestyle="--",
            label=rf"${N}$ particles, $+\hat{{V}}^{{\rm d}}_L$",
        )

    _plot_level_set(
        ax, mLs, result["roots"], colour="k", alpha=1.0, lw=1.0, label="solutions"
    )

    couplings = result["couplings"]
    title = ", ".join(
        rf"$\lambda_{{{low}\leftarrow{high}}}={value:g}$"
        for (low, high), value in sorted(couplings.values.items())
        if value != 0.0
    )
    ax.set_title(title, size=9)
    ax.set_xlabel(r"$mL$", size=12)
    ax.set_ylabel(r"$E/m$", size=12)
    ax.set_xlim([mLs[0], mLs[-1]])
    ax.set_ylim([emin, emax])
    if legend:
        ax.legend(loc="upper right", fontsize=7)

    if filename is None:
        label = couplings.label()
        filename = f"plots/spectrum_{label}_NL{len(mLs)}.pdf"
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    call_PDF(filename, open=open_pdf)
    return filename


def _plot_level_set(
    ax: Any,
    mLs: Array,
    per_volume: Any,
    colour: str,
    alpha: float,
    lw: float,
    label: str | None = None,
    linestyle: str = "-",
) -> None:
    """Draw one family of levels, connecting them by their order in ``E``."""
    levels = pad_levels(per_volume)
    for column in range(levels.shape[1]):
        ax.plot(
            mLs,
            levels[:, column],
            color=colour,
            alpha=alpha,
            lw=lw,
            ls=linestyle,
            label=label if column == 0 else None,
        )


def pad_levels(per_volume: Any) -> Array:
    """Stack per-volume level lists into an ``(n_volumes, n_levels)`` array."""
    rows = [np.atleast_1d(np.asarray(row, dtype=float)) for row in per_volume]
    width = max((len(row) for row in rows), default=0)
    out = np.full((len(rows), width), np.nan, dtype=float)
    for index, row in enumerate(rows):
        out[index, : len(row)] = row
    return out


def get_energy_levels(
    Ls: ArrayLike, roots: Any, correct_skips: bool = False
) -> dict[int, Array]:
    """Backwards-compatible accessor returning one array per level index."""
    del correct_skips  # Levels are now labelled by their order in E.
    levels = pad_levels(roots)
    del Ls
    return {index: levels[:, index] for index in range(levels.shape[1])}


# ---------------------------------------------------------------------------


def main() -> None:
    mLs = np.linspace(4.0, 7.0, 31)
    couplings = Couplings.two_four(l22=20.0, l24=5000.0, l44=1.0, l33=0.0)
    result = scan_volumes(
        mLs,
        couplings,
        emin=1.9,
        emax=6.0,
        cutoff=10.0,
        load=True,
        save=True,
    )
    plot_spectrum(result, emin=1.95, emax=6.0, open_pdf=False)


if __name__ == "__main__":
    main()
