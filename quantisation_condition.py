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

The root search uses the form of the condition

    f(E) = det[ 1 - G(E, L) lambda ] ,

which has the same zeros wherever ``G`` is invertible.  The default is
``root_finding.py``: whichever cutoff shape is used, ``G`` is a sum of simple
poles with non-negative residues, so the roots between two poles can be counted
exactly and each is solved for relative to its nearest pole.  With
``method="grid"`` a search grid is built around the known poles of ``G`` and the
brackets that straddle a pole are discarded instead; that search misses roots
closer than ``delta`` to a pole and pairs of roots inside one grid cell.

The sums inside ``G`` are regulated by a smooth cutoff by default, see
``gfunctions.Cutoff``; ``shape="hard"`` recovers the step of the notes.

Usage
-----
    python orbits.py --N 2 --emax 10 --mL 7
    python orbits.py --N 4 --emax 10 --mL 7
    python quantisation_condition.py
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray
from scipy.optimize import root_scalar
from tqdm import tqdm

from gfunctions import Couplings, Cutoff, GFunction, GMatrix
from orbits import OrbitTable, load_orbits
from plot_settings import plotparams
from utils import call_PDF

Array = NDArray[Any]

EPS = np.finfo(float).eps

plt.rcParams.update(plotparams)

SECTORS: tuple[int, ...] = (2, 4)
ROOT_METHODS: tuple[str, ...] = ("auto", "count", "grid")


def resolve_root_method(method: str) -> str:
    """Resolve ``auto`` to root counting, which is valid for either cutoff.

    Both cutoff shapes weight each level by a non-negative constant, so ``G``
    is a sum of simple poles with non-negative residues and the counting of
    ``root_finding.py`` applies.  ``grid`` remains available as a cross-check.
    """
    if method not in ROOT_METHODS:
        raise ValueError(f"unknown root method {method!r}; use one of {ROOT_METHODS}")
    return "count" if method == "auto" else method


@dataclass(frozen=True)
class EigenState:
    """The null branch of the quantization condition at one energy.

    The eigenvalue method of Hansen and Sharpe, arXiv:1204.0826 section V,
    reads the channel content of a finite-volume state off the eigenvector of
    the quantization-condition matrix with vanishing eigenvalue, their
    eqs. (115)-(116).  This holds the result of that diagonalization for the
    particle-number sectors of section 2.3 of the draft.

    Attributes
    ----------
    energy:
        The energy at which the matrix was diagonalized; a solution of the
        quantization condition if ``eigenvalue`` is zero to working accuracy.
    sectors:
        The particle numbers, in the order of the components below.
    amplitudes:
        The unit-normalized null eigenvector ``c``, i.e. the amplitudes of the
        state in an orthonormal basis of sector states, ``c_N = w_N
        sqrt(-dG_NN/dE)``.  Its squares are the particle-number content.
    weights:
        The same vector divided by the sector norms and renormalized: the
        kernel vector ``w`` of ``1 - lambda G``, as returned by ``null_vector``
        up to sign and scale.
    eigenvalue:
        The eigenvalue of the null branch.  It vanishes exactly at a solution,
        so its size is a residual: how far ``energy`` is from a root.
    gap:
        The magnitude of the next eigenvalue.  A simple zero has ``gap`` of
        order ``spread``; ``gap`` comparable to ``eigenvalue`` signals a
        degenerate solution, where the content is not given by one vector.
    spread:
        The largest eigenvalue in magnitude, the scale against which
        ``eigenvalue`` and ``gap`` are to be read.
    """

    energy: float
    sectors: tuple[int, ...]
    amplitudes: Array
    weights: Array
    eigenvalue: float
    gap: float
    spread: float

    @property
    def content(self) -> Array:
        """``P_N = c_N^2``, the particle-number content, summing to one."""
        squares = np.asarray(self.amplitudes, dtype=float) ** 2
        total = squares.sum()
        return squares / total if total != 0.0 else squares

    @property
    def ratio(self) -> float:
        """``c_last / c_first``, the analogue of ``c_K / c_pi`` of their eq. (115)."""
        first = float(self.amplitudes[0])
        return float(self.amplitudes[-1]) / first if first != 0.0 else np.inf

    def is_simple(self, tolerance: float = 1e-6) -> bool:
        """Whether the zero is simple, so that the content is well defined."""
        scale = self.spread if self.spread > 0.0 else 1.0
        return bool(self.gap > tolerance * scale and abs(self.eigenvalue) < self.gap)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        content = ", ".join(
            f"P{N}={value:.4f}" for N, value in zip(self.sectors, self.content)
        )
        return (
            f"EigenState(E={self.energy:.8f}, {content}, "
            f"eigenvalue={self.eigenvalue:.2e}, gap={self.gap:.2e})"
        )


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
        The regulator on the sums over levels inside ``G``: a ``Cutoff``, or a
        bare ``E_max`` to be combined with ``shape`` and ``width``.  The states
        above it are smooth and are absorbed into the couplings, so the
        couplings depend on it.
    shape, width:
        Used only when ``cutoff`` is a bare number.  ``shape="smooth"`` (the
        default) tapers the sums to zero over the last ``width`` in energy;
        ``shape="hard"`` is the step of the notes.
    mL_ref:
        Reference ``mL`` of the saved orbit tables.  It must be at least the
        largest ``mL`` used, so that the tables cover every volume in a scan.
    """

    def __init__(
        self,
        couplings: Couplings,
        mL: float,
        sectors: Sequence[int] = SECTORS,
        cutoff: Cutoff | float = 10.0,
        shape: str = "smooth",
        width: float = 2.0,
        mL_ref: float | None = None,
        data_dir: str | Path = "data",
        tables: Mapping[int, OrbitTable] | None = None,
    ) -> None:
        self.mL = float(mL)
        self.sectors = tuple(sectors)
        self.cutoff: Cutoff = Cutoff.parse(cutoff, shape, width)
        self.mL_ref = float(mL_ref) if mL_ref is not None else float(mL)

        if tables is None:
            tables = {
                N: load_orbits(
                    N, emax=self.cutoff.support, mL=self.mL_ref, data_dir=data_dir
                )
                for N in self.sectors
            }
        self.tables: dict[int, OrbitTable] = dict(tables)
        self.G = GMatrix(
            {
                N: GFunction(self.tables[N], mL=self.mL, cutoff=self.cutoff)
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

    # -- particle-number content: the eigenvalue method ----------------------

    def sector_norms(self, E: float) -> Array:
        """``D_N = || Ghat |u_N> || = sqrt(-dG_NN/dE)``, one per sector.

        These are the kinematic normalizations of the sector states, and play
        the role of the phase-space matrix ``P`` of Hansen and Sharpe, eq. (46)
        of arXiv:1204.0826.  They are strictly positive away from the poles of
        ``G``, since ``-dG_NN/dE`` is a sum of squares over the levels.
        """
        return np.array(
            [
                np.sqrt(-float(self.G.sectors[N].derivative(float(E))))
                for N in self.sectors
            ],
            dtype=float,
        )

    def eigen_matrix(self, E: float) -> Array:
        """The symmetric matrix whose null eigenvector is the amplitude vector.

        Following the eigenvalue method of Hansen and Sharpe, arXiv:1204.0826,
        section V, the quantization condition is written as a symmetric matrix
        whose kinematic normalization has been stripped off, and the relative
        content of the channels is read off from its zero eigenvector, their
        eqs. (115)-(116).

        Here the kinematics-free form of the condition is

            Y(E) = G(E, L) - lambda^{-1} ,

        whose kernel is the vector ``w`` of section 2.3: ``(G - lambda^{-1}) w
        = 0`` is ``(1 - lambda G) w = 0`` multiplied by ``lambda^{-1}``.  The
        normalized form is then

            X(E) = D^{-1} Y(E) D^{-1} ,   D = diag(sqrt(-dG_NN/dE)) ,

        the analogue of ``X = P^{-1} Y P^{-1}``, their eq. (99).  Its null
        eigenvector is ``c = D w``, whose components are the amplitudes of the
        finite-volume state in an orthonormal basis of sector states, so that
        ``P_N = c_N^2`` with ``c`` unit normalized.

        Two properties make this the form to diagonalize numerically.  It is
        real symmetric, so ``eigh`` returns real eigenvalues and an orthonormal
        eigenbasis.  And it is bounded near the poles of ``G``: there
        ``G_NN ~ r/(E-E_p)`` while ``D_N ~ sqrt(r)/|E-E_p|``, so the division by
        ``D`` cancels the pole, whereas ``1 - lambda G`` itself diverges.  The
        content is largest exactly next to those poles.
        """
        energy = float(E)
        g = np.atleast_2d(self.G(energy))
        inverse_lambda = self._inverse_couplings()
        scale = 1.0 / self.sector_norms(energy)
        matrix = (g - inverse_lambda) * scale[:, None] * scale[None, :]
        return 0.5 * (matrix + matrix.T)

    def _inverse_couplings(self) -> Array:
        """``lambda^{-1}``, symmetrized, with a check that ``lambda`` inverts.

        The eigenvalue form needs the inverse: it is the coupling matrix that
        carries the mixing between sectors, and a singular ``lambda`` has a
        direction that decouples from ``G`` altogether.
        """
        lam = self.lam
        condition = np.linalg.cond(lam)
        if not np.isfinite(condition) or condition > 1.0 / np.sqrt(EPS):
            raise np.linalg.LinAlgError(
                f"the coupling matrix is singular (cond = {condition:.3e}); "
                "the eigenvalue form of the content needs lambda^{-1}. "
                "Use QC.content(E, method='svd') instead."
            )
        inverse = np.linalg.inv(lam)
        return 0.5 * (inverse + inverse.T)

    def eigen_state(self, E: float) -> "EigenState":
        """Diagonalize ``eigen_matrix`` and return the null branch at ``E``.

        At a solution of the quantization condition one eigenvalue of ``X(E)``
        vanishes; its eigenvector is the amplitude vector ``c``.  Away from a
        solution no eigenvalue vanishes, and the smallest one in magnitude
        measures how far ``E`` is from a root, in units set by the spectrum of
        ``X``.  The gap to the next eigenvalue is the diagnostic for the
        assumption of section 2.3 that the zero is simple: two small
        eigenvalues mean a degenerate solution, for which the residue of the
        resolvent is not rank one and the content is not defined by a single
        vector.
        """
        energy = float(E)
        values, vectors = np.linalg.eigh(self.eigen_matrix(energy))
        order = np.argsort(np.abs(values))
        null = int(order[0])
        amplitudes = np.asarray(vectors[:, null], dtype=float)
        # fix the sign convention: the largest component is positive
        if amplitudes[int(np.argmax(np.abs(amplitudes)))] < 0.0:
            amplitudes = -amplitudes
        scale = 1.0 / self.sector_norms(energy)
        weights = amplitudes * scale
        norm = np.linalg.norm(weights)
        spread = float(np.max(np.abs(values)))
        return EigenState(
            energy=energy,
            sectors=self.sectors,
            amplitudes=amplitudes,
            weights=weights / norm if norm != 0.0 else weights,
            eigenvalue=float(values[null]),
            gap=float(np.abs(values[int(order[1])])) if len(values) > 1 else np.inf,
            spread=spread,
        )

    def content(self, E: float, method: str = "eigen") -> Array:
        """Particle-number content of the finite-volume state at ``E``.

        The pole part of the resummed series of section 2.1 is

            Ghat |u_N'> [lambda (1 - G lambda)^{-1}]_{N'N} <u_N| Ghat ,

        so at a solution the state is ``|psi> ~ sum_N w_N Ghat |u_N>``.  Since
        ``Ghat`` does not mix particle numbers here, the overlap with the
        ``N``-particle sector is

            <psi| Pi_N |psi> = w_N^2 <u_N| Ghat^2 |u_N> = -w_N^2 dG_NN/dE ,

        which gives the four- and two-particle contents of figure 2 of the
        draft.  The amplitudes ``c_N = w_N sqrt(-dG_NN/dE)`` are therefore the
        natural objects, and ``method="eigen"`` (the default) obtains them in
        one step as the null eigenvector of the symmetric matrix of
        ``eigen_matrix``, the eigenvalue method of Hansen and Sharpe,
        arXiv:1204.0826, eqs. (115)-(116).  Then ``P_N = c_N^2``.

        ``method="svd"`` is the earlier route, kept as a cross-check: it takes
        the smallest right singular vector of the non-symmetric ``1 - lambda
        G``, giving ``w`` directly, and rescales by ``-dG_NN/dE`` afterwards.
        The two agree analytically; the eigenvalue form is better conditioned
        next to the poles of ``G``, where the rescaling in the ``svd`` route
        multiplies a small number by a very large one.
        """
        if method == "eigen":
            return self.eigen_state(E).content
        if method == "svd":
            return self.content_svd(E)
        raise ValueError(f"unknown content method {method!r}; use 'eigen' or 'svd'")

    def content_svd(self, E: float) -> Array:
        """``content`` by the singular-vector route, kept as a cross-check."""
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

    def null_vector(self, E: float) -> Array:
        """The vector ``w`` with ``(1 - lambda G) w = 0`` at a solution.

        If ``v`` spans the kernel of ``G^{-1} - lambda``, then ``w = lambda v =
        G^{-1} v`` spans the kernel of ``1 - lambda G``.  It is ``w``, not ``v``,
        that multiplies the states ``Ghat |u_N>`` in the residue of the resolvent,
        and the pole-free form used here stays accurate next to the poles of
        ``G``, where ``G^{-1}`` is numerically ill-conditioned.

        This is the input to ``content_svd``; the default ``content`` route gets
        the same direction from the symmetric eigenproblem of ``eigen_matrix``,
        already rescaled into amplitudes.
        """
        matrix = np.eye(len(self.sectors)) - self.lam @ self.G(float(E))
        _, _, right = np.linalg.svd(matrix)
        return right[-1]

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
        method: str = "auto",
        n_coarse: int = 2000,
        delta: float = 1e-11,
        verbose: bool = False,
    ) -> Array:
        """Solve the quantization condition in a window.

        ``method`` is ``"count"`` (``root_finding.py``, exact root counting),
        ``"grid"`` (the grid search; ``n_coarse`` and ``delta`` only apply to
        it), or ``"auto"``, which is ``count``.
        """
        if resolve_root_method(method) == "count":
            from root_finding import roots_newton

            return roots_newton(self, emin, emax)
        grid = energy_grid(emin, emax, self.poles, n_coarse=n_coarse, delta=delta)
        return find_roots(self, grid, self.poles, verbose=verbose)

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"QC(mL={self.mL:g}, sectors={self.sectors}, "
            f"{self.cutoff.label()}, {self.couplings.label()})"
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
        solution = root_scalar(
            func, bracket=[low, high], method="brentq", xtol=1e-300, rtol=4 * EPS
        )
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
    cutoff: Cutoff,
    data_dir: str | os.PathLike[str] = "data",
    method: str = "count",
) -> Path:
    """Stable cache path of a volume scan; ``method`` is the resolved root method."""
    mLs = np.asarray(mLs, dtype=float)
    stem = (
        f"spectrum_{couplings.label()}"
        f"_mL{mLs[0]:g}-{mLs[-1]:g}_NL{len(mLs)}"
        f"_E{emin:g}-{emax:g}_{cutoff.label()}_{method}"
    )
    return Path(data_dir) / f"{stem}.npz"


def scan_volumes(
    mLs: ArrayLike,
    couplings: Couplings,
    emin: float = 1.9,
    emax: float = 6.0,
    cutoff: Cutoff | float = 10.0,
    shape: str = "smooth",
    width: float = 2.0,
    sectors: Sequence[int] = SECTORS,
    method: str = "auto",
    n_coarse: int = 2000,
    delta: float = 1e-11,
    save: bool = True,
    load: bool = True,
    data_dir: str | os.PathLike[str] = "data",
    verbose: bool = True,
) -> dict[str, Any]:
    """Solve the quantization condition over a range of volumes.

    Returns a dictionary with the roots, the free levels of each sector and the
    levels including the disconnected shifts, one entry per volume.  ``method``
    is passed to ``QC.roots``.
    """
    mLs = np.asarray(mLs, dtype=float)
    cutoff = Cutoff.parse(cutoff, shape, width)
    resolved = resolve_root_method(method)
    path = scan_filename(couplings, mLs, emin, emax, cutoff, data_dir, resolved)
    if load and path.exists():
        if verbose:
            print(f"[info] loading {path}")
        return load_scan(path)

    tables = {
        N: load_orbits(
            N, emax=cutoff.support, mL=float(mLs.max()), data_dir=data_dir
        )
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
            tables=tables,
        )
        roots[index] = qc.roots(
            emin, emax, method=resolved, n_coarse=n_coarse, delta=delta
        )
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
        "cutoff": cutoff.emax,
        "shape": cutoff.shape,
        "width": cutoff.width,
        "method": resolved,
        "couplings": couplings,
    }
    for N in sectors:
        result[f"free{N}"] = free[N]
        result[f"shifted{N}"] = shifted[N]

    if save:
        save_scan(path, result)
        if verbose:
            print(f"[info] saved {path}")
    # Kept out of the file, where the cutoff is stored as its three fields.
    result["cutoff_object"] = cutoff
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
    for key in ("emin", "emax", "cutoff", "width"):
        if key in result:
            result[key] = float(result[key])
    for key in ("method", "shape"):
        if key in result:
            result[key] = str(result[key])
    result["cutoff_object"] = Cutoff(
        emax=result["cutoff"],
        shape=result.get("shape", "hard"),
        width=result.get("width", 2.0),
    )
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
