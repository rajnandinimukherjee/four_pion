"""The finite-volume functions ``G_{N'N}(E, L)`` of the quantization condition.

Section 2.2 of the draft writes the quantization condition as

    det[ G(E, L)^{-1} - lambda ] = 0 ,

with ``lambda`` the matrix of effective couplings ``lambda_{N'<-N}`` and ``G``
the matrix of finite-volume functions

    G_{N'N}(E, L) = sum_mu' sum_mu u_mu' <b_mu'| Ghat(E) |b_mu> u_mu ,
    u_mu = L^{-3(N-1)/2} sqrt( d_mu / (N_mu Omega_mu) ) .

At leading order in the disconnected interactions, ``Ghat`` follows from
degenerate perturbation theory inside each non-interacting level, eq. (2.16),

    G_{NN}(E, L) = sum_levels sum_{mu, mu' in level}
                   u_mu' [ (E - E_level) 1 - V^d_level ]^{-1}_{mu' mu} u_mu .

Diagonalizing ``V^d_level`` turns this into a sum over simple poles,

    G_{NN}(E, L) = sum_level sum_n  |<u|psi_n>|^2 / (E - E_level - v_n) ,

whose pole positions are the non-interacting levels shifted and split by the
disconnected sub-processes, and whose residues say how strongly each split level
couples to the fully connected interactions.  That representation is what this
module builds and evaluates.

The orbits, the level assignment and the volume-independent part of ``V^d`` all
come from a table written by ``orbits.py``; nothing here re-derives them.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

from orbits import OrbitTable, load_orbits, omega

Array = NDArray[Any]


# ---------------------------------------------------------------------------
# couplings
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Couplings:
    """The effective couplings ``lambda_{N'<-N}`` of eq. (A.14).

    Entries are symmetric, ``lambda_{N'<-N} = lambda_{N<-N'}``, and absent
    entries vanish.  The same ``lambda_{2<-2}`` that appears as a fully
    connected interaction in the two-particle sector also drives the
    disconnected ``2 -> 2`` sub-processes in the four-particle sector, so a
    single object carries both roles.
    """

    values: Mapping[tuple[int, int], float] = field(default_factory=dict)

    def __post_init__(self) -> None:
        normalized = {
            (min(key), max(key)): float(value) for key, value in self.values.items()
        }
        object.__setattr__(self, "values", normalized)

    def __getitem__(self, key: tuple[int, int]) -> float:
        return self.values.get((min(key), max(key)), 0.0)

    @classmethod
    def two_four(
        cls, l22: float, l24: float, l44: float, l33: float = 0.0
    ) -> "Couplings":
        """The couplings of section 3.1: two and four identical scalars."""
        return cls({(2, 2): l22, (2, 4): l24, (4, 4): l44, (3, 3): l33})

    @classmethod
    def parse(cls, values: Sequence[str | float]) -> "Couplings":
        """Build from the command-line triple/quadruple ``l22 l24 l44 [l33]``."""
        numbers = [float(value) for value in values]
        if len(numbers) == 3:
            numbers.append(0.0)
        if len(numbers) != 4:
            raise ValueError("expected l22 l24 l44 [l33]")
        return cls.two_four(*numbers)

    def matrix(self, sectors: Sequence[int]) -> Array:
        """The connected coupling matrix in the space of particle numbers."""
        return np.array(
            [[self[(row, column)] for column in sectors] for row in sectors],
            dtype=float,
        )

    def label(self) -> str:
        """Short, filename-safe tag listing the non-trivial couplings."""
        parts = [
            f"{low}{high}_{value:g}"
            for (low, high), value in sorted(self.values.items())
            if value != 0.0
        ]
        return "l" + "_l".join(parts) if parts else "free"


# ---------------------------------------------------------------------------
# one particle-number sector
# ---------------------------------------------------------------------------


class GFunction:
    """``G_{NN}(E, L)`` for a single particle-number sector.

    Parameters
    ----------
    orbits:
        Table written by ``orbits.py`` for this sector.
    mL:
        Box size in units of the mass.
    emax:
        Energy cutoff retained in the sum over levels.  The states above it are
        smooth in ``E`` and are absorbed into the couplings (appendix B), so
        this is a regulator and the couplings depend on it.
    alpha:
        Optional smooth damping ``exp(alpha (E - E_pole))`` applied to each
        pole, softening the hard cutoff as discussed in appendix B.
    """

    def __init__(
        self,
        orbits: OrbitTable,
        mL: float,
        emax: float | None = None,
        alpha: float = 0.0,
    ) -> None:
        self.orbits = orbits
        self.N: int = orbits.N
        self.mL = float(mL)
        self.tpiL: float = 2 * np.pi / self.mL
        self.emax = emax
        self.alpha = float(alpha)

        self._select_orbits()
        self._build_u()
        self._build_levels()
        self._build_disconnected()

        self.couplings: Couplings = Couplings()
        self.poles: Array = np.zeros(0)
        self.residues: Array = np.zeros(0)
        self.set_couplings(Couplings())

    # -- construction -------------------------------------------------------

    def _select_orbits(self) -> None:
        energies = self.orbits.energies(self.mL)
        keep = np.ones(len(energies), dtype=bool) if self.emax is None else (
            energies <= self.emax
        )
        self.index: Array = np.flatnonzero(keep)
        self.energies: Array = energies[keep]
        self.sizes: Array = self.orbits.sizes[keep]
        self.occupations: Array = self.orbits.occupations[keep]
        self.nsq: Array = self.orbits.nsq[keep]
        self._position = -np.ones(len(energies), dtype=np.int64)
        self._position[self.index] = np.arange(len(self.index))

    def _build_u(self) -> None:
        """The vertex factors ``u_mu`` of eq. (2.11)."""
        big_omega = np.prod(2.0 * omega(self.nsq, self.tpiL), axis=-1)
        volume = self.mL ** (3 * (self.N - 1) / 2)
        self.u: Array = np.sqrt(self.sizes / (self.occupations * big_omega)) / volume

    def _build_levels(self) -> None:
        """Group the retained orbits into degenerate non-interacting levels."""
        labels = self.orbits.levels[self.index]
        order = np.argsort(labels, kind="stable")
        sorted_labels = labels[order]
        boundaries = np.flatnonzero(np.diff(sorted_labels)) + 1
        self.level_members: list[Array] = [
            np.sort(block) for block in np.split(order, boundaries)
        ]
        self.level_energies: Array = np.array(
            [self.energies[members[0]] for members in self.level_members], dtype=float
        )

    def _build_disconnected(self) -> None:
        """Assemble the ``V^d`` block of every level, per sub-process size ``r``.

        The blocks are linear in the couplings ``lambda_{r<-r}``, so the
        volume-dependent kinematic part is built once here and contracted with
        the couplings in ``set_couplings``.
        """
        self.level_blocks: list[dict[int, Array]] = [
            {} for _ in self.level_members
        ]
        if len(self.orbits.vd_weight) == 0:
            return

        level_of_orbit = np.empty(len(self.index), dtype=np.int64)
        slot_of_orbit = np.empty(len(self.index), dtype=np.int64)
        for level, members in enumerate(self.level_members):
            level_of_orbit[members] = level
            slot_of_orbit[members] = np.arange(len(members))

        rows_out = self._position[self.orbits.vd_out]
        rows_in = self._position[self.orbits.vd_in]
        keep = (rows_out >= 0) & (rows_in >= 0)
        rows_out, rows_in = rows_out[keep], rows_in[keep]
        r_values = self.orbits.vd_r[keep]
        weights = self.orbits.vd_weight[keep]
        scatterer_nsq = self.orbits.vd_nsq[keep]

        # Omega_s = prod_{i in s} 2 omega_i over the scatterers only; the padding
        # entries (-1) are dropped by masking them to a unit factor.
        mask = scatterer_nsq >= 0
        factors = np.where(mask, 2.0 * omega(np.maximum(scatterer_nsq, 0), self.tpiL), 1.0)
        big_omega_s = np.prod(factors, axis=-1)
        volume = self.mL ** (3 * (r_values - 1))
        values = weights / (volume * big_omega_s)

        for row_out, row_in, r, value in zip(
            rows_out.tolist(), rows_in.tolist(), r_values.tolist(), values.tolist()
        ):
            level = int(level_of_orbit[row_in])
            assert level == int(level_of_orbit[row_out]), (
                "a disconnected contribution left its degenerate level"
            )
            size = len(self.level_members[level])
            block = self.level_blocks[level].setdefault(
                int(r), np.zeros((size, size), dtype=float)
            )
            block[int(slot_of_orbit[row_out]), int(slot_of_orbit[row_in])] += value

    # -- couplings ----------------------------------------------------------

    def set_couplings(self, couplings: Couplings) -> None:
        """Diagonalize ``V^d`` level by level and store the poles and residues."""
        self.couplings = couplings
        poles: list[float] = []
        residues: list[float] = []
        self.level_shifts: list[Array] = []
        self.level_vectors: list[Array] = []

        for level, members in enumerate(self.level_members):
            u = self.u[members]
            blocks = self.level_blocks[level]
            size = len(members)
            potential = np.zeros((size, size), dtype=float)
            for r, block in blocks.items():
                potential = potential + couplings[(r, r)] * block

            if size == 1:
                shifts = np.array([potential[0, 0]])
                vectors = np.ones((1, 1))
            else:
                asymmetry = np.max(np.abs(potential - potential.T))
                scale = max(np.max(np.abs(potential)), 1e-300)
                if asymmetry > 1e-8 * scale:
                    raise ValueError(
                        f"V^d block of level {level} is not symmetric "
                        f"(asymmetry {asymmetry:.3e})"
                    )
                potential = 0.5 * (potential + potential.T)
                shifts, vectors = np.linalg.eigh(potential)

            overlaps = vectors.T @ u
            self.level_shifts.append(shifts)
            self.level_vectors.append(vectors)
            poles.extend((self.level_energies[level] + shifts).tolist())
            residues.extend((overlaps**2).tolist())

        order = np.argsort(poles)
        self.poles = np.asarray(poles, dtype=float)[order]
        self.residues = np.asarray(residues, dtype=float)[order]

    # -- evaluation ---------------------------------------------------------

    def __call__(self, E: ArrayLike) -> Any:
        energies = np.atleast_1d(np.asarray(E, dtype=float))
        gaps = energies[:, None] - self.poles[None, :]
        damping = np.exp(self.alpha * gaps) if self.alpha else 1.0
        result = np.sum(damping * self.residues[None, :] / gaps, axis=-1)
        return result[0] if np.size(E) == 1 else result

    def derivative(self, E: ArrayLike) -> Any:
        """``dG/dE``.  Its magnitude sets the residue of a finite-volume pole."""
        energies = np.atleast_1d(np.asarray(E, dtype=float))
        gaps = energies[:, None] - self.poles[None, :]
        damping = np.exp(self.alpha * gaps) if self.alpha else 1.0
        terms = damping * self.residues[None, :] * (self.alpha / gaps - 1.0 / gaps**2)
        result = np.sum(terms, axis=-1)
        return result[0] if np.size(E) == 1 else result

    def levels_in(self, emin: float, emax: float) -> Array:
        """The poles of ``G`` inside a window: the interaction-free spectrum."""
        inside = (self.poles > emin) & (self.poles < emax)
        return self.poles[inside]

    def unshifted_levels(self, emin: float, emax: float) -> Array:
        """The non-interacting levels, without the disconnected shifts."""
        energies = np.unique(self.level_energies)
        return energies[(energies > emin) & (energies < emax)]

    def __repr__(self) -> str:  # pragma: no cover - debugging aid
        return (
            f"GFunction(N={self.N}, mL={self.mL:g}, orbits={len(self.index)}, "
            f"levels={len(self.level_members)}, poles={len(self.poles)})"
        )


# ---------------------------------------------------------------------------
# the matrix in the space of particle numbers
# ---------------------------------------------------------------------------


class GMatrix:
    """``G(E, L)`` as a matrix in the particle-number sectors.

    Within a window in which the disconnected sub-processes cannot change the
    particle number, ``G`` is diagonal (section 3.1); the off-diagonal entries
    would come from levels containing orbits of different particle number, which
    only occur at isolated volumes.
    """

    def __init__(self, sectors: Mapping[int, GFunction]) -> None:
        self.sectors: dict[int, GFunction] = dict(sectors)
        self.numbers: list[int] = sorted(self.sectors)

    @classmethod
    def load(
        cls,
        sectors: Iterable[int],
        mL: float,
        emax: float,
        mL_ref: float | None = None,
        alpha: float = 0.0,
        data_dir: str | Path = "data",
        paths: Mapping[int, str | Path] | None = None,
    ) -> "GMatrix":
        """Load one orbit table per sector and build the finite-volume functions."""
        paths = dict(paths or {})
        functions = {}
        for N in sectors:
            table = load_orbits(
                N,
                emax=emax,
                mL=mL_ref,
                data_dir=data_dir,
                path=paths.get(N),
            )
            functions[N] = GFunction(table, mL=mL, emax=emax, alpha=alpha)
        return cls(functions)

    def set_couplings(self, couplings: Couplings) -> None:
        for function in self.sectors.values():
            function.set_couplings(couplings)

    def __call__(self, E: ArrayLike) -> Array:
        energies = np.atleast_1d(np.asarray(E, dtype=float))
        size = len(self.numbers)
        result = np.zeros((len(energies), size, size), dtype=float)
        for position, N in enumerate(self.numbers):
            result[:, position, position] = self.sectors[N](energies)
        return result[0] if np.size(E) == 1 else result

    @property
    def poles(self) -> Array:
        """All poles of ``G``, i.e. the levels of ``Ghat`` in every sector."""
        return np.sort(
            np.concatenate([function.poles for function in self.sectors.values()])
        )

    def poles_in(self, emin: float, emax: float) -> Array:
        poles = self.poles
        return poles[(poles > emin) & (poles < emax)]
