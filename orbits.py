"""Generation of the finite-volume ``A_1^+`` basis: ``O_h`` orbit representatives.

This script builds, for an arbitrary particle number ``N``, the objects of
appendix A.3 of the draft:

* the set ``B_N(0)`` of distinct ``N``-particle momentum configurations with
  vanishing total momentum, enumerated up to a momentum or energy cutoff,
* its partition into orbits ``O_mu`` of the octahedral group with parity, with
  one representative configuration per orbit, the orbit size ``d_mu`` and the
  occupation factor ``N_mu = prod_alpha n_alpha!``,
* the grouping of orbits into degenerate non-interacting levels, and
* the volume-independent part of the disconnected matrix elements
  ``<b_mu'|V^d_L(0)|b_mu>`` of eq. (A.49).

Everything stored here is independent of ``L`` and of the couplings: the
analysis script (``quantisation_condition.py``) turns it into the finite-volume
functions ``G_{N'N}(E, L)`` at run time.

Conventions follow the draft: ``m = 1``, momenta are written as integer vectors
``n`` with ``k = (2 pi / L) n``, and ``omega_n = sqrt(1 + n^2 (2 pi / mL)^2)``.

Usage
-----
    python orbits.py --N 4 --emax 10 --mL 7
    python orbits.py --N 2 --emax 10 --mL 7

writes ``data/orbits_N4_EMAX10_mL7.npz`` and ``data/orbits_N2_EMAX10_mL7.npz``.
"""

from __future__ import annotations

import argparse
import itertools
import math
import re
from collections import Counter
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any, Iterator, Sequence

import numpy as np
from numpy.typing import NDArray

Array = NDArray[Any]

# Maximum size of the intermediate cartesian product used when solving for the
# outgoing scatterer configurations of a disconnected sub-process.
MAX_PRODUCT = 4_000_000


# ---------------------------------------------------------------------------
# the octahedral group with parity
# ---------------------------------------------------------------------------


@lru_cache(maxsize=1)
def oh_group() -> Array:
    """Return the 48 signed permutation matrices of ``O_h = O x Z_2``."""
    elements = []
    for perm in itertools.permutations(range(3)):
        for signs in itertools.product((1, -1), repeat=3):
            g = np.zeros((3, 3), dtype=np.int64)
            for row, (column, sign) in enumerate(zip(perm, signs)):
                g[row, column] = sign
            elements.append(g)
    group = np.array(elements, dtype=np.int64)
    assert group.shape == (48, 3, 3)
    return group


# ---------------------------------------------------------------------------
# integer momentum shells
# ---------------------------------------------------------------------------


@lru_cache(maxsize=4096)
def _shell_cached(nsq: int) -> tuple[tuple[int, int, int], ...]:
    if nsq < 0:
        return ()
    radius = int(math.isqrt(nsq))
    axis = np.arange(-radius, radius + 1)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    grid = grid.reshape(-1, 3)
    norms = np.einsum("ij,ij->i", grid, grid)
    return tuple(map(tuple, grid[norms == nsq].tolist()))


def shell(nsq: int) -> Array:
    """Return the integer vectors ``n`` with ``n.n == nsq`` as an ``(S, 3)`` array."""
    return np.asarray(_shell_cached(int(nsq)), dtype=np.int64).reshape(-1, 3)


def omega(nsq: Array | float, tpiL: float) -> Any:
    """Single-particle energy ``sqrt(m^2 + k^2)`` in units of ``m``."""
    return np.sqrt(1.0 + np.asarray(nsq, dtype=float) * tpiL**2)


def candidate_vectors(nsq_max: int) -> tuple[Array, Array]:
    """Integer vectors with ``n.n <= nsq_max``, ordered by ``(n.n, lexicographic)``.

    The ordering is the total order ``<`` used to enumerate each configuration
    exactly once, as a non-decreasing sequence of vectors.  Sorting by ``n.n``
    first makes the energy cutoff monotonic along the enumeration, which is what
    allows the backtracking search to break out of its loops.
    """
    axis = np.arange(-nsq_max, nsq_max + 1)
    radius = int(math.isqrt(nsq_max))
    axis = np.arange(-radius, radius + 1)
    grid = np.stack(np.meshgrid(axis, axis, axis, indexing="ij"), axis=-1)
    grid = grid.reshape(-1, 3)
    norms = np.einsum("ij,ij->i", grid, grid)
    keep = norms <= nsq_max
    grid, norms = grid[keep], norms[keep]
    order = np.lexsort((grid[:, 2], grid[:, 1], grid[:, 0], norms))
    return grid[order], norms[order]


# ---------------------------------------------------------------------------
# configuration encoding
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Encoder:
    """Injective encoding of a momentum configuration into a single integer.

    A vector with components in ``[-B, B]`` is mapped to a digit in base
    ``2B + 1``; the sorted digits of a configuration are then read as an integer
    in base ``(2B + 1)^3``.  Two configurations have the same code if and only if
    they are the same multiset of momenta, so the minimum of the codes over the
    48 group images is a canonical label for the orbit.
    """

    bound: int
    size: int

    @property
    def base(self) -> int:
        return (2 * self.bound + 1) ** 3

    @property
    def dtype(self) -> Any:
        """``int64`` while the codes fit, Python integers otherwise."""
        return np.int64 if self.base ** self.size < 2**62 else object

    def digits(self, configs: Array) -> Array:
        """Map ``(..., 3)`` integer vectors to ``(...)`` non-negative digits."""
        shifted = np.asarray(configs, dtype=np.int64) + self.bound
        if np.any(shifted < 0) or np.any(shifted > 2 * self.bound):
            raise ValueError("momentum component outside the encoder bound")
        span = 2 * self.bound + 1
        return (shifted[..., 0] * span + shifted[..., 1]) * span + shifted[..., 2]

    def keys(self, configs: Array) -> Array:
        """Encode ``(M, N, 3)`` configurations into ``(M,)`` multiset codes."""
        digits = np.sort(self.digits(configs), axis=-1).astype(self.dtype)
        powers = np.array(
            [self.base**i for i in range(self.size)], dtype=self.dtype
        )
        return (digits * powers).sum(axis=-1)

    def canonical_keys(self, configs: Array, chunk: int = 20_000) -> Array:
        """Encode configurations by the smallest code over their ``O_h`` images."""
        group = oh_group()
        configs = np.asarray(configs, dtype=np.int64)
        out = np.empty(len(configs), dtype=self.dtype)
        for start in range(0, len(configs), chunk):
            block = configs[start : start + chunk]
            images = np.einsum("gij,mnj->mgni", group, block)
            keys = self.keys(images.reshape(-1, self.size, 3))
            out[start : start + len(block)] = keys.reshape(len(block), 48).min(axis=1)
        return out


def occupation_factor(config: Array) -> int:
    """``N_c = prod_alpha n_alpha!``, the occupation factor of eq. (A.8)."""
    counts = Counter(map(tuple, np.asarray(config, dtype=np.int64).tolist()))
    factor = 1
    for count in counts.values():
        factor *= math.factorial(count)
    return factor


def sort_config(config: Array) -> Array:
    """Return the momenta of a configuration in the canonical vector order."""
    config = np.asarray(config, dtype=np.int64).reshape(-1, 3)
    norms = np.einsum("ij,ij->i", config, config)
    order = np.lexsort((config[:, 2], config[:, 1], config[:, 0], norms))
    return config[order]


# ---------------------------------------------------------------------------
# enumeration of B_N(0)
# ---------------------------------------------------------------------------


def enumerate_configurations(
    N: int,
    nsq_max: int,
    emax: float | None = None,
    tpiL: float | None = None,
) -> Array:
    """Enumerate every configuration in ``B_N(0)`` below the cutoff.

    Configurations are returned as ``(M, N, 3)`` integer arrays whose momenta are
    in non-decreasing order, so that each multiset appears exactly once.  The
    cutoff is ``n.n <= nsq_max`` per particle and, when ``emax`` is given,
    ``sum_i omega_i <= emax`` at the reference ``tpiL``.
    """
    if N < 2:
        raise ValueError("N must be at least 2")
    vectors, norms = candidate_vectors(nsq_max)
    if emax is None:
        energies = np.zeros(len(vectors))
        cutoff = float("inf")
    else:
        if tpiL is None:
            raise ValueError("an energy cutoff needs a reference tpiL")
        energies = omega(norms, tpiL)
        cutoff = float(emax)
        keep = energies <= cutoff - (N - 1)
        vectors, norms, energies = vectors[keep], norms[keep], energies[keep]

    index_of = {tuple(v): i for i, v in enumerate(vectors.tolist())}
    tpiL_eff = 0.0 if tpiL is None else tpiL

    found: list[list[int]] = []
    chosen = [0] * N

    def recurse(depth: int, start: int, total: Array, energy: float) -> None:
        remaining = N - depth
        if remaining == 1:
            index = index_of.get(tuple((-total).tolist()))
            if index is None or index < start:
                return
            if energy + energies[index] > cutoff:
                return
            chosen[depth] = index
            found.append(chosen.copy())
            return

        for index in range(start, len(vectors)):
            step = energies[index]
            if energy + remaining * step > cutoff:
                break
            new_total = total + vectors[index]
            # The remaining particles carry total momentum -new_total, so their
            # energy is at least that of a single lump of that momentum.
            if emax is not None:
                floor = math.sqrt(
                    (remaining - 1) ** 2 + tpiL_eff**2 * float(new_total @ new_total)
                )
                if energy + step + floor > cutoff:
                    continue
            chosen[depth] = index
            recurse(depth + 1, index, new_total, energy + step)

    recurse(0, 0, np.zeros(3, dtype=np.int64), 0.0)
    if not found:
        return np.zeros((0, N, 3), dtype=np.int64)
    return vectors[np.array(found, dtype=np.int64)]


# ---------------------------------------------------------------------------
# orbits
# ---------------------------------------------------------------------------


@dataclass
class OrbitTable:
    """The ``A_1^+`` basis of the ``N``-particle sector, grouped into orbits."""

    N: int
    reps: Array  # (M, N, 3) representative configurations
    sizes: Array  # (M,) orbit sizes d_mu
    occupations: Array  # (M,) occupation factors N_mu
    nsq: Array  # (M, N) sorted squared norms of the representative
    levels: Array  # (M,) index of the degenerate level each orbit belongs to
    level_nsq: Array  # (n_levels, N) sorted squared norms of each level
    vd_out: Array  # (K,) orbit index mu' of a disconnected contribution
    vd_in: Array  # (K,) orbit index mu
    vd_r: Array  # (K,) number of scatterers r
    vd_weight: Array  # (K,) volume-independent weight, see disconnected_table
    vd_nsq: Array  # (K, N) squared norms of the scatterers, padded with -1
    metadata: dict[str, Any]

    @property
    def n_orbits(self) -> int:
        return len(self.reps)

    def energies(self, mL: float) -> Array:
        """Non-interacting energies ``E_mu`` of each orbit at a given ``mL``."""
        return omega(self.nsq, 2 * np.pi / mL).sum(axis=-1)

    def save(self, path: str | Path) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(
            path,
            N=self.N,
            reps=self.reps.astype(np.int16),
            sizes=self.sizes.astype(np.int32),
            occupations=self.occupations.astype(np.int64),
            nsq=self.nsq.astype(np.int32),
            levels=self.levels.astype(np.int32),
            level_nsq=self.level_nsq.astype(np.int32),
            vd_out=self.vd_out.astype(np.int32),
            vd_in=self.vd_in.astype(np.int32),
            vd_r=self.vd_r.astype(np.int8),
            vd_weight=self.vd_weight.astype(np.float64),
            vd_nsq=self.vd_nsq.astype(np.int32),
            metadata=np.array(repr(self.metadata)),
        )
        return path

    @classmethod
    def load(cls, path: str | Path) -> "OrbitTable":
        with np.load(Path(path), allow_pickle=False) as data:
            return cls(
                N=int(data["N"]),
                reps=data["reps"].astype(np.int64),
                sizes=data["sizes"].astype(np.int64),
                occupations=data["occupations"].astype(np.int64),
                nsq=data["nsq"].astype(np.int64),
                levels=data["levels"].astype(np.int64),
                level_nsq=data["level_nsq"].astype(np.int64),
                vd_out=data["vd_out"].astype(np.int64),
                vd_in=data["vd_in"].astype(np.int64),
                vd_r=data["vd_r"].astype(np.int64),
                vd_weight=data["vd_weight"].astype(float),
                vd_nsq=data["vd_nsq"].astype(np.int64),
                metadata=eval(str(data["metadata"])),  # written by save() above
            )


def build_orbits(configs: Array, encoder: Encoder) -> tuple[Array, Array, Array]:
    """Partition configurations into ``O_h`` orbits.

    Returns the representatives, the orbit sizes ``d_mu`` and, for each input
    configuration, the index of its orbit.
    """
    keys = encoder.canonical_keys(configs)
    unique, first, inverse, counts = np.unique(
        keys, return_index=True, return_inverse=True, return_counts=True
    )
    del unique
    return configs[first], counts, inverse.reshape(-1)


def level_labels(nsq: Array) -> tuple[Array, Array]:
    """Group orbits into degenerate levels by their multiset of squared norms.

    Two orbits share a non-interacting energy at every ``L`` exactly when their
    momenta have the same squared norms.  Accidental degeneracies between
    different norm multisets occur only at isolated ``L`` and are not resolved.
    """
    level_nsq, labels = np.unique(nsq, axis=0, return_inverse=True)
    return labels.reshape(-1), level_nsq


# ---------------------------------------------------------------------------
# disconnected matrix elements
# ---------------------------------------------------------------------------


def scatterer_choices(rep: Array, r: int) -> Iterator[tuple[Array, Array, float]]:
    """Yield the distinct ``r``-particle scatterer sets of a configuration.

    For every ``s`` in ``P_r(c)`` of eq. (A.41) this yields the scatterers, the
    spectators ``sigma``, and the factor ``beta(c, s)`` of eq. (A.43).
    """
    counts = Counter(map(tuple, np.asarray(rep, dtype=np.int64).tolist()))
    distinct = list(counts)
    occupations = [counts[v] for v in distinct]

    for choice in itertools.product(*(range(n + 1) for n in occupations)):
        if sum(choice) != r:
            continue
        beta_squared = 1.0
        scatterers: list[tuple[int, int, int]] = []
        spectators: list[tuple[int, int, int]] = []
        for vector, taken, total in zip(distinct, choice, occupations):
            beta_squared *= math.comb(total, taken)
            scatterers.extend([vector] * taken)
            spectators.extend([vector] * (total - taken))
        yield (
            np.array(scatterers, dtype=np.int64).reshape(r, 3),
            np.array(spectators, dtype=np.int64).reshape(len(rep) - r, 3),
            math.sqrt(beta_squared),
        )


@lru_cache(maxsize=200_000)
def _outgoing_cached(
    norms: tuple[int, ...], total: tuple[int, int, int]
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Configurations with given total momentum and given squared norms.

    Energy conservation inside a degenerate level forces the outgoing scatterers
    ``s'`` to carry the same multiset of squared norms as the incoming ``s``, so
    this is the full set of final states contributing to ``V^d`` at leading
    order in the disconnected interactions.
    """
    target = np.array(total, dtype=np.int64)
    solutions: set[tuple[tuple[int, int, int], ...]] = set()

    for order in set(itertools.permutations(norms)):
        shells = [shell(n) for n in order]
        if any(len(s) == 0 for s in shells):
            continue

        partial = np.zeros((1, 0, 3), dtype=np.int64)
        sums = np.zeros((1, 3), dtype=np.int64)
        for vectors in shells[:-1]:
            if len(partial) * len(vectors) > MAX_PRODUCT:
                raise MemoryError(
                    "outgoing-state search exceeded the product limit; "
                    "lower the cutoff or raise MAX_PRODUCT"
                )
            repeats = len(vectors)
            partial = np.concatenate(
                [
                    np.repeat(partial, repeats, axis=0),
                    np.tile(vectors, (len(partial), 1)).reshape(-1, 1, 3),
                ],
                axis=1,
            )
            sums = np.repeat(sums, repeats, axis=0) + np.tile(
                vectors, (len(sums), 1)
            ).reshape(-1, 3)

        last = target[None, :] - sums
        keep = np.einsum("ij,ij->i", last, last) == order[-1]
        if not np.any(keep):
            continue
        complete = np.concatenate(
            [partial[keep], last[keep].reshape(-1, 1, 3)], axis=1
        )
        for config in complete:
            solutions.add(tuple(map(tuple, sort_config(config).tolist())))

    return tuple(sorted(solutions))


def outgoing_configurations(norms: Sequence[int], total: Array) -> list[Array]:
    """Wrapper around the cached search, returning ``(r, 3)`` arrays."""
    key_norms = tuple(sorted(int(n) for n in norms))
    key_total = tuple(int(component) for component in np.asarray(total).tolist())
    return [
        np.array(solution, dtype=np.int64)
        for solution in _outgoing_cached(key_norms, key_total)
    ]


def disconnected_table(
    reps: Array,
    sizes: Array,
    encoder: Encoder,
    key_to_orbit: dict[Any, int],
    r_values: Sequence[int] | None = None,
    progress: bool = True,
) -> tuple[Array, Array, Array, Array, Array]:
    """Tabulate the volume-independent part of ``<b_mu'|V^d_L(0)|b_mu>``.

    Equation (A.49) of the draft, restricted to the energy-conserving terms that
    survive in degenerate perturbation theory, reads

        <b_mu'|V^d|b_mu> = sum over contributions of
            sqrt(d_mu / d_mu') beta(c, s) beta(c', s')
            lambda_{r<-r} / (L^{3(r-1)} sqrt(N_s N_s') Omega_s Omega_s')

    with ``Omega_s' = Omega_s`` because ``s'`` and ``s`` share their squared
    norms.  Everything except ``lambda_{r<-r}``, the power of ``L`` and
    ``Omega_s`` is volume independent and is stored as the weight

        w = sqrt(d_mu / d_mu') beta(c, s) beta(c', s') / sqrt(N_s N_s') .

    Only particle-number-conserving sub-processes ``r -> r`` are tabulated:
    sub-processes that change the particle number connect different sectors and
    contribute to ``G`` only at the isolated volumes where levels of different
    particle number cross.
    """
    N = reps.shape[1]
    r_values = list(r_values) if r_values is not None else list(range(2, N))

    out_index: list[int] = []
    in_index: list[int] = []
    scatterer_count: list[int] = []
    weights: list[float] = []
    scatterer_nsq: list[list[int]] = []

    iterator: Any = range(len(reps))
    if progress:
        try:
            from tqdm import tqdm

            iterator = tqdm(iterator, desc="V^d", leave=False)
        except ImportError:  # pragma: no cover - tqdm is a declared dependency
            pass

    for mu in iterator:
        rep = reps[mu]
        occupation_in = occupation_factor(rep)
        for r in r_values:
            for scatterers, spectators, beta in scatterer_choices(rep, r):
                norms = np.einsum("ij,ij->i", scatterers, scatterers)
                total = scatterers.sum(axis=0)
                occ_s = occupation_factor(scatterers)
                occ_sigma = occupation_factor(spectators)
                # beta(c, s) is recomputed here only as a consistency check.
                assert math.isclose(
                    beta, math.sqrt(occupation_in / (occ_s * occ_sigma))
                )

                for outgoing in outgoing_configurations(norms, total):
                    final = sort_config(np.concatenate([outgoing, spectators]))
                    key = encoder.canonical_keys(final[None, ...])[0]
                    mu_prime = key_to_orbit[key]
                    occ_out = occupation_factor(outgoing)
                    occ_final = occupation_factor(final)
                    beta_out = math.sqrt(occ_final / (occ_out * occ_sigma))

                    weight = (
                        math.sqrt(sizes[mu] / sizes[mu_prime])
                        * beta
                        * beta_out
                        / math.sqrt(occ_s * occ_out)
                    )
                    out_index.append(mu_prime)
                    in_index.append(mu)
                    scatterer_count.append(r)
                    weights.append(weight)
                    padded = sorted(int(n) for n in norms.tolist())
                    scatterer_nsq.append(padded + [-1] * (N - r))

    if not weights:
        empty_int = np.zeros(0, dtype=np.int64)
        return (
            empty_int,
            empty_int,
            empty_int,
            np.zeros(0, dtype=float),
            np.zeros((0, N), dtype=np.int64),
        )

    return (
        np.array(out_index, dtype=np.int64),
        np.array(in_index, dtype=np.int64),
        np.array(scatterer_count, dtype=np.int64),
        np.array(weights, dtype=float),
        np.array(scatterer_nsq, dtype=np.int64),
    )


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def generate(
    N: int,
    nsq_max: int | None = None,
    emax: float | None = None,
    mL: float | None = None,
    disconnected: bool = True,
    verbose: bool = True,
) -> OrbitTable:
    """Build the orbit table of the ``N``-particle sector below the cutoff."""
    if emax is None and nsq_max is None:
        raise ValueError("give a momentum cutoff (nsq_max) or an energy cutoff (emax)")

    tpiL = None if mL is None else 2 * np.pi / mL
    if nsq_max is None:
        if tpiL is None:
            raise ValueError("an energy cutoff needs a reference mL")
        # The lightest companions sit at rest, so one particle can carry at most
        # emax - (N - 1) m of energy.
        largest = (float(emax) - (N - 1)) ** 2 - 1.0
        nsq_max = max(0, int(math.floor(largest / tpiL**2)))

    configs = enumerate_configurations(N, nsq_max, emax, tpiL)
    if verbose:
        print(f"[info] N={N}: {len(configs)} configurations in B_{N}(0)")
    if len(configs) == 0:
        raise ValueError("no configurations below the cutoff")

    bound = int(np.abs(configs).max())
    encoder = Encoder(bound=bound, size=N)
    reps, sizes, _ = build_orbits(configs, encoder)
    if verbose:
        print(f"[info] N={N}: {len(reps)} orbits, largest d_mu = {sizes.max()}")

    nsq = np.sort(np.einsum("mnj,mnj->mn", reps, reps), axis=-1)
    levels, level_nsq = level_labels(nsq)
    occupations = np.array([occupation_factor(rep) for rep in reps], dtype=np.int64)

    keys = encoder.canonical_keys(reps)
    key_to_orbit = {key: index for index, key in enumerate(keys.tolist())}

    if disconnected and N >= 3:
        vd = disconnected_table(reps, sizes, encoder, key_to_orbit, progress=verbose)
        if verbose:
            print(f"[info] N={N}: {len(vd[3])} disconnected contributions")
    else:
        vd = (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=float),
            np.zeros((0, N), dtype=np.int64),
        )

    metadata = {
        "N": N,
        "nsq_max": int(nsq_max),
        "emax": None if emax is None else float(emax),
        "mL": None if mL is None else float(mL),
        "n_configurations": int(len(configs)),
        "encoder_bound": bound,
    }
    return OrbitTable(
        N=N,
        reps=reps,
        sizes=sizes,
        occupations=occupations,
        nsq=nsq,
        levels=levels,
        level_nsq=level_nsq,
        vd_out=vd[0],
        vd_in=vd[1],
        vd_r=vd[2],
        vd_weight=vd[3],
        vd_nsq=vd[4],
        metadata=metadata,
    )


def orbit_filename(
    N: int,
    emax: float | None,
    mL: float | None,
    nsq_max: int | None = None,
    data_dir: str | Path = "data",
) -> Path:
    """Canonical path of a saved orbit table."""
    if emax is not None and mL is not None:
        stem = f"orbits_N{N}_EMAX{emax:g}_mL{mL:g}"
    else:
        stem = f"orbits_N{N}_nsq{nsq_max}"
    return Path(data_dir) / f"{stem}.npz"


def covering_table(
    N: int, emax: float, mL: float, data_dir: str | Path = "data"
) -> Path | None:
    """Find a saved table that covers a request, i.e. a superset of its states.

    A table generated with a larger cutoff, or at a larger reference ``mL``,
    contains every configuration the request needs; the analysis then applies
    its own cutoff.  The smallest such table is returned, or ``None``.
    """
    pattern = re.compile(rf"orbits_N{N}_EMAX([0-9.eE+-]+)_mL([0-9.eE+-]+)\.npz$")
    candidates: list[tuple[float, float, Path]] = []
    for candidate in Path(data_dir).glob(f"orbits_N{N}_EMAX*_mL*.npz"):
        match = pattern.search(candidate.name)
        if match is None:
            continue
        try:
            saved_emax, saved_mL = float(match.group(1)), float(match.group(2))
        except ValueError:
            continue
        if saved_emax >= emax and saved_mL >= mL:
            candidates.append((saved_emax, saved_mL, candidate))
    if not candidates:
        return None
    return min(candidates)[2]


def load_orbits(
    N: int,
    emax: float | None = None,
    mL: float | None = None,
    nsq_max: int | None = None,
    data_dir: str | Path = "data",
    path: str | Path | None = None,
) -> OrbitTable:
    """Load a saved orbit table, pointing at ``orbits.py`` when it is missing."""
    target = Path(path) if path is not None else orbit_filename(
        N, emax, mL, nsq_max, data_dir
    )
    if not target.exists() and path is None and emax is not None and mL is not None:
        covering = covering_table(N, emax, mL, data_dir)
        if covering is not None:
            return OrbitTable.load(covering)
    if not target.exists():
        if emax is not None and mL is not None:
            command = f"python orbits.py --N {N} --emax {emax:g} --mL {mL:g}"
        else:
            command = f"python orbits.py --N {N} --nsq-max {nsq_max}"
        raise FileNotFoundError(f"{target} not found; generate it with\n    {command}")
    return OrbitTable.load(target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="orbits.py",
        description="Generate O_h orbit representatives of the P=0, A_1^+ basis.",
    )
    parser.add_argument("--N", type=int, required=True, help="number of particles")
    parser.add_argument(
        "--emax",
        type=float,
        help="energy cutoff in units of m, imposed at the reference mL",
    )
    parser.add_argument(
        "--mL", type=float, help="reference mL for the energy cutoff (use the largest "
        "mL of the intended scan, so that smaller volumes are covered too)"
    )
    parser.add_argument(
        "--nsq-max", type=int, help="momentum cutoff n.n per particle, instead of --emax"
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--output", type=Path, help="explicit output path")
    parser.add_argument(
        "--no-disconnected",
        action="store_true",
        help="skip the V^d table (orbits and levels only)",
    )
    parser.add_argument("--quiet", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    table = generate(
        N=args.N,
        nsq_max=args.nsq_max,
        emax=args.emax,
        mL=args.mL,
        disconnected=not args.no_disconnected,
        verbose=not args.quiet,
    )
    output = args.output or orbit_filename(
        args.N, args.emax, args.mL, args.nsq_max, args.data_dir
    )
    table.save(output)
    print(f"[info] saved {table.n_orbits} orbits to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
