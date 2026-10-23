"""Figure 3 of the draft: two, four and six identical scalars.

Produces ``plots/spec_246.pdf``, the finite-volume spectrum for ``2m < E < 8m``
as a function of ``mL``, showing

* the non-interacting levels of each of the ``N = 2, 4, 6`` sectors,
* the same levels shifted and split by the disconnected sub-processes, and
* the solutions of ``det[G^{-1} - lambda] = 0`` in the three-sector space,

with the avoided crossings the off-diagonal couplings induce between sectors.
It is drawn through ``figures.spectrum_figure``, so it has by construction the
size of ``plots/spec_24.pdf`` (``plot_settings.SPECTRUM_FIG_SIZE``) and the two
can sit side by side in the draft.

Section 3.2 adds nothing to the machinery of section 3.1 but a third row and
column of ``lambda``: ``orbits.py`` is already written for arbitrary ``N``,
``gfunctions.GMatrix`` for an arbitrary set of sectors, and ``root_finding.py``
counts and solves for any number of them.  What this module adds is the orbit
table for ``N = 6``, which is the expensive input, a choice of couplings that
makes a crossing visible, and the measurement of the crossings themselves.

Usage
-----
    python figures_246.py

generating ``data/orbits_N6_*.npz`` on the first run, which takes about a
minute; ``--report`` adds the timings and the crossings it found.
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from numpy.typing import ArrayLike, NDArray

import orbits as orbit_module
from figures import coupling_title, spectrum_figure
from gfunctions import Couplings, Cutoff
from orbits import load_orbits
from quantisation_condition import QC, scan_volumes

Array = NDArray[Any]

SECTORS: tuple[int, ...] = (2, 4, 6)

# The window of the draft, and a regulator whose plateau covers it.  The
# couplings depend on the regulator, so this is a definition, not an
# approximation; what matters is that the levels inside the window are not
# themselves tapered, which needs ``emax - width`` above the top of the window.
EMIN, EMAX = 1.8, 8.0
CUTOFF = Cutoff(emax=11.0, shape="smooth", width=2.0)


# ---------------------------------------------------------------------------
# the orbit tables
# ---------------------------------------------------------------------------


def ensure_orbits(
    mL_ref: float,
    sectors: Sequence[int] = SECTORS,
    cutoff: Cutoff = CUTOFF,
    data_dir: str | Path = "data",
    verbose: bool = True,
) -> dict[int, Any]:
    """Load the orbit table of each sector, generating the ones that are absent.

    The tables depend only on ``N``, the cutoff and the reference volume, never
    on the couplings, so they are built once and reused by every scan.  ``N =
    6`` is far and away the expensive one: the number of configurations with
    vanishing total momentum grows steeply with both ``N`` and the cutoff, and
    the disconnected table has to solve for the outgoing configurations of every
    ``r``-particle sub-process with ``2 <= r <= N - 1``.
    """
    tables: dict[int, Any] = {}
    for N in sectors:
        start = time.perf_counter()
        try:
            tables[N] = load_orbits(
                N, emax=cutoff.support, mL=mL_ref, data_dir=data_dir
            )
            if verbose:
                print(
                    f"[info] N={N}: loaded {tables[N].n_orbits} orbits "
                    f"in {time.perf_counter() - start:.2f}s"
                )
        except FileNotFoundError:
            if verbose:
                print(f"[info] N={N}: no table below the cutoff, generating one")
            table = orbit_module.generate(
                N=N, emax=cutoff.support, mL=mL_ref, verbose=verbose
            )
            path = orbit_module.orbit_filename(
                N, cutoff.support, mL_ref, data_dir=data_dir
            )
            table.save(path)
            tables[N] = table
            if verbose:
                print(
                    f"[info] N={N}: generated {table.n_orbits} orbits, "
                    f"{len(table.vd_weight)} disconnected contributions, "
                    f"in {time.perf_counter() - start:.1f}s -> {path}"
                )
    return tables


# ---------------------------------------------------------------------------
# measuring the avoided crossings
# ---------------------------------------------------------------------------


def content_at(qc: QC, energy: float) -> Array | None:
    """``qc.content`` at a root, or ``None`` where it cannot be taken.

    A root can land within rounding of a pole of ``G``, which happens when its
    shift off that pole is below ``eps`` relative: ``G`` is then infinite and
    the SVD in ``null_vector`` has nothing to work with.  Such a level is
    decoupled from the connected vertex -- the opposite of a crossing -- so
    nothing is lost by passing over it.
    """
    with np.errstate(all="ignore"):
        try:
            content = qc.content(float(energy))
        except np.linalg.LinAlgError:
            return None
    return content if np.all(np.isfinite(content)) else None


def crossings(
    mLs: ArrayLike,
    couplings: Couplings,
    sectors: Sequence[int] = SECTORS,
    cutoff: Cutoff = CUTOFF,
    emin: float = EMIN,
    emax: float = EMAX,
    tables: Mapping[int, Any] | None = None,
    mixing: float = 0.35,
    data_dir: str | Path = "data",
) -> list[dict[str, Any]]:
    """The avoided crossings in a scan, found by the mixing of the states.

    Following tracks through the spectrum and looking for a local minimum of
    the spacing is delicate once the levels are dense.  The particle-number
    content says it directly: away from a crossing a finite-volume state is
    almost entirely one sector, and at the closest approach of two levels it is
    an even mixture of the two.  A root whose second-largest content exceeds
    ``mixing`` is therefore near a crossing, and the gap there is its distance
    to the partner it is repelling, whichever neighbour is nearer.

    The width of the crossing is the gap at its *closest* approach, which is the
    volume where the mixture is most even.  Away from that volume the levels
    have begun to separate again and the same measurement returns something
    larger, so the entries are ranked by how even the mixture is rather than by
    the gap, and the first entry of a crossing is the one to quote.

    Returns one entry per mixed level, with the volume, the energy, the pair of
    sectors, the gap and the imbalance, most evenly mixed first.
    """
    mLs = np.asarray(mLs, dtype=float)
    sectors = tuple(sectors)
    if tables is None:
        tables = ensure_orbits(
            float(mLs.max()), sectors, cutoff, data_dir, verbose=False
        )

    found: list[dict[str, Any]] = []
    for mL in mLs:
        qc = QC(
            couplings,
            float(mL),
            sectors=sectors,
            cutoff=cutoff,
            mL_ref=float(mLs.max()),
            tables=tables,
        )
        roots = qc.roots(emin, emax)
        for position, energy in enumerate(roots):
            content = content_at(qc, energy)
            if content is None:
                continue
            order = np.argsort(content)
            if content[order[-2]] < mixing:
                continue
            left = roots[position - 1] if position > 0 else -np.inf
            right = roots[position + 1] if position + 1 < len(roots) else np.inf
            found.append(
                {
                    "mL": float(mL),
                    "E": float(energy),
                    "gap": float(min(energy - left, right - energy)),
                    "imbalance": float(content[order[-1]] - content[order[-2]]),
                    "sectors": tuple(
                        sorted((sectors[order[-1]], sectors[order[-2]]))
                    ),
                    "content": content.copy(),
                }
            )
    return sorted(found, key=lambda entry: entry["imbalance"])


def report_crossings(found: Sequence[Mapping[str, Any]], limit: int = 10) -> None:
    """Print the most even mixtures; the gap is in the ``E/m`` of the figure.

    One crossing contributes several volumes, so the same crossing appears more
    than once, with the first of them the closest approach.
    """
    if not found:
        print("[info] no avoided crossing above the mixing threshold")
        return
    print(f"[info] {len(found)} mixed levels; closest approaches:")
    print(
        f"   {'mL':>7}{'E/m':>9}{'sectors':>10}{'gap':>9}"
        f"{'imbal':>8}{'content':>26}"
    )
    for entry in found[:limit]:
        low, high = entry["sectors"]
        content = ", ".join(f"{value:.2f}" for value in entry["content"])
        print(
            f"   {entry['mL']:>7.3f}{entry['E']:>9.4f}"
            f"{f'{low}-{high}':>10}{entry['gap']:>9.4f}"
            f"{entry['imbalance']:>8.3f}   ({content})"
        )


# ---------------------------------------------------------------------------
# the figure
# ---------------------------------------------------------------------------


def main(
    mLs: ArrayLike | None = None,
    couplings: Couplings | None = None,
    cutoff: Cutoff = CUTOFF,
    emin: float = EMIN,
    emax: float = EMAX,
    sectors: Sequence[int] = SECTORS,
    filename: str | Path = "plots/spec_246.pdf",
    data_dir: str | Path = "data",
    report: bool = False,
    open_pdf: bool = False,
) -> Path:
    """Draw figure 3 of the draft.

    The default couplings are the two-and-four-particle ones of ``figures.main``
    extended to the six-particle sector.  They are chosen the same way, and run
    into the same bound: the shifts a coupling can produce saturate once
    ``lambda u^2`` reaches the spacing of the poles of ``G``, because between
    two poles there are at most ``rank(lambda)`` roots and a level cannot be
    pushed past the next one.  The vertex factor falls steeply with the particle
    number, ``u_N ~ L^{-3(N - 1) / 2}``, so the six-particle couplings have to be
    larger by many orders of magnitude to do the same work, and ``lambda_{6<-4}``
    is what opens the four-six crossings.

    The volumes stop short of the ``mL = 7`` of the two-and-four-particle figure
    because the six-particle levels proliferate: the window holds 14 levels at
    ``mL = 3.8``, 72 at ``mL = 6`` and 207 at ``mL = 7``, and by the last of
    those the upper half of the figure is solid ink.  The spacing there also
    falls below the step a level takes between neighbouring volumes, so
    ``figures.level_tracks`` can no longer say which level continues into which
    and the curves acquire a visible zigzag.  Both are a statement about how
    much of this spectrum can be drawn at once, not about the solutions, which
    the root counting returns in full at every volume; ``--mL-max`` goes further
    out for anyone who wants to look.
    """
    mLs = np.linspace(3.8, 6.0, 177) if mLs is None else np.asarray(mLs, dtype=float)
    couplings = couplings or DEFAULT_COUPLINGS
    sectors = tuple(sectors)

    start = time.perf_counter()
    tables = ensure_orbits(float(mLs.max()), sectors, cutoff, data_dir)
    t_orbits = time.perf_counter() - start

    start = time.perf_counter()
    result = scan_volumes(
        mLs,
        couplings,
        emin=emin,
        emax=emax,
        cutoff=cutoff,
        sectors=sectors,
        data_dir=data_dir,
    )
    t_scan = time.perf_counter() - start

    path = spectrum_figure(
        result,
        emin=emin,
        emax=emax,
        filename=filename,
        title=coupling_title(couplings),
        # Levels only ever come into this window over the top -- 13 of them at
        # ``mL = 3.8`` and 77 at ``mL = 6``, and the ground state stays well
        # above ``emin`` -- so the rank from the bottom is stable and is the
        # track.  Following them by continuity instead would be free to mismatch
        # where the six-particle levels are closer together than the step
        # between volumes, and visibly does.
        tracks=False,
        open_pdf=open_pdf,
    )
    print(f"[info] {path}")

    if report:
        counts = [len(np.atleast_1d(row)) for row in result["roots"]]
        print(
            f"[time] orbit tables {t_orbits:.1f}s; scan {t_scan:.1f}s for "
            f"{len(mLs)} volumes ({1e3 * t_scan / len(mLs):.0f} ms/volume), "
            f"{sum(counts)} roots in all, {min(counts)}-{max(counts)} per volume"
        )
        start = time.perf_counter()
        report_crossings(
            crossings(
                mLs, couplings, sectors, cutoff, emin, emax, tables, data_dir=data_dir
            )
        )
        print(f"[time] crossing search {time.perf_counter() - start:.1f}s")
    return path


# Chosen by scanning ``lambda_{4<-2}`` and ``lambda_{6<-4}`` over decades and
# measuring the crossings with ``crossings`` below.  At these values the window
# holds a two-four avoided crossing of width 0.21 in ``E/m`` at ``mL = 4.2``,
# which is 3% of the axis and plainly visible, together with four-six and
# two-six crossings of about 0.04 -- narrow, but enough that the six-particle
# sector is doing something rather than riding along.  ``lambda_{4<-2}`` past
# about 1e4 and ``lambda_{6<-4}`` past about 1e9 close the crossings again, for
# the reason given in ``main``.
DEFAULT_COUPLINGS = Couplings.two_four_six(
    l22=20.0, l24=3.0e3, l44=1.0, l26=0.0, l46=3.0e8, l66=1.0
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="figures_246.py", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--mL-min", type=float, default=3.8)
    parser.add_argument("--mL-max", type=float, default=6.0)
    parser.add_argument("--n-volumes", type=int, default=177)
    parser.add_argument("--emin", type=float, default=EMIN)
    parser.add_argument("--emax", type=float, default=EMAX)
    parser.add_argument("--output", type=Path, default=Path("plots/spec_246.pdf"))
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--report",
        action="store_true",
        help="print the runtimes and the avoided crossings that were found",
    )
    parser.add_argument("--open", action="store_true", help="open the PDF when done")
    return parser


if __name__ == "__main__":
    args = build_parser().parse_args()
    main(
        mLs=np.linspace(args.mL_min, args.mL_max, args.n_volumes),
        emin=args.emin,
        emax=args.emax,
        filename=args.output,
        data_dir=args.data_dir,
        report=args.report,
        open_pdf=args.open,
    )
