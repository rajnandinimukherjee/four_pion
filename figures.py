"""Figures for section 3.1 of the draft: two and four identical scalars.

Produces

* ``plots/spec_24.pdf``  -- the finite-volume spectrum for ``2m < E < 6m``,
  showing the free levels, the levels shifted and split by the disconnected
  sub-processes, and the solutions of the quantization condition, with the
  avoided crossings induced by ``lambda_{4<-2}`` (figure 1 of the draft), and
* ``plots/ME_24.pdf``    -- the four-particle content of each finite-volume
  state (figure 2 of the draft).

Both read a scan produced by ``quantisation_condition.scan_volumes``, computing
it if the cache is absent.  Both are drawn at the size of the reference figure
``plots/EFV_N6_g20_5000_1_NL16.pdf``; see ``plot_settings.SPECTRUM_FIG_SIZE``.
"""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.cm import ScalarMappable
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from numpy.typing import ArrayLike, NDArray
from scipy.interpolate import PchipInterpolator

from gfunctions import Couplings
from plot_settings import SPECTRUM_FIG_SIZE, plotparams
from quantisation_condition import QC, SECTORS, pad_levels, scan_volumes
from utils import call_PDF

Array = NDArray[Any]

plt.rcParams.update(plotparams)

COLOURS: dict[int, str] = {2: "tab:red", 4: "tab:blue", 6: "tab:green"}


def coupling_title(couplings: Couplings, per_line: int = 3) -> str:
    """The non-vanishing couplings, in the ``lambda_{N+N'}`` convention.

    Eq. (A.4) of the draft labels the vertices descending from the Lagrangian
    monomial ``lambda_k phi^k / k!`` by ``lambda_{N'<-N} = lambda_{N+N'}``, so
    that ``lambda_{2<-2}`` is ``lambda_4``, ``lambda_{4<-2}`` is ``lambda_6``
    and ``lambda_{4<-4}`` is ``lambda_8``.

    The relation holds for the bare couplings only: once the ``N <= 1``
    vertices have been absorbed, the remaining couplings are independent
    parameters, so two of them can share a value of ``N + N'`` --
    ``lambda_{3<-3}`` and ``lambda_{4<-2}`` both descend from ``lambda_6`` --
    without being equal.  Where that happens the explicit ``N'<-N`` labels are
    used instead, since ``lambda_{N+N'}`` would not say which is meant.
    """
    entries = [
        (low, high, value)
        for (low, high), value in sorted(couplings.values.items())
        if value != 0.0
    ]
    orders = Counter(low + high for low, high, _ in entries)
    parts = []
    for low, high, value in entries:
        order = low + high
        label = (
            rf"\lambda_{{{order}}}"
            if orders[order] == 1
            else rf"\lambda_{{{high}\leftarrow{low}}}"
        )
        parts.append(rf"{label}={value:g}")
    # Past a few couplings the list is wider than the narrow spectrum canvas,
    # so it is broken into lines of at most ``per_line`` of them.
    lines = [
        ",\\; ".join(parts[start : start + per_line])
        for start in range(0, len(parts), per_line)
    ]
    return "\n".join(f"${line}$" for line in lines)


# ---------------------------------------------------------------------------
# following a level across the scan
# ---------------------------------------------------------------------------


def level_tracks(per_volume: Any, gap: float = 0.1) -> Array:
    """Stack per-volume level lists into an ``(n_volumes, n_tracks)`` array.

    ``pad_levels`` labels a level by its rank in ``E``, which is not stable: a
    level entering or leaving the energy window shifts the rank of every level
    above it, and a curve drawn through one column then jumps vertically at
    that volume.  Here each level is instead followed by continuity, matching
    the levels of consecutive volumes with an order-preserving alignment.  The
    order may not be crossed, which is what the spectrum does -- levels of the
    quantization condition repel rather than cross -- so the only freedom is
    which levels are left unmatched, at the cost ``gap`` each.

    ``gap`` therefore has to be larger than the step a level takes from one
    volume of the scan to the next, and smaller than the spacing between
    neighbouring levels; it is in units of ``E/m``.
    """
    rows = [
        np.sort(np.atleast_1d(np.asarray(row, dtype=float))) for row in per_volume
    ]
    tracks: list[Array] = []
    columns: list[int] = []

    for index, row in enumerate(rows):
        if index == 0:
            assigned: list[int | None] = []
            for _ in row:
                tracks.append(np.full(len(rows), np.nan))
                assigned.append(len(tracks) - 1)
        else:
            assigned = [None] * len(row)
            for position, target in enumerate(_align(rows[index - 1], row, gap)):
                if target is not None:
                    assigned[target] = columns[position]
            for position, column in enumerate(assigned):
                if column is None:
                    tracks.append(np.full(len(rows), np.nan))
                    assigned[position] = len(tracks) - 1

        for position, column in enumerate(assigned):
            tracks[int(column)][index] = row[position]
        columns = [int(column) for column in assigned]

    if not tracks:
        return np.empty((len(rows), 0), dtype=float)
    return np.column_stack(tracks)


def _align(previous: Array, current: Array, gap: float) -> list[int | None]:
    """Order-preserving match of ``previous`` levels onto ``current`` ones.

    Returns, for each level of ``previous``, the index of the level of
    ``current`` it continues into, or ``None`` where it has no partner.  This
    is the usual global sequence alignment: matching two levels costs the
    distance between them, and leaving one unmatched costs ``gap``.
    """
    n, m = len(previous), len(current)
    cost = np.zeros((n + 1, m + 1), dtype=float)
    cost[0, :] = gap * np.arange(m + 1)
    cost[:, 0] = gap * np.arange(n + 1)
    move = np.zeros((n + 1, m + 1), dtype=np.int8)
    move[0, 1:], move[1:, 0] = 2, 1

    for i in range(1, n + 1):
        for j in range(1, m + 1):
            options = (
                cost[i - 1, j - 1] + abs(previous[i - 1] - current[j - 1]),
                cost[i - 1, j] + gap,
                cost[i, j - 1] + gap,
            )
            best = int(np.argmin(options))
            move[i, j], cost[i, j] = best, options[best]

    match: list[int | None] = [None] * n
    i, j = n, m
    while i > 0 and j > 0:
        if move[i, j] == 0:
            match[i - 1] = j - 1
            i, j = i - 1, j - 1
        elif move[i, j] == 1:
            i -= 1
        else:
            j -= 1
    return match


# ---------------------------------------------------------------------------
# the figures
# ---------------------------------------------------------------------------


def spectrum_figure(
    result: Mapping[str, Any],
    emin: float = 1.8,
    emax: float = 6.0,
    filename: str | Path = "plots/spec_24.pdf",
    figsize: tuple[float, float] = SPECTRUM_FIG_SIZE,
    title: str | None = None,
    tracks: bool = True,
    open_pdf: bool = False,
) -> Path:
    """Figure 1: the spectrum, against the free and ``V^d``-shifted levels.

    ``title`` overrides the list of couplings, which past three or four of them
    is wider than the canvas; ``coupling_title`` can wrap it instead.

    ``tracks`` follows each level by continuity (``level_tracks``).  That is the
    safe choice, but it is only needed where a level enters or leaves the window
    somewhere other than the top: entering at the top adds a level above all the
    others and leaves the rank of the rest alone, so there the rank *is* the
    track and ``pad_levels`` gives the same curves for none of the work.  It
    also gives better ones once the levels are dense, as in the ``N in
    {2, 4, 6}`` window, where the spacing drops below the step a level takes
    between volumes, the alignment can no longer tell which level continues into
    which, and the curves it returns zigzag between neighbours.  Pass
    ``tracks=False`` when the level count only ever grows from the top.
    """
    mLs = np.asarray(result["mLs"], dtype=float)
    sectors = [int(N) for N in np.asarray(result["sectors"]).tolist()]
    couplings = result["couplings"]

    _, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
    for N in sectors:
        colour = COLOURS.get(N, "grey")
        _draw(ax, mLs, pad_levels(result[f"free{N}"]), colour, alpha=0.25, lw=0.7,
              label=rf"$N={N}$, free")
        if _is_shifted(result[f"free{N}"], result[f"shifted{N}"]):
            _draw(ax, mLs, pad_levels(result[f"shifted{N}"]), colour, alpha=0.75,
                  lw=0.7, linestyle=(0, (4, 2)),
                  label=rf"$N={N}$, $+\hat{{V}}^{{\sf d}}_L$")
    roots = level_tracks(result["roots"]) if tracks else pad_levels(result["roots"])
    _draw(ax, mLs, roots, "k", alpha=1.0, lw=1.0,
          label=r"$\det[\mathcal{G}^{-1}-\lambda]=0$")

    ax.set_title(coupling_title(couplings) if title is None else title, size=8)
    ax.set_xlabel(r"$mL$", size=12)
    ax.set_ylabel(r"$E/m$", size=12)
    ax.set_xlim([mLs[0], mLs[-1]])
    ax.set_ylim([emin, emax])
    ax.legend(loc="lower right", fontsize=6.5, framealpha=0.9)

    return _save(filename, open_pdf)


def content_figure(
    result: Mapping[str, Any],
    couplings: Couplings | None = None,
    cutoff: Any | None = None,
    emin: float = 1.8,
    emax: float = 6.0,
    filename: str | Path = "plots/ME_24.pdf",
    figsize: tuple[float, float] = SPECTRUM_FIG_SIZE,
    sectors: Sequence[int] = SECTORS,
    data_dir: str | Path = "data",
    smooth: int = 12,
    cmap: str = "coolwarm",
    open_pdf: bool = False,
) -> Path:
    """Figure 2: the four-particle content of each finite-volume state.

    The content is recomputed from the saved roots, since it needs ``G`` itself
    and not only the energies.  Each level is then drawn as one continuous
    curve, interpolated between the volumes of the scan and coloured along its
    length by the content, so that an avoided crossing reads as a gradient from
    one sector into the other rather than as a change of marker colour.

    The content swaps over a range in ``mL`` set by the gap of the avoided
    crossing, which is narrow, so the scan itself has to resolve it: give
    ``scan_volumes`` enough volumes that the swap spans several of them.
    ``smooth`` only interpolates for drawing and adds no information.

    The gaps are a fraction of a percent of the energy range, because the
    two-to-four vertex carries ``u_4 ~ L^(-9/2)``, so at full scale an avoided
    crossing reads as a kink in the track and a short gradient in its colour
    rather than as a visible separation.
    """
    mLs = np.asarray(result["mLs"], dtype=float)
    couplings = couplings or result["couplings"]
    cutoff = result.get("cutoff_object", result["cutoff"]) if cutoff is None else cutoff

    energies = level_tracks(result["roots"])
    content = np.full_like(energies, np.nan)

    tables = None
    for index, mL in enumerate(mLs):
        qc = QC(
            couplings,
            float(mL),
            sectors=sectors,
            cutoff=cutoff,
            mL_ref=float(mLs.max()),
            data_dir=data_dir,
            tables=tables,
        )
        tables = qc.tables
        for column, energy in enumerate(energies[index]):
            if np.isnan(energy):
                continue
            content[index, column] = qc.content(float(energy))[-1]

    _, ax = plt.subplots(1, 1, figsize=figsize, layout="constrained")
    norm = Normalize(vmin=0.0, vmax=1.0)
    _draw_tracks(ax, mLs, energies, content, norm=norm, cmap=cmap, smooth=smooth)

    bar = plt.colorbar(
        ScalarMappable(norm=norm, cmap=cmap), ax=ax, pad=0.02, aspect=45
    )
    bar.set_label("four-particle content", size=9)

    ax.set_title(coupling_title(couplings), size=8)
    ax.set_xlabel(r"$mL$", size=12)
    ax.set_ylabel(r"$E/m$", size=12)
    ax.set_xlim([mLs[0], mLs[-1]])
    ax.set_ylim([emin, emax])

    return _save(filename, open_pdf)


def _is_shifted(free: Any, shifted: Any, tol: float = 1e-10) -> bool:
    """Whether ``V^d`` moves the levels of a sector at all.

    A disconnected sub-process acts on a subset of the particles and needs at
    least one spectator, so in the two-particle sector there is nothing for
    ``V^d`` to do: its levels are the free ones and the dashed curve would lie
    exactly on the solid one.  The two lists are compared as sets of distinct
    energies, since ``shifted_levels`` returns one entry per orbit of a
    degenerate level where ``unshifted_levels`` returns one entry per level,
    so a sector with no shift still gives the longer list.
    """
    for one, other in zip(free, shifted):
        a = np.unique(np.round(np.atleast_1d(np.asarray(one, float)), 10))
        b = np.unique(np.round(np.atleast_1d(np.asarray(other, float)), 10))
        if a.shape != b.shape or not np.allclose(a, b, rtol=0.0, atol=tol):
            return True
    return False


def _draw_tracks(
    ax: Any,
    mLs: Array,
    energies: Array,
    content: Array,
    norm: Normalize,
    cmap: str,
    smooth: int,
    lw: float = 1.3,
) -> None:
    """Draw every level as one continuous line, coloured by its content.

    ``smooth`` is the number of interpolated points inserted between adjacent
    volumes of the scan.  The interpolant is ``PchipInterpolator`` rather than a
    natural cubic spline: the tracks bend sharply where two levels repel, and a
    shape-preserving interpolant neither rings around those bends nor carries
    the content outside ``[0, 1]``.
    """
    for column in range(energies.shape[1]):
        finite = np.isfinite(energies[:, column]) & np.isfinite(content[:, column])
        for rows in _runs(finite):
            x = mLs[rows]
            if x.size < 2:
                continue
            fine = np.linspace(x[0], x[-1], (x.size - 1) * smooth + 1)
            levels = PchipInterpolator(x, energies[rows, column])(fine)
            weights = PchipInterpolator(x, content[rows, column])(fine)

            points = np.column_stack([fine, levels]).reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)
            line = LineCollection(
                segments, cmap=cmap, norm=norm, linewidth=lw, capstyle="round"
            )
            line.set_array(np.clip(0.5 * (weights[:-1] + weights[1:]), 0.0, 1.0))
            ax.add_collection(line)


def _runs(mask: Array) -> Iterator[Array]:
    """The indices of each maximal run of ``True`` in ``mask``.

    A level can leave the energy window and return as the volume grows, which
    would otherwise be bridged by a spurious interpolated segment.
    """
    indices = np.flatnonzero(mask)
    if indices.size == 0:
        return
    breaks = np.flatnonzero(np.diff(indices) > 1) + 1
    yield from np.split(indices, breaks)


def _draw(
    ax: Any,
    mLs: Array,
    levels: Array,
    colour: str,
    alpha: float,
    lw: float,
    label: str | None = None,
    linestyle: Any = "-",
) -> None:
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


def _save(filename: str | Path, open_pdf: bool) -> Path:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    # The ``tight`` bounding box of ``plotparams`` would crop the page back to
    # the drawing, so the saved figure would no longer have the reference size.
    # Constrained layout already keeps the decorations inside the canvas.
    with plt.rc_context({"savefig.bbox": None}):
        call_PDF(filename, open=open_pdf)
    return filename


def main(
    mLs: ArrayLike | None = None,
    couplings: Couplings | None = None,
    cutoff: Any = 10.0,
    emin: float = 1.5,
    emax: float = 6.0,
    open_pdf: bool = False,
) -> None:
    """Draw both figures of section 3.1.

    The default couplings are chosen so that the figures show something.  Only
    ``lambda_6`` has any real range, because the shifts are bounded: between
    consecutive poles of ``G`` there are at most ``rank(lambda)`` roots, and as
    a coupling grows the roots move onto the zeros of the corresponding entry
    of ``G``, so no level can be pushed past the next pole.  A coupling
    therefore stops doing anything once ``lambda u^2`` reaches the pole
    spacing, which is also where the linear threshold expansions of section 3.1
    stop describing the levels.

    In practice ``lambda_4`` and ``lambda_8`` saturate: ``lambda_8`` moves the
    four-particle threshold by at most ``+6%`` at ``mL = 4`` however large it
    is (``1e8`` and ``1e10`` give the same spectrum to four decimals), and
    beyond about ``1e6`` it also drives the content to pure 0 or 1 and closes
    the avoided crossings.  ``lambda_4`` is kept small so that the two-particle
    ground state stays near ``2m``; ``lambda_6`` then sets both the level
    shifts and the crossings.

    At these values the levels sit 10-14% from their non-interacting positions
    at ``mL = 4`` and 3-7% at ``mL = 7``, the two-particle ground state stays
    within 8% of ``2m``, and the two lowest avoided crossings are the widest
    the model allows.  The ground state dips just below ``2m``, so the scan
    starts at ``emin`` below threshold and the figures show from just below it
    too, rather than from the ``2m`` of the draft.
    """
    mLs = np.linspace(4.0, 7.0, 121) if mLs is None else np.asarray(mLs, dtype=float)
    couplings = couplings or Couplings.two_four(
        l22=20.0, l24=15000.0, l44=1.0, l33=0.0
    )
    result = scan_volumes(mLs, couplings, emin=emin, emax=emax, cutoff=cutoff)
    print(f"[info] {spectrum_figure(result, open_pdf=open_pdf)}")
    print(f"[info] {content_figure(result, open_pdf=open_pdf)}")


if __name__ == "__main__":
    main()
