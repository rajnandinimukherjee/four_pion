"""Figures for section 3.1 of the draft: two and four identical scalars.

Produces

* ``plots/spec_24.pdf``  -- the finite-volume spectrum for ``2m < E < 6m``,
  showing the free levels, the levels shifted and split by the disconnected
  sub-processes, and the solutions of the quantization condition, with the
  avoided crossings induced by ``lambda_{4<-2}`` (figure 1 of the draft), and
* ``plots/ME_24.pdf``    -- the four-particle content of each finite-volume
  state (figure 2 of the draft).

Both read a scan produced by ``quantisation_condition.scan_volumes``, computing
it if the cache is absent.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import ArrayLike, NDArray

from gfunctions import Couplings
from plot_settings import plotparams
from quantisation_condition import QC, SECTORS, pad_levels, scan_volumes
from utils import call_PDF

Array = NDArray[Any]

plt.rcParams.update(plotparams)

COLOURS: dict[int, str] = {2: "tab:red", 4: "tab:blue", 6: "tab:green"}


def coupling_title(couplings: Couplings) -> str:
    """A title listing the non-vanishing couplings."""
    return ",\\; ".join(
        rf"\lambda_{{{low}\leftarrow{high}}}={value:g}"
        for (low, high), value in sorted(couplings.values.items())
        if value != 0.0
    )


def spectrum_figure(
    result: Mapping[str, Any],
    emin: float = 1.95,
    emax: float = 6.0,
    filename: str | Path = "plots/spec_24.pdf",
    figsize: tuple[float, float] = (3.4, 6.2),
    open_pdf: bool = False,
) -> Path:
    """Figure 1: the spectrum, against the free and ``V^d``-shifted levels."""
    mLs = np.asarray(result["mLs"], dtype=float)
    sectors = [int(N) for N in np.asarray(result["sectors"]).tolist()]
    couplings = result["couplings"]

    _, ax = plt.subplots(1, 1, figsize=figsize)
    for N in sectors:
        colour = COLOURS.get(N, "grey")
        _draw(ax, mLs, result[f"free{N}"], colour, alpha=0.25, lw=0.7,
              label=rf"$N={N}$, free")
        _draw(ax, mLs, result[f"shifted{N}"], colour, alpha=0.75, lw=0.7,
              linestyle=(0, (4, 2)), label=rf"$N={N}$, $+\hat{{V}}^{{\sf d}}_L$")
    _draw(ax, mLs, result["roots"], "k", alpha=1.0, lw=1.0,
          label=r"$\det[\mathcal{G}^{-1}-\lambda]=0$")

    ax.set_title(f"${coupling_title(couplings)}$", size=8)
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
    emin: float = 1.95,
    emax: float = 6.0,
    filename: str | Path = "plots/ME_24.pdf",
    figsize: tuple[float, float] = (3.6, 6.2),
    sectors: Sequence[int] = SECTORS,
    data_dir: str | Path = "data",
    open_pdf: bool = False,
) -> Path:
    """Figure 2: the four-particle content of each finite-volume state.

    The content is recomputed from the saved roots, since it needs ``G`` itself
    and not only the energies.
    """
    mLs = np.asarray(result["mLs"], dtype=float)
    couplings = couplings or result["couplings"]
    cutoff = result.get("cutoff_object", result["cutoff"]) if cutoff is None else cutoff

    energies = pad_levels(result["roots"])
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

    _, ax = plt.subplots(1, 1, figsize=figsize)
    for column in range(energies.shape[1]):
        ax.plot(mLs, energies[:, column], "-", color="0.75", lw=0.6, zorder=1)
    scatter = ax.scatter(
        np.repeat(mLs[:, None], energies.shape[1], axis=1),
        energies,
        c=content,
        s=4,
        cmap="coolwarm",
        vmin=0.0,
        vmax=1.0,
        zorder=2,
    )
    bar = plt.colorbar(scatter, ax=ax, pad=0.02)
    bar.set_label("four-particle content", size=9)

    ax.set_title(f"${coupling_title(couplings)}$", size=8)
    ax.set_xlabel(r"$mL$", size=12)
    ax.set_ylabel(r"$E/m$", size=12)
    ax.set_xlim([mLs[0], mLs[-1]])
    ax.set_ylim([emin, emax])

    return _save(filename, open_pdf)


def _draw(
    ax: Any,
    mLs: Array,
    per_volume: Any,
    colour: str,
    alpha: float,
    lw: float,
    label: str | None = None,
    linestyle: Any = "-",
) -> None:
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


def _save(filename: str | Path, open_pdf: bool) -> Path:
    filename = Path(filename)
    filename.parent.mkdir(parents=True, exist_ok=True)
    call_PDF(filename, open=open_pdf)
    return filename


def main(
    mLs: ArrayLike | None = None,
    couplings: Couplings | None = None,
    cutoff: Any = 10.0,
    emin: float = 1.9,
    emax: float = 6.0,
    open_pdf: bool = False,
) -> None:
    mLs = np.linspace(4.0, 7.0, 61) if mLs is None else np.asarray(mLs, dtype=float)
    couplings = couplings or Couplings.two_four(
        l22=20.0, l24=5000.0, l44=1.0, l33=0.0
    )
    result = scan_volumes(mLs, couplings, emin=emin, emax=emax, cutoff=cutoff)
    print(f"[info] {spectrum_figure(result, open_pdf=open_pdf)}")
    print(f"[info] {content_figure(result, open_pdf=open_pdf)}")


if __name__ == "__main__":
    main()
