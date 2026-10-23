"""Command-line interface for the perturbative finite-volume analyses."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Any, Sequence

SECTORS_HELP = "particle-number sectors to couple (default: 2 4)"


def _add_project_directory(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--project-dir",
        type=Path,
        default=Path.cwd(),
        help="directory containing the analysis data/ and plots/ folders",
    )


def _add_scan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--mL-range",
        nargs=2,
        type=float,
        metavar=("START", "STOP"),
        default=(4.0, 7.0),
        help="inclusive endpoints of the mL scan (default: 4 7)",
    )
    parser.add_argument(
        "--mL-count", type=int, default=61, help="number of mL values (default: 61)"
    )
    parser.add_argument(
        "--couplings",
        nargs="+",
        metavar="LAMBDA",
        default=("20", "5000", "1", "0"),
        help="lambda_{2<-2} lambda_{2<-4} lambda_{4<-4} [lambda_{3<-3}]",
    )
    parser.add_argument(
        "--energy-range",
        nargs=2,
        type=float,
        metavar=("MIN", "MAX"),
        default=(1.9, 6.0),
        help="energy window searched for solutions (default: 1.9 6)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=10.0,
        help="energy cutoff EMAX on the sums inside G (default: 10)",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.0,
        help="smooth damping exp(alpha (E - E_pole)) of the cutoff (default: 0)",
    )
    parser.add_argument("--sectors", nargs="+", type=int, default=(2, 4),
                        help=SECTORS_HELP)
    parser.add_argument("--coarse-points", type=int, default=2000)
    parser.add_argument(
        "--delta",
        type=float,
        default=1e-11,
        help="half-width of the bracket isolating each pole of G",
    )
    parser.add_argument("--data-dir", type=Path, default=Path("data"))


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="four-pion-analysis",
        description="Generate the finite-volume basis and solve the "
        "quantization condition det[G^{-1} - lambda] = 0.",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    orbits_parser = subparsers.add_parser(
        "orbits", help="generate O_h orbit representatives for a sector"
    )
    _add_project_directory(orbits_parser)
    orbits_parser.add_argument("--N", type=int, required=True)
    orbits_parser.add_argument("--emax", type=float, help="energy cutoff at --mL")
    orbits_parser.add_argument(
        "--mL", type=float, help="reference mL; use the largest mL of the scan"
    )
    orbits_parser.add_argument("--nsq-max", type=int, help="momentum cutoff n.n")
    orbits_parser.add_argument("--data-dir", type=Path, default=Path("data"))
    orbits_parser.add_argument("--output", type=Path)
    orbits_parser.add_argument("--no-disconnected", action="store_true")
    orbits_parser.add_argument("--quiet", action="store_true")
    orbits_parser.set_defaults(handler=_run_orbits)

    spectrum_parser = subparsers.add_parser(
        "spectrum", help="solve the quantization condition over an mL scan"
    )
    _add_project_directory(spectrum_parser)
    _add_scan_arguments(spectrum_parser)
    spectrum_parser.add_argument("--no-save", action="store_true")
    spectrum_parser.add_argument(
        "--recompute", action="store_true", help="ignore a cached scan"
    )
    spectrum_parser.add_argument("--plot", action="store_true")
    spectrum_parser.add_argument("--output", type=Path, help="PDF output path")
    spectrum_parser.add_argument("--no-open", action="store_true")
    spectrum_parser.add_argument("--quiet", action="store_true")
    spectrum_parser.set_defaults(handler=_run_spectrum)

    figures_parser = subparsers.add_parser(
        "figures", help="produce the section 3.1 spectrum and content figures"
    )
    _add_project_directory(figures_parser)
    _add_scan_arguments(figures_parser)
    figures_parser.add_argument("--no-open", action="store_true")
    figures_parser.set_defaults(handler=_run_figures)

    diagonalise_parser = subparsers.add_parser(
        "diagonalise",
        help="finite-volume levels by diagonalizing the truncated Hamiltonian",
    )
    _add_project_directory(diagonalise_parser)
    _add_scan_arguments(diagonalise_parser)
    diagonalise_parser.add_argument("--output", type=Path)
    diagonalise_parser.add_argument("--no-open", action="store_true")
    diagonalise_parser.set_defaults(handler=_run_diagonalise)

    check_parser = subparsers.add_parser(
        "check", help="run the analytic checks of the construction"
    )
    _add_project_directory(check_parser)
    check_parser.set_defaults(handler=_run_checks)

    return parser


def _scan_configuration(args: argparse.Namespace) -> tuple[Any, Any, float, float]:
    import numpy as np

    from gfunctions import Couplings

    mLs = np.linspace(*args.mL_range, args.mL_count)
    couplings = Couplings.parse(args.couplings)
    emin, emax = args.energy_range
    return mLs, couplings, emin, emax


def _scan(args: argparse.Namespace, load: bool = True) -> dict[str, Any]:
    from quantisation_condition import scan_volumes

    mLs, couplings, emin, emax = _scan_configuration(args)
    return scan_volumes(
        mLs,
        couplings,
        emin=emin,
        emax=emax,
        cutoff=args.cutoff,
        alpha=args.alpha,
        sectors=tuple(args.sectors),
        n_coarse=args.coarse_points,
        delta=args.delta,
        save=not getattr(args, "no_save", False),
        load=load and not getattr(args, "recompute", False),
        data_dir=args.data_dir,
        verbose=not getattr(args, "quiet", False),
    )


def _run_orbits(args: argparse.Namespace) -> None:
    from orbits import generate, orbit_filename

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


def _run_spectrum(args: argparse.Namespace) -> None:
    from quantisation_condition import plot_spectrum

    result = _scan(args)
    counts = [len(roots) for roots in result["roots"]]
    print(f"[info] {min(counts)}-{max(counts)} solutions per volume")
    if args.plot:
        path = plot_spectrum(
            result, filename=args.output, open_pdf=not args.no_open
        )
        print(f"[info] saved {path}")


def _run_figures(args: argparse.Namespace) -> None:
    from figures import content_figure, spectrum_figure

    result = _scan(args)
    print(f"[info] saved {spectrum_figure(result, open_pdf=not args.no_open)}")
    print(f"[info] saved {content_figure(result, open_pdf=not args.no_open)}")


def _run_diagonalise(args: argparse.Namespace) -> None:
    from analytic_roots import plot_levels, scan_volumes

    mLs, couplings, emin, emax = _scan_configuration(args)
    result = scan_volumes(
        mLs,
        couplings,
        emin=emin,
        emax=emax,
        cutoff=args.cutoff,
        sectors=tuple(args.sectors),
        alpha=args.alpha,
        data_dir=str(args.data_dir),
    )
    plot_levels(
        result,
        filename=str(args.output or "plots/levels_diagonalised.pdf"),
        open_pdf=not args.no_open,
    )


def _run_checks(_args: argparse.Namespace) -> None:
    from checks import main as run

    raise SystemExit(run())


def main(argv: Sequence[str] | None = None) -> int:
    """Run a command from the directory holding its data and output files."""
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.project_dir.is_dir():
        parser.error(f"project directory does not exist: {args.project_dir}")
    os.chdir(args.project_dir)
    try:
        args.handler(args)
    except FileNotFoundError as missing:
        parser.exit(2, f"[error] {missing}\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
