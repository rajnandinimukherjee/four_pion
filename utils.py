"""Shared momentum-grid, quadrature-grid, and figure helpers."""

from __future__ import annotations

import os
import subprocess
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from numpy.typing import ArrayLike, NDArray

from plot_settings import plotparams

Array = NDArray[Any]


plt.rcParams.update(plotparams)


def build_omega_3D_array(L: float, N: int) -> tuple[Array, Array]:
    """Return single-particle energies and momenta on the ``[-N, N]^3`` grid."""
    ns = range(-N, N + 1)
    nx, ny, nz = np.meshgrid(ns, ns, ns, indexing="ij")
    momenta = (2 * np.pi / L) * np.stack((nx, ny, nz), axis=-1)
    omegas = (np.sum(momenta**2, axis=-1) + 1) ** 0.5
    return omegas, momenta


def momentum_to_index(P: ArrayLike, L: float, N: int) -> tuple[int, ...]:
    """Map a grid momentum ``P`` to its ``(ix, iy, iz)`` array index."""
    tpiL = 2 * np.pi / L
    p = np.array(P, dtype=float) / tpiL
    p_int = np.rint(p).astype(int)
    if not np.allclose(p, p_int, atol=1e-10):
        raise ValueError("Total momentum P is not on the discrete grid.")

    index = tuple((p_int + N).tolist())
    if any(axis_index < 0 or axis_index > 2 * N for axis_index in index):
        raise IndexError("Index for P is outside grid (increase N).")
    return index


def set_sgrid(
    eps: float,
    omega_max: float,
    theta: float,
    tol: float = 1e-12,
    spp: int = 20,
) -> tuple[Array, float]:
    """Choose a real ``s`` grid for a rotated contour integral."""
    if eps <= 0:
        raise ValueError("eps must be positive for damping.")

    cos_theta = np.cos(theta)
    if cos_theta <= 0:
        raise ValueError("theta must be in (0, pi/2) so cos(theta)>0.")

    s_max = max(1.0, -np.log(tol) / (eps * cos_theta))
    if omega_max <= 0:
        ds = 0.1
    else:
        ds = (2.0 * np.pi / (omega_max * cos_theta)) / spp

    n_points = int(np.ceil(s_max / ds)) + 1
    s = np.linspace(0.0, ds * (n_points - 1), n_points)
    return s, ds


def set_trange(eps: float, omega_max: float) -> tuple[Array, float]:
    """Choose a time grid that resolves oscillations and exponential damping."""
    if eps <= 0:
        raise ValueError("eps must be positive to ensure damping in time integral.")

    t_max = min(200.0, max(20.0, -np.log(1e-12) / eps))
    dt = (2.0 * np.pi / omega_max) / 10.0
    n_points = int(np.ceil(t_max / dt)) + 1
    t = np.linspace(0.0, dt * (n_points - 1), n_points)
    return t, dt


def call_PDF(
    filename: str | os.PathLike[str], open: bool = True, **kwargs: Any
) -> None:
    """Save every open figure to a multi-page PDF and close the figures."""
    del kwargs  # Retained for compatibility with existing plotting calls.
    filename = os.fspath(filename)
    figures = [plt.figure(number) for number in plt.get_fignums()]

    with PdfPages(filename) as pdf:
        for figure in figures:
            figure.savefig(pdf, format="pdf")

    plt.close("all")
    if open:
        try:
            subprocess.run(["open", filename], check=False)
        except OSError:
            # Match the previous fire-and-forget behavior when no desktop opener
            # is available (for example, on a headless machine).
            pass
