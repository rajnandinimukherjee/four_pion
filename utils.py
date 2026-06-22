import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_pdf import PdfPages
from plot_settings import plotparams

plt.rcParams.update(plotparams)


def build_omega_3D_array(L, N):
    """
    Build a list of momentum vectors k = 2*pi*n/L
    n ranges over [-N, N]^3.
    """
    ns = range(-N, N+1)
    nx, ny, nz = np.meshgrid(ns, ns, ns, indexing='ij')
    k = (2*np.pi/L) * np.stack((nx, ny, nz), axis=-1)
    omegas = (np.sum(k**2, axis=-1) + 1)**0.5
    return omegas, k


def momentum_to_index(P, L, N):
    """
    Map a physical momentum vector P (3-tuple) to array index (ix,iy,iz).
    Requires P to lie on the discrete grid: P = 2*pi * p / L with integer p.
    """
    tpiL = 2*np.pi/L
    p = np.array(P, dtype=float) / tpiL
    p_int = np.rint(p).astype(int)   # round to nearest integer
    if not np.allclose(p, p_int, atol=1e-10):
        raise ValueError("Total momentum P is not on the discrete grid.")
    idx = tuple((p_int + N).tolist())
    for a in idx:
        if a < 0 or a > 2*N:
            raise IndexError("Index for P is outside grid (increase N).")
    return idx


def set_sgrid(eps, omega_max, theta, tol=1e-12, spp=20):
    """
    Heuristics to build a real s-grid for the rotated contour:
      - s_max chosen so exp(-eps * s_max * cos(theta)) ~ tol  (simple heuristic)
      - ds chosen to resolve fastest oscillation: ds << 2*pi / (omega_max * cos(theta))
        because omega * Re(t_complex) ~ omega * s * cos(theta) sets oscillation frequency in s.
    """
    if eps <= 0:
        raise ValueError("eps must be positive for damping.")
    # Heuristic for s_max: require exp(-eps * s_max * cos(theta)) ~ tol
    cos_theta = np.cos(theta)
    if cos_theta <= 0:
        raise ValueError("theta must be in (0, pi/2) so cos(theta)>0.")
    s_max = max(1.0, -np.log(tol) / (eps * cos_theta))
    # choose ds resolving fastest oscillation in s:
    # period in s approx 2*pi / (omega_max * cos(theta)), so samples_per_period = spp
    if omega_max <= 0:
        ds = 0.1
    else:
        ds = (2.0 * np.pi / (omega_max * cos_theta)) / spp
    Nt = int(np.ceil(s_max / ds)) + 1
    s = np.linspace(0.0, ds * (Nt - 1), Nt)
    return s, ds


def set_trange(eps, omega_max):
    # choose t_max so that e^{-eps * t_max} ~ 1e-10
    if eps <= 0:
        raise ValueError(
            "eps must be positive to ensure damping in time integral.")
    # clamp to avoid absurdly large
    t_max = min(200.0, max(20.0, -np.log(1e-12) / eps))
    # choose dt resolving fastest oscillation: dt << 2*pi/omega_max
    dt = (2.0*np.pi/omega_max) / 10.0
    Nt = int(np.ceil(t_max / dt)) + 1
    t = np.linspace(0.0, dt * (Nt - 1), Nt)
    return t, dt


def call_PDF(filename, open=True, **kwargs):
    pdf = PdfPages(filename)
    fig_nums = plt.get_fignums()
    figs = [plt.figure(n) for n in fig_nums]
    for fig in figs:
        fig.savefig(pdf, format="pdf")
    pdf.close()
    plt.close("all")
    if open:
        os.system("open " + filename)
