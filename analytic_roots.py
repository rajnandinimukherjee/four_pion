import pdb

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from utils import call_PDF

from quantisation_condition import QC, B


def plotLevels(Ls, I_levels, NI2_levels, NI4_levels,
               Emin=1.98, Emax=6.0, figsize=(3, 6), N=6,
               filename=None, **kwargs):
    fig, ax = plt.subplots(1, 1, figsize=figsize)
    ax.set_ylabel(r"$E_{FV}/m_\pi$", size=16)
    ax.set_xlabel(r"$m_\pi L$", size=16)
    for lvl in range(NI2_levels.shape[1]):
        ax.plot(Ls, NI2_levels[:, lvl], "r-", alpha=0.5, lw=0.8)
    for lvl in range(NI4_levels.shape[1]):
        ax.plot(Ls, NI4_levels[:, lvl], "b-", alpha=0.5, lw=0.8)
    for lvl in range(I_levels.shape[1]):
        ax.plot(Ls, I_levels[:, lvl], "k-", lw=0.8)
    ax.set_ylim([Emin, Emax])
    ax.set_xlim([Ls[0], Ls[-1]])
    filename = filename or f"plots/EFV_N{N}_analytic.pdf"
    call_PDF(filename)


def energies_to_levels(energies, N_L):
    max_len = max(len(lst) for lst in energies)
    levels = np.full((N_L, max_len), np.nan, dtype=float)
    for i, lst in enumerate(energies):
        new_list = np.array(lst)
        levels[i, :len(new_list)] = new_list
    return levels


def solve_FV_energies(gs, L, N=6, Emin=1.9, Emax=6.0, cutoff="hard", **kwargs):
    g22, g24, g44 = gs
    qc = QC(B(gs), N, L, cutoff=cutoff, Emin=Emin,
            Emax=Emax, shift=True)
    NI2, NI4 = qc.FVE2, qc.FVE4
    C2 = (qc.I2.coeffs/(1-g22*qc.I2.remainders()))[:len(NI2)]
    C4 = qc.I4.coeffs[:len(NI4)]/(1-g44*qc.I4.remainders(max_idx=len(NI4)))
    I2 = NI2 + g22*C2
    I4 = NI4 + g44*C4
    V = g24*np.outer(C2, C4)**0.5
    H = np.block([[np.diag(I2), V], [V.T, np.diag(I4)]])
    evals, evecs = np.linalg.eigh(H)
    return evals, NI2, NI4


def levels_Lrange(gs, Ls, **kwargs):
    I_energies, NI2_energies, NI4_energies = [], [], []
    for L in tqdm(list(Ls), desc="L", leave=False):
        I, NI2, NI4 = solve_FV_energies(gs, L, **kwargs)
        I_energies.append(list(I))
        NI2_energies.append(NI2)
        NI4_energies.append(NI4)

    NI2_levels = energies_to_levels(NI2_energies, len(Ls))
    NI4_levels = energies_to_levels(NI4_energies, len(Ls))
    I_levels = energies_to_levels(I_energies, len(Ls))
    plotLevels(Ls, I_levels, NI2_levels, NI4_levels, **kwargs)


if __name__ == "__main__":
    # gs = [2.0, (48**0.5)*1e+3, 24*5e+4]
    gs = [2.0, 0.0, 24*5e+4]
    Ls = np.linspace(4, 7, 31)
    levels_Lrange(gs, Ls)
