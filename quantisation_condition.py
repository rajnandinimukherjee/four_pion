import os
import pdb

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import root_scalar
from tqdm import tqdm

from integrals import FourPionIntegral, TwoPionIntegral
from plot_settings import plotparams
from utils import call_PDF
from scipy.linalg import eig
from S4_utils import Q_tilde

plt.rcParams.update(plotparams)


def Imtx(E, I2, I4):
    E = np.atleast_1d(E)

    I2_vals = I2(E)
    I4_vals = I4(E)

    result = np.zeros((len(E), 2, 2))
    result[:, 0, 0] = I2_vals
    result[:, 1, 1] = I4_vals

    return result[0] if np.size(E) == 1 else result


class QC:

    def __init__(self, B, mL, Emin=1.8, Emax=6.0,
                 EMAX=20, alpha=0.1):
        self.mL = mL
        self.tpiL = 2*np.pi/mL
        self.EMAX, self.alpha = EMAX, alpha
        self.Emin, self.Emax = Emin, Emax

        self.I2 = TwoPionIntegral(mL, EMAX, alpha)
        self.FVE2 = self.FVE_twoptc()
        self.set_B(B)

    def set_B(self, B):
        self.B = B
        self.invB = np.linalg.inv(B)
        if "I4" in self.__dict__:
            self.I4.shift_levels(g22=B[0, 0])
        else:
            self.I4 = FourPionIntegral(self.mL, g22=B[0, 0],
                                       EMAX=self.EMAX, alpha=self.alpha)
        self.FVE4 = self.FVE_fourptc()
        self.NI = sorted(self.FVE2+self.FVE4)

    def qc(self, E):
        S = Imtx(E, self.I2, self.I4)
        invB = self.invB[None, :, :] if S.ndim == 3 else self.invB
        mats = invB - S
        return mats

    def qc2(self, E):
        invS = np.linalg.inv(Imtx(E, self.I2, self.I4))
        B = self.B[None, :, :] if invS.ndim == 3 else self.B
        mats = invS - B
        return mats

    def __call__(self, E):
        mats = self.qc(E)
        dets = np.linalg.det(mats)
        return dets

    def plot_Edep(self, Erange, yrange=None):
        qcs = np.linalg.det(self.qc(Erange))
        fig, ax = plt.subplots(1, 1)
        ax.plot(Erange, qcs)
        ymin, ymax = ax.get_ylim()
        ax.vlines(self.NI, ymin, ymax, color='k', alpha=0.1)
        yrange = yrange or [ymin, ymax]
        ax.set_ylim(yrange)
        ax.set_xlim([Erange[0], Erange[-1]])
        ax.set_xlabel(r"$E/m_\pi$")
        ax.set_ylabel(r"Re[det$(B^{-1}-I)$]")
        call_PDF(f"qc_L{self.L}.pdf")

    def partialE(self, E, dE=1e-6, **kwargs):
        f_E = self.qc(E, **kwargs)
        lams1, evecs1 = eig(f_E.real)

        f_E_ep = self.qc(E+dE, **kwargs)
        lams2, evecs2 = eig(f_E_ep.real)

        i, j = np.argmin(lams1), np.argmin(lams2)
        dlamE = (lams2[j]-lams1[i])/dE
        evec = evecs1[i]
        return dlamE.real, evec

    def FVE_twoptc(self):
        energies = self.I2.energies
        cutoff = energies[(self.Emin < energies) & (energies < self.Emax)]
        return np.sort(cutoff).tolist()

    def FVE_fourptc(self):
        energies = self.I4.energies
        cutoff = energies[(self.Emin < energies) & (energies < self.Emax)]
        return cutoff.tolist()


def plotRoots(mLs, roots, FVE2, FVE4, gs, Emin, Emax,
              EMAX=20, alpha=0.1, legend=False,
              plot_style="ko-", figsize=(3, 6),
              lw=0.1, I_energies=None):
    N_L = len(mLs)
    fig, ax = plt.subplots(1, 1, figsize=figsize)

    Two_NI_energies = get_energy_levels(mLs, FVE2)
    Four_NI_energies = get_energy_levels(mLs, FVE4)

    for level, energies in Two_NI_energies.items():
        if level == 0 and legend:
            ax.plot(mLs, energies, c='r', alpha=0.3, label=r"$2\pi$")
        else:
            ax.plot(mLs, energies, c='r', alpha=0.3)
    for level, energies in Four_NI_energies.items():
        if level == 0 and legend:
            ax.plot(mLs, energies, c='b', alpha=0.3, label=r"$4\pi$")
        else:
            ax.plot(mLs, energies, c='b', alpha=0.3)

    try:
        I_energies = I_energies or \
            get_energy_levels(mLs, roots, correct_skips=True)
        for level, energies in I_energies.items():
            if not (energies == np.nan).all() and level < 13:
                ax.plot(mLs, energies, plot_style, lw=lw, ms=2)
    except RuntimeError:
        for i, L in enumerate(mLs):
            ys = roots[i]
            ax.scatter([L]*len(ys), ys, c='k', s=1)

    ax.set_ylabel(r"$E/m_\pi$", size=12)
    ax.set_xlabel(r"$m_\pi L$", size=12)
    gstr_ = ', '.join(gs)
    ax.set_title(f"$gs = ({gstr_}), EMAX={EMAX}, alpha={alpha}$")
    ax.set_ylim([Emin, Emax])
    ax.set_xlim([mLs[0], mLs[-1]])
    if legend:
        ax.legend(loc="center", bbox_to_anchor=(
            4.5, 2.5), bbox_transform=ax.transData)

    gstr = f"g{'_'.join(gs)}"
    call_PDF(f"plots/EFV_{gstr}_NL{N_L}.pdf")


def find_roots(xs, func, NI_levels=[], tol=1e-3, verbose=False):
    roots = []
    ys = func(xs)
    sign_changes = np.where(ys[:-1]*ys[1:] < 0)[0]
    Ebar = tqdm(sign_changes, leave=False, desc="roots")
    for i in Ebar:
        sol = root_scalar(
            func, bracket=[xs[i], xs[i+1]], method='brentq')
        roots.append(sol.root)
        Ebar.set_description(f"{len(roots)} root(s) found")
    if verbose:
        print(f"[info] {len(sign_changes)} sign changes found")

    if len(NI_levels) > 0:
        root_arr, NI_arr = np.array(roots), np.array(NI_levels)
        mask = np.all(np.abs(root_arr[:, None] - NI_arr) > tol, axis=1)
        if verbose:
            print(f"[info] {len(root_arr)-np.count_nonzero(mask)} removed")
            print(f"[info] {np.count_nonzero(mask)} roots found")
        return root_arr[mask].tolist()
    else:
        return roots


def get_energy_levels(Ls, roots, correct_skips=False):
    max_len = max(len(lst) for lst in roots)
    energies = np.full((len(Ls), max_len), np.nan, dtype=float)
    for i, lst in enumerate(roots):
        new_list = np.array(lst)
        if len(new_list) > 2:
            new_list = new_list[np.concatenate(
                ([True], np.diff(new_list) > 1e-8))]
        energies[i, :len(new_list)] = new_list

    if correct_skips:
        for lvl in range(max_len):
            diffs = np.diff(energies[:, lvl], axis=0)
            sign_changes = np.where(diffs[:-1]*diffs[1:] < 0)[0]
            iters = 0
            level_end = False
            while len(sign_changes) > 0 and not level_end:
                print(f"[info] {len(sign_changes)
                                } sign changes in level {lvl}")
                outlier_idx = sign_changes[0]+2
                if outlier_idx+1 == len(Ls):
                    energies[outlier_idx, lvl +
                             1:] = energies[outlier_idx, lvl:-1]
                    energies[outlier_idx, lvl] = np.nan
                    level_end = True
                else:
                    signs = np.sign(np.diff(
                        np.delete(energies[:, lvl], outlier_idx)))
                    if np.any(signs[outlier_idx-1:] == -1.0):
                        next_idx = outlier_idx + 1 + np.where(
                            signs[outlier_idx-1:] == -1.0)[0][0]
                        prev_ = energies[outlier_idx-1, lvl]
                        next_ = energies[next_idx, lvl]
                        if np.sign(next_-prev_) != -1.0:
                            energies[outlier_idx:, lvl +
                                     1:] = energies[outlier_idx:, lvl:-1]
                            energies[outlier_idx:, lvl] = np.nan
                            print(f"[info] level {lvl} ended at L={
                                  Ls[outlier_idx]}")
                            level_end = True
                        else:
                            if lvl+1 < max_len:
                                energies[outlier_idx, lvl +
                                         1:] = energies[outlier_idx, lvl:-1]
                                print(f"[info] {energies[outlier_idx, lvl+1]} between {prev_} and {
                                      next_} pushed up, replaced with {energies[outlier_idx, lvl]}")
                            else:
                                print(f"[info] {energies[outlier_idx, lvl]} between {prev_} and {
                                      next_} replaced with {energies[outlier_idx, lvl]}")
                            energies[outlier_idx, lvl] = (prev_+next_)/2
                    else:
                        energies[outlier_idx:, lvl +
                                 1:] = energies[outlier_idx:, lvl:-1]
                        energies[outlier_idx:, lvl] = np.nan
                        print(f"[info] level {lvl} ended at L={
                              Ls[outlier_idx]}")
                        level_end = True

                diffs = np.diff(energies[:, lvl], axis=0)
                sign_changes = np.where(diffs[:-1]*diffs[1:] < 0)[0]
                iters += 1
                if iters > max_len:
                    print(
                        "[info] ran into an issue, giving up on connecting levels.")
                    raise RuntimeError
            print(f"[info] finished correcting level {
                  lvl} in {iters} iterations")
    result = {lvl: energies[:, lvl] for lvl in range(max_len)}
    for lvl, vals in result.copy().items():
        if (vals == np.nan).all():
            del result[lvl]

    return result


def B(gs):
    gs_ = list(map(float, gs))
    return np.array([[gs_[0], gs_[1]],
                     [gs_[1], gs_[2]]])


def fine_grid(levels, scale, N_E=1000, nmax=13):
    levels = np.asarray(levels, dtype=float)
    assert len(levels) >= 1, "provide levels to fine grid around"
    Es = [np.linspace(h, h + 10*scale, N_E) for h in levels[:nmax]]
    return np.unique(np.concatenate(Es))


def compute_roots(mLs, gs,
                  Emin=1.8, Emax=6.0,
                  N_EC=2000, ctol=1e-8,
                  N_EF=1000, ftol=1e-12,
                  save=True, load=False,
                  plot=False, verbose=True, **kwargs):
    N_L = len(mLs)
    gstr = gs if type(gs[0]) == str else [str(g) for g in gs]
    if verbose:
        print(f"[info] gs={gstr}")
    gstr_ = f"g{'_'.join(gstr)}"
    fname = f"data/roots_{gstr_}_NE{N_EC}_NL{N_L}_Emin{Emin}_Emax{Emax}.npy"

    if load:
        if os.path.exists(fname):
            roots, FVE2, FVE4 = np.load(fname, allow_pickle=True)
            print(f"[info] loaded roots from {fname}")
            if plot:
                plotRoots(mLs, roots, FVE2, FVE4, gs,
                          Emin, Emax, legend=False)
            return roots, FVE2, FVE4

    roots = np.empty(N_L, dtype=object)
    FVE2 = np.empty(N_L, dtype=object)
    FVE4 = np.empty(N_L, dtype=object)
    # with tqdm(mLs, desc="Volumes", dynamic_ncols=True, leave=False) as Lbar:
    #    for Li, L in enumerate(Lbar):
    #        Lbar.set_description(f"L={L:.2f}")
    for Li, mL in enumerate(mLs):
        qc = QC(B(gs), mL, Emin=Emin, Emax=Emax, **kwargs)
        coarse_search = find_roots(np.linspace(Emin, Emax, N_EC),
                                   qc, qc.NI, tol=ctol, verbose=verbose)
        if N_EF > 0:
            scale = B(gs)[1, 1] / (mL**9)
            fine_search = find_roots(fine_grid(qc.FVE4, scale, N_E=N_EF),
                                     qc, qc.NI, tol=ftol, verbose=verbose)
        else:
            fine_search = []
        roots[Li] = sorted(set(coarse_search) | set(fine_search))
        FVE2[Li], FVE4[Li] = qc.FVE2, qc.FVE4
    if save:
        np.save(fname, (roots, FVE2, FVE4))
        print(f"[info] saved roots to {fname}")

    if plot:
        plotRoots(mLs, roots, FVE2, FVE4, gs, Emin, Emax, legend=False)
    return roots, FVE2, FVE4


def loop_gs(mLs, g24=["1e+3"], g44=["3e+4"], **kwargs):
    for g1 in g24:
        for g2 in g44:
            print(f"[info] g24={g1}, g44={g2}")
            gs = ["1.0", g1, g2]
            compute_roots(mLs, gs, **kwargs)


if __name__ == "__main__":
    EMAX = 20
    mLs = np.linspace(4, 7, 61)

    Emin, Emax = 1.9, 6.0
    gs = ["20", "5000", "1"]

    roots, FVE2, FVE4 = compute_roots(
        mLs, gs, Emin=1.9, Emax=6.1,
        N_EC=1000, N_EF=500,
        load=True, plot=False)
    plotRoots(mLs, roots, FVE2, FVE4, gs,
              Emin=1.98, Emax=6.0, plot_style="k-",
              figsize=(3, 8), lw=0.8)

    # gs = [1.0, 0.0, 1.0]
    # mL = 6.0
    # qc = QC(B(gs), N, mL=mL, shift=True, cutoff="smooth")

    # N_EC, N_EF = 4000, 2000
    # print("coarse search")
    # coarse_search = find_roots(np.linspace(Emin, Emax, N_EC),
    #                           qc, qc.NI, tol=1e-8, verbose=True)
    # print("fine search")
    # fine_search = find_roots(fine_grid(qc.FVE4, B(gs)[1, 1], mL=mL, N_E=N_EF),
    #                         qc, qc.NI, tol=1e-10, verbose=True)
    # roots = sorted(set(coarse_search) | set(fine_search))
    # print(f"expected {len(qc.NI)} levels, found {len(roots)} levels")
