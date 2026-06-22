from quantisation_condition import get_energy_levels, B, QC
import numpy as np
from utils import call_PDF
import matplotlib.pyplot as plt
from plot_settings import plotparams
from integrals import FourPionIntegral
plt.rcParams.update(plotparams)


N = 6
mLs = np.linspace(4, 7, 61)
N_L = len(mLs)
Emin, Emax = 1.9, 6.1
N_EC, N_EF = 1000, 500

gs = ["20", "5000", "1"]
print(f"[info] gs={gs}")
gstr = f"g{'_'.join(gs)}"
fname = f"data/roots_N{N}_{gstr}_NE{N_EC}_NL{N_L}_Emin{Emin}_Emax{Emax}.npy"

roots, FVE2, FVE4 = np.load(fname, allow_pickle=True)
print(f"[info] loaded roots from {fname}")

I4 = FourPionIntegral(mL=6)
quads = I4.levels[:20]
FVE4_NI = np.zeros(len(mLs), dtype=object)
for L_idx, L in enumerate(mLs):
    tpiL = 2*np.pi/L
    FVE4_NI[L_idx] = np.array([np.sum((q*tpiL**2+1)**0.5)
                              for q in quads], dtype=float)


fname2 = f"data/energies_splined_N{N}_{gstr}_NE{
    N_EC}_NL{N_L}_Emin{Emin}_Emax{Emax}.npy"
new_energies = np.load(fname2, allow_pickle=True).item()

Emin, Emax = 1.98, 6.0
MAX_LVLS = 13
# plot_style, figsize, lw = "k-", (3, 7), 1
plot_style, figsize, lw = "k-", (3, 6), 1
N_L = len(mLs)
fig, ax = plt.subplots(1, 1, figsize=figsize)

Two_NI_energies = get_energy_levels(mLs, FVE2)
Four_NI_g22_energies = get_energy_levels(mLs, FVE4)
Four_NI_energies = get_energy_levels(mLs, FVE4_NI)

for level, energies in Two_NI_energies.items():
    if level == 0:
        ax.plot(mLs, energies, c='r', alpha=0.3, label=r"2-ptc NI level")
    else:
        ax.plot(mLs, energies, c='r', alpha=0.3)
for level, energies in Four_NI_energies.items():
    if level == 0:
        ax.plot(mLs, energies, c='b', alpha=0.1, label=r"4-ptc NI level")
    else:
        ax.plot(mLs, energies, c='b', alpha=0.1)
for level, energies in Four_NI_g22_energies.items():
    if level == 0:
        ax.plot(mLs, energies, c='b', alpha=0.4,
                label=r"4-ptc with $2\to2$")
    else:
        ax.plot(mLs, energies, c='b', alpha=0.4)

for level, energies in new_energies.items():
    if not (energies == np.nan).all() and level < MAX_LVLS:
        if level == 0:
            ax.plot(mLs, energies, plot_style, lw=lw,
                    ms=2, label="interacting level")
        else:
            ax.plot(mLs, energies, plot_style, lw=lw, ms=2)

L_star = 4.5
idx = list(mLs).index(L_star)
E_star = new_energies[2][idx]
plt.scatter([L_star], E_star, marker="*", s=40, c="k",
            label=r"$E^{(n=2)}(m L=4.5)$")

ax.set_ylabel(r"$E/m$", size=13)
ax.set_xlabel(r"$mL$", size=13)
gstr_ = r"$g_{2\to 2}=0$, $g_{2\leftrightarrow 4}=" +\
    r"0$, $g_{4\to4}=0$"
gstr_ = r"$g_{2\to 2}="+gs[0]+r"$, $g_{2\leftrightarrow 4}=" +\
    gs[1]+r"$, $g_{4\to4}="+gs[2]+r"$"
reg_str = r"$\left\lfloor\frac{L}{2\pi}\Lambda\right\rfloor=" +\
    str(N)+r"$, $m\alpha=1.0$"
ax.set_title(gstr_+"\n"+reg_str, size=11)
# ax.set_title(gstr_, size=11)
ax.set_ylim([0.98*Emin, Emax])
ax.set_xlim([mLs[0], mLs[-1]])
ax.legend(loc="center", bbox_to_anchor=(
    5.0, 2.5), bbox_transform=ax.transData)


call_PDF(f"plots/EFV_N{N}_{gstr}_NL{N_L}.pdf")
