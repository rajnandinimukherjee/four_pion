import numpy as np
import pdb
from quantisation_condition import QC, B, find_roots, fine_grid
from scipy.optimize import minimize
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)


mL_init, mL_comp = 10, 5
EMAX = 20


def get_roots(qc, Emin=1.98, Emax=4.0,
              N_EC=1000, N_EF=200,
              tol=1e-10, verbose=False):
    NI = np.array(qc.NI)
    NI = NI[(NI > Emin) & (NI < Emax)]

    # coarse grid
    CG = np.linspace(Emin, Emax, N_EC)

    # fine grid
    scale = 10*qc.B[1, 1]/(qc.mL**9)
    FG = fine_grid(NI, scale, N_E=N_EF)
    FG = FG[(FG > Emin) & (FG < Emax)]
    if verbose:
        print(f"[info] mL={qc.mL}, EMAX={qc.EMAX}, alpha={qc.alpha}")
        print(f"[info] g22={qc.B[0, 0]}, g24={qc.B[0, 1]}, g44={qc.B[1, 1]}")
        print(f"[info] {len(qc.I2.levels)} levels in S2, {
              len(qc.I4.levels)} levels in S4")
        print(f"[info] expecting {len(NI)} roots between {Emin} and {Emax}")
        print(f"[info] using Egrid with {
              len(CG)} coarse and {len(FG)} fine points")

    CG_roots = find_roots(CG, qc, NI, tol=tol, verbose=verbose)
    FG_roots = find_roots(FG, qc, NI, tol=tol, verbose=verbose)
    roots = list(sorted(set(CG_roots) | set(FG_roots)))
    if len(roots) < len(NI):
        print("not all roots founds")
        pdb.set_trace()
    elif len(roots) > len(NI):
        print("too many roots found")
        pdb.set_trace()
    return np.array(roots)


alpha1 = 0.1
gs1 = np.array([20, 5000, 1])
qc1 = QC(B(gs1), mL_init, alpha=alpha1, EMAX=EMAX)
E1 = get_roots(qc1, verbose=True)

alpha2 = 0.2
qc2 = QC(B(gs1), mL=mL_init, alpha=alpha2, EMAX=EMAX)


def det(qc, gs, Es):
    g22, g24, g44 = gs
    qc.set_B(B(gs))
    S2, S4 = qc.I2(Es), qc2.I4(Es)
    return (1-S2*g22)*(1-S4*g44) - S2*S4*g24**2


def loss(gs2, alpha_root=10.0, beta_mid=1.0, kappa=10.0):
    root_loss = np.sum(det(qc2, gs2, E1)**2)

    E_mids = 0.5*(E1[:-1]+E1[1:])
    mids1 = det(qc1, gs1, E_mids)
    mids2 = det(qc2, gs2, E_mids)
    sign1 = np.tanh(kappa*mids1)
    sign2 = np.tanh(kappa*mids2)
    mid_loss = np.sum((sign1-sign2)**2)

    return alpha_root*root_loss+beta_mid*mid_loss


# bounds?
result = minimize(loss, gs1, method='L-BFGS-B',
                  tol=1e-12)
print(result)
gs2 = result.x
qc2.set_B(B(gs2))
E2 = get_roots(qc2, verbose=True)
print("\n===================================================")
print(
    f"at mL={mL_init}:\n"
    f"E(alpha1={alpha1}, g1={', '.join(f'{g:.6g}' for g in gs1)}) = "
    f"E(alpha2={alpha2}, g2={', '.join(f'{g:.6g}' for g in gs2)})"
)
for ri in range(len(E1)):
    print(f"{E1[ri]:.10g}\t{E2[ri]:.10g}")

qc_comp_1 = QC(B(gs1), mL_comp, alpha=alpha1, EMAX=EMAX)
E_comp_1 = get_roots(qc_comp_1, Emin=1.98, Emax=6.0, verbose=False)
qc_comp_2 = QC(B(gs2), mL_comp, alpha=alpha2, EMAX=EMAX)
E_comp_2 = get_roots(qc_comp_2, Emin=1.98, Emax=6.0, verbose=False)
print(
    f"at mL={mL_comp}:\n"
    f"E(alpha1, g1)\tE(alpha2, g2)\t % change"
)
percs = np.abs(E_comp_1-E_comp_2)*100/E_comp_1
for ri in range(len(E_comp_1)):
    print(f"{E_comp_1[ri]:.10g}\t{E_comp_2[ri]:.10g}\t{percs[ri]:.4g}%")
