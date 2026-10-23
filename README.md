# Perturbative finite-volume analysis

Numerics for the quantization condition of the draft in `paper/`,

```
det[ G(E, L)^{-1} - lambda ] = 0 ,
```

where `lambda` holds the effective couplings `lambda_{N'<-N}` and `G` the
finite-volume functions

```
G_{N'N}(E, L) = <u_{N'}| Ghat(E) |u_N> ,
u_mu = L^{-3(N-1)/2} sqrt( d_mu / (N_mu Omega_mu) ) .
```

The code works in the rest frame and the `A_1^+` irrep, so the basis is the set
of `O_h` orbits `mu` of `N`-particle momentum configurations with `P = 0`
(appendix A.3).  At leading order `Ghat` follows from degenerate perturbation
theory in the disconnected potential `V^d_L` inside each non-interacting level,

```
G_{NN}(E, L) = sum_levels sum_{mu,mu' in level}
               u_mu' [ (E - E_level) 1 - V^d_level ]^{-1}_{mu' mu} u_mu .
```

Energies are in units of the mass and momenta are integer vectors `n` with
`k = (2 pi / L) n`.

## Modules

| module | role |
| --- | --- |
| `orbits.py` | **standalone generator.** Enumerates `B_N(0)`, partitions it into `O_h` orbits, stores `d_mu`, `N_mu`, the degenerate levels, and the volume-independent part of `<b_mu'\|V^d_L\|b_mu>` (eq. A.49). Everything in the file is independent of `L` and of the couplings. |
| `gfunctions.py` | Turns an orbit table into `G_{NN}(E, L)`: the vertex factors `u_mu`, the `V^d` block of each level, its diagonalization into poles and residues, and the matrix `G` in the space of particle numbers. Also defines `Couplings`. |
| `quantisation_condition.py` | **analysis script.** Solves `det[G^{-1} - lambda] = 0` at one volume or over an `mL` scan, gives the particle-number content of each state, caches scans and plots them. |
| `figures.py` | The two figures of section 3.1: the spectrum, and the four-particle content of each level. |
| `analytic_roots.py` | The same levels by diagonalizing the truncated Hamiltonian `H = H0 + V^d + \|u> lambda <u\|`, which the matrix determinant lemma says must agree with the roots. |
| `checks.py` | Analytic checks: the orbit sum rule, `G_22` against the direct momentum sum, the threshold expansions of section 3.1, the symmetry of `V^d`, and the roots against diagonalization. |

## Workflow

Generate the basis once per sector and cutoff.  The cutoff is an energy `EMAX`
imposed at a reference `mL`, which should be the **largest** `mL` of the
intended scan, so that every smaller volume is covered:

```bash
python orbits.py --N 2 --emax 10 --mL 7
python orbits.py --N 4 --emax 10 --mL 7
```

This writes `data/orbits_N{N}_EMAX10_mL7.npz`.  The `N = 4` table at this cutoff
holds 1683 orbits and about 1.5e5 disconnected contributions, and takes a few
seconds; the cost grows quickly with `EMAX`.

Then solve the quantization condition and draw the figures:

```bash
python quantisation_condition.py         # mL scan, cached under data/
python figures.py                        # plots/spec_24.pdf, plots/ME_24.pdf
python checks.py                         # analytic checks
```

The same workflows are available through the installed command:

```bash
python -m pip install -e .

four-pion-analysis orbits --N 4 --emax 10 --mL 7
four-pion-analysis spectrum --mL-range 4 7 --mL-count 61 \
    --couplings 20 5000 1 0 --energy-range 1.9 6 --cutoff 10 --plot
four-pion-analysis figures --couplings 20 5000 1 0 --no-open
four-pion-analysis check
```

`--couplings` takes `lambda_{2<-2} lambda_{2<-4} lambda_{4<-4}` and optionally
`lambda_{3<-3}`, which drives the disconnected three-particle sub-process inside
`G_44`.

## Conventions and caveats

* `EMAX` is a regulator.  States above it are smooth in `E` and are absorbed
  into the couplings (appendix B), so the couplings depend on it; `--alpha`
  replaces the hard cutoff by the smooth damping `exp(alpha (E - E_pole))`.
* Levels are identified by the multiset of squared norms, which is exact at
  every `L`.  Accidental degeneracies between different multisets, which occur
  at isolated `L`, are not resolved.
* Only particle-number-conserving disconnected sub-processes are tabulated.
  Sub-processes that change the particle number contribute to `G` only where
  levels of different particle number cross, which happens at isolated volumes.
* The root search uses the pole-free form `det[1 - G lambda]`, brackets every
  pole of `G` within `+-delta`, and discards the brackets that hold a pole.
  `delta` must be smaller than the smallest level shift to be resolved, of order
  `lambda_{4<-4} / (384 (mL)^9)` for a four-particle level.
