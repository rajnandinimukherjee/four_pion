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
| `gfunctions.py` | Turns an orbit table into `G_{NN}(E, L)`: the vertex factors `u_mu`, the `V^d` block of each level, its diagonalization into poles and residues, and the matrix `G` in the space of particle numbers. Also defines `Couplings` and `Cutoff`, the regulator on the sums. |
| `quantisation_condition.py` | **analysis script.** Solves `det[G^{-1} - lambda] = 0` at one volume or over an `mL` scan, gives the particle-number content of each state, caches scans and plots them. |
| `root_finding.py` | The default root search for `alpha = 0`: exact root counting between the poles of `G`, then Newton relative to the nearest pole. `python root_finding.py` benchmarks it against the grid search and dense diagonalization. |
| `figures.py` | The two figures of section 3.1: the spectrum, and the four-particle content of each level. |
| `analytic_roots.py` | The same levels by diagonalizing the truncated Hamiltonian `H = H0 + V^d + \|u> lambda <u\|`, which the matrix determinant lemma says must agree with the roots. |
| `checks.py` | Analytic checks: the orbit sum rule, `G_22` against the direct momentum sum, the threshold expansions of section 3.1, the symmetry of `V^d`, the roots against diagonalization, and root counting at weak couplings. |

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
    --couplings 20 5000 1 0 --energy-range 1.9 6 \
    --cutoff 10 --cutoff-shape smooth --cutoff-width 2 --plot
four-pion-analysis figures --couplings 20 5000 1 0 --no-open
four-pion-analysis check
```

`--couplings` takes `lambda_{2<-2} lambda_{2<-4} lambda_{4<-4}` and optionally
`lambda_{3<-3}`, which drives the disconnected three-particle sub-process inside
`G_44`.

## Conventions and caveats

* The sums over levels inside `G` are regulated by a **smooth cutoff** by
  default: a level of non-interacting energy `E0` enters with a weight `w(E0)`
  that is 1 below `EMAX - width` and falls to 0 at `EMAX` through a `C^infinity`
  transition, so that

  ```
  G_NN(E, L) = sum_k w(E_k) r_k / (E - E_k) .
  ```

  A sharp cutoff makes the summand discontinuous across the cutoff surface and
  so produces power-law volume effects (appendix B.3, B.4); the smooth one
  leaves corrections that fall faster than any power of `L`.  `--cutoff-shape
  hard` restores the step `Theta(EMAX - E0)` of the notes, and
  `--cutoff-width` sets the width (default 2).
* Either way `EMAX`, and with it `width`, is a regulator: the states above the
  cutoff are smooth in `E` and are absorbed into the couplings (appendix B), so
  the couplings depend on both.  Moving between the two shapes at fixed
  couplings moves the levels by of order `1e-2 m`.
* The weights are constants in `E` and non-negative, so `G` stays a sum of
  simple poles with non-negative residues under either shape, which is what
  `root_finding.py` and the `H = D + W lambda W^T` equivalence rely on.
* Levels are identified by the multiset of squared norms, which is exact at
  every `L`.  Accidental degeneracies between different multisets, which occur
  at isolated `L`, are not resolved.
* Only particle-number-conserving disconnected sub-processes are tabulated.
  Sub-processes that change the particle number contribute to `G` only where
  levels of different particle number cross, which happens at isolated volumes.
* The roots are zeros of `det[1 - G lambda]`.  With `--root-method auto` (the
  default) `G` is a sum of simple poles with non-negative residues, so
  `root_finding.py` counts the roots between consecutive poles exactly through
  the inertia of `lambda^{-1} - G(E)` and solves each one relative to its nearest
  pole: no root is missed, and `E - pole` is accurate to ~1e-13 relative.
  `root_finding.shifts` returns these offsets directly.
* With `--root-method grid` the grid search is used instead: it brackets every
  pole of `G` within `+-delta` and discards the brackets that hold a pole, so it
  misses roots closer than `delta` to a pole (shifts of order
  `lambda_{4<-4} / (384 (mL)^9)` at weak coupling) and pairs of roots inside one
  grid cell.
* Cached scans carry the cutoff and the root method in their file name
  (`..._EMAX10_smooth2_count.npz`, `..._EMAX10_hard_grid.npz`); scans saved
  under an earlier naming are not reused.
