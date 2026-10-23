from __future__ import annotations

import itertools
from collections import Counter, defaultdict
from typing import Any

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

Array = NDArray[Any]


def back_to_back_pairs(N: int) -> Array:
    """
    compute all back-to-back momenta pairs allowed in a grid size up to N
    and save the value of n · n and corresponding multiplicities
    """
    # Generate all integer 3-vectors in cube [-N, N]^3
    grid = np.array(list(itertools.product(range(-N, N + 1), repeat=3)), dtype=np.int64)
    M = len(grid)
    print(f"[info] grid has {M} points")

    # Precompute squared norms
    norms = np.abs(np.einsum("ij,ij->i", grid, grid))  # shape (M,)
    norm_list, multiplicities = np.unique(norms, return_counts=True)
    result = np.array(
        [
            (np.array([norm_list[i], norm_list[i]]), int(multiplicities[i]))
            for i in range(len(norm_list))
        ],
        dtype=object,
    )
    return result


def momentum_norm_quadruples(N: int) -> Array:
    """
    Compute all unique unordered quadruples of squared norms [n1,n2,n3,n4],
    with multiplicities, for p_i in [-N,N]^3 satisfying p1+p2+p3+p4 = 0.
    """

    # --- generate momentum grid ---
    grid = np.array(list(itertools.product(range(-N, N + 1), repeat=3)), dtype=np.int16)
    M = len(grid)
    print(f"[info] grid has {M} points")

    # squared norms
    norms = np.einsum("ij,ij->i", grid, grid)

    # ------------------------------------------------------------------
    # Step 1: pair-sum dictionary with norm-pair multiplicities
    #   sum_dict[s][(n_i, n_j)] = multiplicity
    # ------------------------------------------------------------------
    sum_dict = defaultdict(Counter)

    for i in tqdm(range(M), desc="building (i,j) pairs"):
        pi = grid[i]
        ni = norms[i]
        for j in range(i, M):  # symmetry i <= j
            pj = grid[j]
            nj = norms[j]

            s = tuple((pi + pj).tolist())
            key = tuple(sorted((ni, nj)))

            weight = 1 if i == j else 2
            sum_dict[s][key] += weight

    print(f"[info] built {len(sum_dict)} unique pair-sums")

    # ------------------------------------------------------------------
    # Step 2: match against -(p3+p4)
    # ------------------------------------------------------------------
    norm_counter = Counter()

    for k in tqdm(range(M), desc="matching (k,l) pairs"):
        pk = grid[k]
        nk = norms[k]
        for l in range(k, M):  # symmetry k <= l
            pl = grid[l]
            nl = norms[l]

            s = tuple((-(pk + pl)).tolist())
            if s not in sum_dict:
                continue

            key_kl = tuple(sorted((nk, nl)))
            weight_kl = 1 if k == l else 2

            for key_ij, count_ij in sum_dict[s].items():
                combo = tuple(sorted((*key_ij, *key_kl)))
                norm_counter[combo] += count_ij * weight_kl

    print(f"[info] found {len(norm_counter)} unique squared-norm quadruples")

    # convert to array form
    result = [(np.array(k, dtype=int), v) for k, v in norm_counter.items()]

    return np.array(result, dtype=object)


def symmetryFactor(vec: Array) -> float:
    factor = 1
    for i in range(len(vec)):
        factor *= i + 1
    counts = [1 for _ in range(len(vec))]
    for i in range(len(vec)):
        for j in range(i + 1, len(vec)):
            if (vec[i] == vec[j]).all():
                counts[i] += 1
    for c in counts:
        factor /= c
    return factor


if __name__ == "__main__":
    N = 8
    # pairs = back_to_back_pairs(N)
    # filename = f"data/N_{N}_pairs.npy"
    # np.save(filename, pairs)
    quads = momentum_norm_quadruples(N)
    filename = f"data/N_{N}_quads.npy"
    np.save(filename, quads)
    print(f"{len(quads)} levels and multiplicities saved to {filename}")
