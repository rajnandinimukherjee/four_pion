from __future__ import annotations

import pdb
import pickle
import re
import sys
from functools import lru_cache
from itertools import combinations, permutations, product
from multiprocessing import Pool, cpu_count
from typing import Any, Iterator, Mapping, Sequence

import numpy as np
from numpy.typing import NDArray
from tqdm import tqdm

Array = NDArray[Any]


MOMENTUM_LABELS: dict[tuple[int, ...], str] = {
    (4,): "aaaa",
    (3, 1): "aaab",
    (2, 2): "aabb",
    (2, 1, 1): "aabc",
    (1, 1, 1, 1): "abcd",
}
C_KK_WEIGHTS: dict[str, int] = {
    "abcd": 1,
    "aabc": 2,
    "aaab": 6,
    "aaaa": 24,
    "aabb": 4,
}


@lru_cache(maxsize=256)
def _integer_vectors_with_norm_cached(n: int) -> tuple[tuple[int, int, int], ...]:
    """Cache immutable vector tuples to avoid rebuilding the same norm shell."""
    if n < 0:
        return ()

    radius = int(np.sqrt(n)) + 1
    grid = np.array(list(product(range(-radius, radius + 1), repeat=3)), dtype=int)
    norms = np.einsum("ij,ij->i", grid, grid)
    return tuple(map(tuple, grid[norms == n].tolist()))


def integer_vectors_with_norm(n: int) -> Array:
    """
    Return an (N,3) NumPy array of integer vectors v with v·v = n
    """
    vectors = _integer_vectors_with_norm_cached(n)
    return np.asarray(vectors, dtype=int).reshape(-1, 3)


def get_momentum_combos_unordered(N_sqs_list: Sequence[int]) -> Array:
    solutions = []

    # unique permutations of the norms
    unique_orders = set(permutations(N_sqs_list))

    for order in unique_orders:
        vecs = [integer_vectors_with_norm(n) for n in order]
        v4_set = {tuple(v) for v in vecs[3]}

        for v1 in vecs[0]:
            for v2 in vecs[1]:
                v3s = vecs[2]
                v4s = -(v1 + v2 + v3s)

                for v3, v4 in zip(v3s, v4s):
                    if tuple(v4) in v4_set:
                        solutions.append(np.array([v1, v2, v3, v4]))

    return np.asarray(solutions)


def label_momentum_sets(arr: Array) -> Array:
    N = arr.shape[0]

    # View each 3-vector as a single object
    arrv = arr.view([("", arr.dtype)] * arr.shape[2]).reshape(N, 4)

    counts_list = []
    for row in arrv:
        _, c = np.unique(row, return_counts=True)
        counts_list.append(tuple(sorted(c, reverse=True)))

    labels = np.array(
        [MOMENTUM_LABELS.get(counts, "unknown") for counts in counts_list],
        dtype=object,
    )

    return labels


def label_and_distinct_momenta(X: Array) -> tuple[Array, list[Array]]:
    """
    X: array of shape (N,4,3) (N samples, 4 vectors each, 3 components)

    Returns:
    labels: array of shape (N,), the label for each row ("aaaa", "aaab", etc.)
    distinct_momenta: list of N elements, each a 2D array of shape (k,3) of unique vectors
    """
    X = np.atleast_3d(X)
    N = X.shape[0]

    # View each 3-vector as a single object for np.unique
    Xv = X.view([("", X.dtype)] * X.shape[2]).reshape(N, 4)

    labels = np.empty(N, dtype=object)
    distinct_momenta = []

    for index, row in enumerate(Xv):
        # Find unique vectors and counts
        uniq, counts = np.unique(row, return_counts=True)
        counts_tuple = tuple(sorted(counts, reverse=True))

        # Label
        labels[index] = MOMENTUM_LABELS.get(counts_tuple, "unknown")

        # Convert unique structured array back to normal 3-vector array
        uniq_vectors = np.vstack([np.array(u.tolist()[0]) for u in uniq])
        distinct_momenta.append(uniq_vectors)

    return labels, distinct_momenta


def load_data(N: int, mL: float = 4) -> tuple[Array, Array]:
    tpiL = 2 * np.pi / mL
    quads_data = np.load(f"data/N_{N}_quads.npy", allow_pickle=True)
    energies = np.array(
        [
            [
                np.sum((q[0] * tpiL**2 + 1) ** 0.5, axis=-1),
                "_".join(map(str, list(q[0]))),
            ]
            for q in quads_data
        ],
        dtype=object,
    )
    sort = energies[energies[:, 0].argsort()]

    energies = sort[:, 0]
    levels = sort[:, 1]
    return energies, levels


# Preserve the original public constant name used by external notebooks.
C_kk_weights: dict[str, int] = C_KK_WEIGHTS


def _cubic_symmetry_group() -> Any:
    """Load NetKet only for calculations that need its cubic group actions."""
    import netket

    return netket.symmetry.group.cubic.Oh()


def compute_level(level_tuple: tuple[int, ...]) -> int:
    momenta = get_momentum_combos_unordered(level_tuple)
    labels = label_momentum_sets(momenta)
    return sum(C_KK_WEIGHTS.get(label, 0) for label in labels)


def C_kk(N: int, save: bool = False, chunkstep: int = 10) -> Array:
    energies, levels = load_data(N, mL=4)
    print(f"N={N}: {len(levels)} levels loaded from file")

    parsed_levels = [tuple(map(int, lvl.split("_"))) for lvl in levels]
    n_proc = cpu_count()
    print(f"using {n_proc} CPU cores")

    with Pool(processes=n_proc) as pool:
        c_kk_list = list(
            tqdm(
                pool.imap(compute_level, parsed_levels, chunksize=n_proc * chunkstep),
                total=len(parsed_levels),
            )
        )

    c_kk = np.array(c_kk_list, dtype=int)
    if save:
        filename = f"data/N_{N}_quads_coeffs.npy"
        np.save(filename, np.vstack((levels, c_kk)))
        print(f"saved coefficients to {filename}")
    return c_kk


def load_C_kk(
    N: int, verbose: bool = True, max_lvl: int | None = None
) -> tuple[Array, Array]:
    filename = f"data/N_{N}_quads_coeffs.npy"
    data = np.load(filename, allow_pickle=True)[:, :max_lvl]
    if verbose:
        print(f"N={N}: loaded {data.shape[1]} levels from {filename}")
    return data[0, :], data[1, :]


def omega(k: Array, mL: float) -> Any:
    tpiL = 2 * np.pi / mL
    return (np.sum(k * k, axis=-1) * tpiL**2 + 1) ** 0.5


def split_momenta(ks: Array) -> tuple[Array, Array]:
    idx = np.triu_indices(len(ks), 1)
    pairs = np.stack((ks[idx[0]], ks[idx[1]]), axis=1)
    order = np.lexsort(
        (
            pairs[:, :, 2],  # z
            pairs[:, :, 1],  # y
            pairs[:, :, 0],  # x
        )
    )

    sorted_pairs = np.take_along_axis(pairs, order[:, :, None], axis=1)
    flat = sorted_pairs.reshape(sorted_pairs.shape[0], -1)
    unique_flat, multiplicities = np.unique(flat, axis=0, return_counts=True)
    unique_pairs = unique_flat.reshape(-1, 2, 3)
    return unique_pairs, multiplicities


def mom_str(k: Array) -> str:
    return "[" + "".join([str(i) for i in k]) + "]"


def parse_mom_str(s: str) -> Array:
    content = s.strip()[1:-1]  # remove brackets
    matches = re.findall(r"-?\d", content)

    if len(matches) != 3:
        raise ValueError(f"Expected 3 components in {s}")

    return np.array([int(x) for x in matches], dtype=int)


def get_kfree(kins: Array, ki: Array, kj: Array) -> Array:
    k_strs = [mom_str(k) for k in kins]
    k_strs.remove(mom_str(ki))
    k_strs.remove(mom_str(kj))
    assert len(k_strs) == 2, pdb.set_trace()
    return np.array([parse_mom_str(k_strs[0]), parse_mom_str(k_strs[1])], dtype=int)


def _pair_contributions(
    kins: Array,
    G: Any,
    mL: float,
    pair_data: tuple[Array, Array] | None = None,
) -> Iterator[tuple[Array, Array, int, Array, Array, Array, Array, Any, Any]]:
    """Yield kinematic data shared by the two contraction calculations."""
    if pair_data is None:
        pair_data = split_momenta(kins)
    unique_pairs, multiplicities = pair_data
    omegas = {mom_str(k): omega(k, mL) for k in kins}
    group_actions = {mom_str(k): [g(k).astype(int) for g in G] for k in kins}

    for pair, multiplicity in zip(unique_pairs, multiplicities):
        ki, kj = pair[0], pair[1]
        kfree = get_kfree(kins, ki, kj)

        wi, wj = omegas[mom_str(ki)], omegas[mom_str(kj)]
        omega_in = wi + wj
        omega_product = 1 / (4 * wi * wj)

        k1s = np.unique(
            np.vstack((group_actions[mom_str(ki)], group_actions[mom_str(kj)])),
            axis=0,
        )
        k2s = -ki[None, :] - kj[None, :] - k1s
        omega_out = omega(k1s, mL) + omega(k2s, mL)

        conserving_indices = np.where(np.isclose(omega_in, omega_out))
        k1, k2 = k1s[conserving_indices], k2s[conserving_indices]
        outgoing_momenta = np.empty((len(k1), 4, 3), dtype=int)
        outgoing_momenta[:, 0], outgoing_momenta[:, 1] = k1, k2
        outgoing_momenta[:, 2], outgoing_momenta[:, 3] = -kfree[0], -kfree[1]

        if np.all(kfree[0] == kfree[1]):
            q_ij = np.sum(np.all(outgoing_momenta == -kfree[0], axis=-1), axis=-1)
            counts = q_ij * (q_ij - 1)
        else:
            matches = outgoing_momenta[:, None, :, :] == -kfree[None, :, None, :]
            matches = np.all(matches, axis=-1)
            counts = np.prod(np.sum(matches, axis=-1), axis=1)
        q_ij = np.sum(counts)

        yield (
            ki,
            kj,
            multiplicity,
            kfree,
            k1,
            k2,
            counts,
            q_ij,
            omega_product,
        )


def compute_Q_tilde(
    level_tuple: tuple[int, ...],
    c_kk: float,
    mL: float,
    G: Any,
    verbose: bool = False,
) -> float:
    momenta = get_momentum_combos_unordered(level_tuple)
    q_tilde_k = 0
    if verbose:
        print(
            f"\nlevel n_sqs: j={''.join([str(i) for i in level_tuple])}, "
            f"{len(momenta)} momentum combos"
        )

    for idx, kins in enumerate(momenta):
        pair_data = split_momenta(kins) if verbose else None
        if verbose:
            if idx > 0:
                pdb.set_trace()
            print(
                f"[{idx + 1}/{len(momenta)}] kins = "
                f"{', '.join([mom_str(k) for k in kins])}"
            )
            print(f"{len(pair_data[0])} unique contractions")

        for (
            ki,
            kj,
            multiplicity,
            kfree,
            k1,
            k2,
            counts,
            q_ij,
            omega_product,
        ) in _pair_contributions(kins, G, mL, pair_data):
            q_tilde_k += multiplicity * omega_product * q_ij

            if verbose:
                k1_str = ", ".join([f"{mom_str(k):<8}" for k in k1])
                k2_str = ", ".join([f"{mom_str(k):<8}" for k in k2])
                q_ij_str = ", ".join([f"{str(count):<8}" for count in counts])
                print(f"  {mom_str(ki):<8}---\\ /--- k1     {k1_str}")
                print(f"{str(multiplicity):<1}             x{' ' * 12}{q_ij_str}")
                print(f"  {mom_str(kj):<8}---/ \\--- k2     {k2_str}")
                print(f"  {mom_str(kfree[0]):<8}---------{mom_str(-kfree[0]):>8}")
                print(f"  {mom_str(kfree[1]):<8}---------{mom_str(-kfree[1]):>8}")
                print("\n")

    return q_tilde_k / c_kk


def nsq_combos(
    level_tuple: tuple[int, ...], G: Any, mL: float = 1, verbose: bool = False
) -> dict[tuple[int, int], int]:
    momenta = get_momentum_combos_unordered(level_tuple)
    norm_combinations = {pair: 0 for pair in set(combinations(level_tuple, 2))}

    if verbose:
        print(
            f"\nlevel n_sqs: j={''.join([str(i) for i in level_tuple])}, "
            f"{len(momenta)} momentum combos"
        )

    for kins in momenta:
        for ki, kj, multiplicity, _, _, _, _, q_ij, _ in _pair_contributions(
            kins, G, mL
        ):
            ni, nj = int(np.dot(ki, ki)), int(np.dot(kj, kj))
            norm_pair = (ni, nj) if ni < nj else (nj, ni)
            norm_combinations[norm_pair] += int(q_ij * multiplicity)

    if verbose:
        print(norm_combinations)
    return norm_combinations


def Q_tilde(
    mL: float,
    N: int = 3,
    max_lvl: int | None = None,
    verbose: bool = True,
    MPI: bool = True,
) -> tuple[Array, Array]:
    levels, C_kks = load_C_kk(N, verbose=verbose, max_lvl=max_lvl)
    max_lvl = max_lvl or len(levels)
    G = _cubic_symmetry_group()

    parsed_levels = [tuple(map(int, lvl.split("_"))) for lvl in levels[:max_lvl]]
    args = [(parsed_levels[lvl], C_kks[lvl], mL, G) for lvl in range(max_lvl)]

    if MPI:
        n_proc = cpu_count()
        with Pool(processes=n_proc) as p:
            q_tilde = p.starmap(compute_Q_tilde, args)
    else:
        q_tilde = np.zeros(max_lvl, dtype=float)
        for lvl, arg in enumerate(args):
            q_tilde[lvl] = compute_Q_tilde(*arg)

    return levels, np.array(q_tilde, dtype=float)


def store_nsq_combos(
    max_lvl: int = 50, save: bool = True
) -> list[dict[tuple[int, int], int]]:
    levels, C_kks = load_C_kk(N=3, verbose=True, max_lvl=max_lvl)
    max_lvl = max_lvl or len(levels)
    G = _cubic_symmetry_group()

    parsed_levels = [tuple(map(int, lvl.split("_"))) for lvl in levels[:max_lvl]]

    nsq_data = []
    for lvl in tqdm(range(max_lvl)):
        nsq_data.append(nsq_combos(parsed_levels[lvl], G, verbose=False))

    if save:
        filename = f"data/g22L3_coeffs_{max_lvl}.pkl"
        with open(filename, "wb") as data_file:
            pickle.dump(nsq_data, data_file)
        print(f"saved data to {filename}")

    return nsq_data


def convert_nsq_dict(
    data: Mapping[tuple[int, int], int], mL: float, c_kk: float
) -> float:
    q_tilde = 0
    for pair, count in data.items():
        ki, kj = pair
        prod_om = 1 / (4 * omega(ki**0.5, mL) * omega(kj**0.5, mL))
        q_tilde += count * prod_om / c_kk
    return q_tilde


def convert_nsq_data(mL: float, max_lvl: int = 50) -> Array:
    filename = f"data/g22L3_coeffs_{max_lvl}.pkl"
    levels, C_kks = load_C_kk(N=3, max_lvl=max_lvl, verbose=False)
    with open(filename, "rb") as data_file:
        data = pickle.load(data_file)[:max_lvl]
    Q_tildes = np.array(
        [convert_nsq_dict(data[lvl], mL, C_kks[lvl]) for lvl in range(max_lvl)]
    )
    return Q_tildes


if __name__ == "__main__":
    N, mL = 6, 4
    j = int(sys.argv[1]) if len(sys.argv) > 1 else 0
    # levels, C_kks = load_C_kk(N, verbose=True)
    # q_tilde = compute_Q_tilde(tuple(map(int, levels[j].split("_"))), C_kks[j],
    #                          mL, netket.symmetry.group.cubic.Oh(), verbose=True)
    # levels, Q_tildes1 = Q_tilde(mL, max_lvl=50)
    # nsqs = nsq_combos(tuple(map(int, levels[j].split("_"))),
    #                  netket.symmetry.group.cubic.Oh(), verbose=True)
    # print(f"q_tilde({j},mL={mL})={convert_nsq_dict(nsqs, mL, C_kks[j])}")
    # data = store_nsq_combos(max_lvl=100)
    # Q_tildes2 = convert_nsq_data(mL)
