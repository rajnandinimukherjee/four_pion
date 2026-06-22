import pdb
import re
from itertools import permutations, combinations, product
from multiprocessing import Pool, cpu_count

import netket
import numpy as np
from tqdm import tqdm
import sys
import pickle


def integer_vectors_with_norm(n):
    """
    Return an (N,3) NumPy array of integer vectors v with v·v = n
    """
    if n < 0:
        return np.empty((0, 3), dtype=int)

    R = int(np.sqrt(n)) + 1
    grid = np.array(list(product(range(-R, R+1), repeat=3)), dtype=int)

    norms = np.einsum("ij,ij->i", grid, grid)
    return grid[norms == n]


def get_momentum_combos_unordered(N_sqs_list):
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


def label_momentum_sets(arr):
    N = arr.shape[0]

    # View each 3-vector as a single object
    arrv = arr.view([('', arr.dtype)] * arr.shape[2]).reshape(N, 4)

    counts_list = []
    for row in arrv:
        _, c = np.unique(row, return_counts=True)
        counts_list.append(tuple(sorted(c, reverse=True)))

    # Map tuples to labels
    label_map = {
        (4,): "aaaa",
        (3, 1): "aaab",
        (2, 2): "aabb",
        (2, 1, 1): "aabc",
        (1, 1, 1, 1): "abcd"
    }

    labels = np.array([label_map.get(c, "unknown")
                      for c in counts_list], dtype=object)

    return labels


def label_and_distinct_momenta(X):
    """
    X: array of shape (N,4,3) (N samples, 4 vectors each, 3 components)

    Returns:
    labels: array of shape (N,), the label for each row ("aaaa", "aaab", etc.)
    distinct_momenta: list of N elements, each a 2D array of shape (k,3) of unique vectors
    """
    X = np.atleast_3d(X)
    N = X.shape[0]

    # View each 3-vector as a single object for np.unique
    Xv = X.view([('', X.dtype)] * X.shape[2]).reshape(N, 4)

    labels = np.empty(N, dtype=object)
    distinct_momenta = []

    label_map = {
        (4,): "aaaa",
        (3, 1): "aaab",
        (2, 2): "aabb",
        (2, 1, 1): "aabc",
        (1, 1, 1, 1): "abcd"
    }

    for row in Xv:
        # Find unique vectors and counts
        uniq, counts = np.unique(row, return_counts=True)
        counts_tuple = tuple(sorted(counts, reverse=True))

        # Label
        labels[distinct_momenta.__len__()] = label_map.get(
            counts_tuple, "unknown")

        # Convert unique structured array back to normal 3-vector array
        uniq_vectors = np.vstack([np.array(u.tolist()[0]) for u in uniq])
        distinct_momenta.append(uniq_vectors)

    return labels, distinct_momenta


def load_data(N, mL=4):
    tpiL = 2*np.pi/mL
    quads_data = np.load(f"data/N_{N}_quads.npy", allow_pickle=True)
    energies = np.array(
        [[np.sum((q[0]*tpiL**2+1)**0.5, axis=-1),
          '_'.join(map(str, list(q[0])))] for q in quads_data],
        dtype=object)
    sort = energies[energies[:, 0].argsort()]

    energies = sort[:, 0]
    levels = sort[:, 1]
    return energies, levels


C_kk_weights = {
    "abcd": 1,
    "aabc": 2,
    "aaab": 6,
    "aaaa": 24,
    "aabb": 4
}


def compute_level(level_tuple):
    momenta = get_momentum_combos_unordered(level_tuple)
    labels = label_momentum_sets(momenta)
    return sum(C_kk_weights.get(label, 0) for label in labels)


def C_kk(N, save=False, chunkstep=10):
    energies, levels = load_data(N, mL=4)
    print(f"N={N}: {len(levels)} levels loaded from file")

    parsed_levels = [tuple(map(int, lvl.split("_"))) for lvl in levels]
    n_proc = cpu_count()
    print(f"using {n_proc} CPU cores")

    with Pool(processes=n_proc) as pool:
        c_kk_list = list(
            tqdm(
                pool.imap(compute_level, parsed_levels,
                          chunksize=n_proc*chunkstep),
                total=len(parsed_levels)
            )
        )

    c_kk = np.array(c_kk_list, dtype=int)
    if save:
        filename = f"data/N_{N}_quads_coeffs.npy"
        np.save(filename, np.vstack((levels, c_kk)))
        print(f"saved coefficients to {filename}")
    return c_kk


def load_C_kk(N, verbose=True, max_lvl=None):
    filename = f"data/N_{N}_quads_coeffs.npy"
    data = np.load(filename, allow_pickle=True)[:, :max_lvl]
    if verbose:
        print(f"N={N}: loaded {data.shape[1]} levels from {filename}")
    return data[0, :], data[1, :]


def omega(k, mL):
    tpiL = 2*np.pi/mL
    return (np.sum(k*k, axis=-1)*tpiL**2+1)**0.5


def split_momenta(ks):
    idx = np.triu_indices(len(ks), 1)
    pairs = np.stack((ks[idx[0]], ks[idx[1]]), axis=1)
    order = np.lexsort((
        pairs[:, :, 2],  # z
        pairs[:, :, 1],  # y
        pairs[:, :, 0],  # x
    ))

    sorted_pairs = np.take_along_axis(
        pairs,
        order[:, :, None],
        axis=1
    )
    flat = sorted_pairs.reshape(sorted_pairs.shape[0], -1)
    unique_flat, multiplicities = np.unique(
        flat, axis=0, return_counts=True)
    unique_pairs = unique_flat.reshape(-1, 2, 3)
    return unique_pairs, multiplicities


def mom_str(k):
    return '['+''.join([str(i) for i in k])+']'


def parse_mom_str(s: str) -> np.ndarray:
    content = s.strip()[1:-1]  # remove brackets
    matches = re.findall(r'-?\d', content)

    if len(matches) != 3:
        raise ValueError(f"Expected 3 components in {s}")

    return np.array([int(x) for x in matches], dtype=int)


def get_kfree(kins, ki, kj):
    k_strs = [mom_str(k) for k in kins]
    k_strs.remove(mom_str(ki))
    k_strs.remove(mom_str(kj))
    assert len(k_strs) == 2, pdb.set_trace()
    return np.array([parse_mom_str(k_strs[0]), parse_mom_str(k_strs[1])], dtype=int)


def compute_Q_tilde(level_tuple, c_kk, mL, G, verbose=False):
    momenta = get_momentum_combos_unordered(level_tuple)
    Q_tilde_k = 0
    if verbose:
        print(f"\nlevel n_sqs: j={''.join([str(i) for i in level_tuple])}, {
              len(momenta)} momentum combos")
    for idx, kins in enumerate(momenta):
        unique_pairs, multiplicities = split_momenta(kins)
        omegas = {mom_str(k): omega(k, mL) for k in kins}
        G_actions = {mom_str(k): [g(k).astype(int) for g in G]
                     for k in kins}
        if verbose:
            if idx > 0:
                pdb.set_trace()
            print(f"[{idx+1}/{len(momenta)}] kins = {', '.join([mom_str(k)
                  for k in kins])}")
            print(f"{len(unique_pairs)} unique contractions")
        for pair, mult in zip(unique_pairs, multiplicities):
            ki, kj = pair[0], pair[1]
            kfree = get_kfree(kins, ki, kj)

            wi, wj = omegas[mom_str(ki)], omegas[mom_str(kj)]
            omin = wi+wj
            prod_oms = 1/(4*wi*wj)

            k1s = np.unique(np.vstack((G_actions[mom_str(ki)],
                                       G_actions[mom_str(kj)])), axis=0)
            k2s = -ki[None, :]-kj[None, :]-k1s
            omout = omega(k1s, mL)+omega(k2s, mL)

            cons_idx = np.where(np.isclose(omin, omout))
            k1, k2 = k1s[cons_idx], k2s[cons_idx]
            kouts = np.empty((len(k1), 4, 3), dtype=int)
            kouts[:, 0], kouts[:, 1] = k1, k2
            kouts[:, 2], kouts[:, 3] = -kfree[0], -kfree[1]

            if np.all(kfree[0] == kfree[1]):
                Q_ij = np.sum(np.all(kouts == -kfree[0], axis=-1), axis=-1)
                counts = Q_ij*(Q_ij-1)
            else:
                matches = (kouts[:, None, :, :] == -kfree[None, :, None, :])
                matches = np.all(matches, axis=-1)
                counts = np.prod(np.sum(matches, axis=-1), axis=1)
            Q_ij = np.sum(counts)
            Q_tilde_k += mult*prod_oms*Q_ij

            if verbose:
                k1_str = ', '.join([f"{mom_str(k):<8}" for k in k1])
                k2_str = ', '.join([f"{mom_str(k):<8}" for k in k2])
                Q_ij_str = ', '.join([f"{str(cnt):<8}" for cnt in counts])
                print(f"  {mom_str(ki):<8}---\\ /--- k1     {k1_str}")
                print(f"{str(mult):<1}             x{' '*12}{Q_ij_str}")
                print(f"  {mom_str(kj):<8}---/ \\--- k2     {k2_str}")
                print(f"  {mom_str(kfree[0]):<8}---------{mom_str(-kfree[0]):>8}")
                print(f"  {mom_str(kfree[1]):<8}---------{mom_str(-kfree[1]):>8}")
                print("\n")
    return Q_tilde_k/c_kk


def nsq_combos(level_tuple, G, mL=1, verbose=False):
    momenta = get_momentum_combos_unordered(level_tuple)
    nsq_combos = {pair: 0 for pair in set(combinations(level_tuple, 2))}

    if verbose:
        print(f"\nlevel n_sqs: j={''.join([str(i) for i in level_tuple])}, {
              len(momenta)} momentum combos")
    for idx, kins in enumerate(momenta):
        unique_pairs, multiplicities = split_momenta(kins)
        omegas = {mom_str(k): omega(k, mL) for k in kins}
        G_actions = {mom_str(k): [g(k).astype(int) for g in G]
                     for k in kins}
        for pair, mult in zip(unique_pairs, multiplicities):
            ki, kj = pair[0], pair[1]
            kfree = get_kfree(kins, ki, kj)

            wi, wj = omegas[mom_str(ki)], omegas[mom_str(kj)]
            omin = wi+wj

            k1s = np.unique(np.vstack((G_actions[mom_str(ki)],
                                       G_actions[mom_str(kj)])), axis=0)
            k2s = -ki[None, :]-kj[None, :]-k1s
            omout = omega(k1s, mL)+omega(k2s, mL)

            cons_idx = np.where(np.isclose(omin, omout))
            k1, k2 = k1s[cons_idx], k2s[cons_idx]
            kouts = np.empty((len(k1), 4, 3), dtype=int)
            kouts[:, 0], kouts[:, 1] = k1, k2
            kouts[:, 2], kouts[:, 3] = -kfree[0], -kfree[1]

            if np.all(kfree[0] == kfree[1]):
                Q_ij = np.sum(np.all(kouts == -kfree[0], axis=-1), axis=-1)
                counts = Q_ij*(Q_ij-1)
            else:
                matches = (kouts[:, None, :, :] == -kfree[None, :, None, :])
                matches = np.all(matches, axis=-1)
                counts = np.prod(np.sum(matches, axis=-1), axis=1)
            Q_ij = np.sum(counts)

            ni, nj = int(np.dot(ki, ki)), int(np.dot(kj, kj))
            nsq = (ni, nj) if ni < nj else (nj, ni)

            nsq_combos[nsq] += int(Q_ij*mult)

    if verbose:
        print(nsq_combos)
    return nsq_combos


def Q_tilde(mL, N=3, max_lvl=None, verbose=True, MPI=True):
    levels, C_kks = load_C_kk(N, verbose=verbose, max_lvl=max_lvl)
    max_lvl = max_lvl or len(levels)
    G = netket.symmetry.group.cubic.Oh()

    parsed_levels = [tuple(map(int, lvl.split("_")))
                     for lvl in levels[:max_lvl]]
    args = [(parsed_levels[lvl], C_kks[lvl], mL, G)
            for lvl in range(max_lvl)]

    if MPI:
        n_proc = cpu_count()
        with Pool(processes=n_proc) as p:
            q_tilde = p.starmap(compute_Q_tilde, args)
    else:
        q_tilde = np.zeros(max_lvl, dtype=float)
        for lvl, arg in enumerate(args):
            q_tilde[lvl] = compute_Q_tilde(*arg)

    return levels, np.array(q_tilde, dtype=float)


def store_nsq_combos(max_lvl=50, save=True):
    levels, C_kks = load_C_kk(N=3, verbose=True, max_lvl=max_lvl)
    max_lvl = max_lvl or len(levels)
    G = netket.symmetry.group.cubic.Oh()

    parsed_levels = [tuple(map(int, lvl.split("_")))
                     for lvl in levels[:max_lvl]]

    nsq_data = []
    for lvl in tqdm(range(max_lvl)):
        nsq_data.append(nsq_combos(parsed_levels[lvl], G, verbose=False))

    if save:
        filename = f"data/g22L3_coeffs_{max_lvl}.pkl"
        pickle.dump(nsq_data, open(filename, "wb"))
        print(f"saved data to {filename}")

    return nsq_data


def convert_nsq_dict(data, mL, c_kk):
    q_tilde = 0
    for pair, count in data.items():
        ki, kj = pair
        prod_om = 1/(4*omega(ki**0.5, mL)*omega(kj**0.5, mL))
        q_tilde += count*prod_om/c_kk
    return q_tilde


def convert_nsq_data(mL, max_lvl=50):
    filename = f"data/g22L3_coeffs_{max_lvl}.pkl"
    levels, C_kks = load_C_kk(N=3, max_lvl=max_lvl, verbose=False)
    data = pickle.load(open(filename, "rb"))[:max_lvl]
    Q_tildes = np.array([convert_nsq_dict(data[lvl], mL, C_kks[lvl])
                         for lvl in range(max_lvl)])
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
