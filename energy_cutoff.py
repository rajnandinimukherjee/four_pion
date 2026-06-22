import numpy as np


def generate_j_vectors(N, E_max, L, m):
    k = 2 * np.pi / L

    # --- determine j_max ---
    j_max = 0
    while np.sqrt(j_max * k**2 + m**2) * N <= E_max:
        j_max += 1

    # --- vectorized omega computation ---
    j_vals = np.arange(j_max + 1)
    omegas = np.sqrt(j_vals * k**2 + m**2)

    # --- recursive generator ---
    def backtrack(start_j, depth, current_energy, current):
        if depth == N:
            if current_energy <= E_max:
                yield tuple(current)
            return

        remaining = N - depth

        for j in range(start_j, j_max + 1):
            w = omegas[j]
            new_energy = current_energy + w

            # vectorized-style pruning (cheap scalar check)
            if new_energy + (remaining - 1) * w > E_max:
                break

            current.append(j)
            yield from backtrack(j, depth + 1, new_energy, current)
            current.pop()

    return backtrack(0, 0, 0.0, [])


def r3(n):
    # Legendre's theorem: no solutions if n = 4^a (8b+7)
    if n == 0:
        return 1
    m = n
    while m % 4 == 0:
        m //= 4
    if m % 8 == 7:
        return 0

    # sum divisors d of n with 4 ∤ d
    s = 0
    d = 1
    while d * d <= n:
        if n % d == 0:
            if d % 4 != 0:
                s += d
            if d * d != n:
                d2 = n // d
                if d2 % 4 != 0:
                    s += d2
        d += 1

    return 12 * s


def mult(j_list):
    prod = 1
    for j in j_list:
        prod *= r3(j)
    return prod


if __name__ == "__main__":
    N, Emax, L, m = 4, 15, 20, 1
    configs = generate_j_vectors(N, Emax, L, m)
    count = sum(1 for _ in configs)
    print(f"N={N}, Emax={Emax}, L={L}, N_shells={count}")
    if count < 1e+6:
        cfgs = np.array([j for j in configs])
