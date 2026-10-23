"""Integer momentum-shell enumeration for fixed-particle-number states."""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np


def generate_j_vectors(
    N: int, E_max: float, L: float, m: float
) -> Iterator[tuple[int, ...]]:
    """Yield ordered shell-index tuples whose total energy is below ``E_max``."""
    k = 2 * np.pi / L

    j_max = 0
    while np.sqrt(j_max * k**2 + m**2) * N <= E_max:
        j_max += 1

    j_values = np.arange(j_max + 1)
    omegas = np.sqrt(j_values * k**2 + m**2)

    def backtrack(
        start_j: int, depth: int, current_energy: float, current: list[int]
    ) -> Iterator[tuple[int, ...]]:
        if depth == N:
            if current_energy <= E_max:
                yield tuple(current)
            return

        remaining = N - depth
        for j in range(start_j, j_max + 1):
            omega = omegas[j]
            new_energy = current_energy + omega

            # Later shell energies are no smaller than this one.
            if new_energy + (remaining - 1) * omega > E_max:
                break

            current.append(j)
            yield from backtrack(j, depth + 1, new_energy, current)
            current.pop()

    return backtrack(0, 0, 0.0, [])


def r3(n: int) -> int:
    """Count integer triples with squared norm ``n`` using Legendre's theorem."""
    if n == 0:
        return 1

    reduced_n = n
    while reduced_n % 4 == 0:
        reduced_n //= 4
    if reduced_n % 8 == 7:
        return 0

    divisor_sum = 0
    divisor = 1
    while divisor * divisor <= n:
        if n % divisor == 0:
            if divisor % 4 != 0:
                divisor_sum += divisor

            if divisor * divisor != n:
                paired_divisor = n // divisor
                if paired_divisor % 4 != 0:
                    divisor_sum += paired_divisor
        divisor += 1

    return 12 * divisor_sum


def mult(j_list: tuple[int, ...] | list[int]) -> int:
    """Return the product of shell degeneracies for a tuple of norms."""
    product = 1
    for j in j_list:
        product *= r3(j)
    return product


if __name__ == "__main__":
    N, Emax, L, m = 4, 15, 20, 1
    configs = generate_j_vectors(N, Emax, L, m)
    count = sum(1 for _ in configs)
    print(f"N={N}, Emax={Emax}, L={L}, N_shells={count}")
    if count < 1e6:
        cfgs = np.array([config for config in configs])
