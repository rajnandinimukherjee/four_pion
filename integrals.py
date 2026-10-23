"""Finite-volume two-pion and four-pion loop integrals."""

from __future__ import annotations

import pdb
from typing import Any, Sequence

import matplotlib.pyplot as plt
import mpi4py.MPI as MPI_
import numpy as np
from numpy.typing import ArrayLike, NDArray

import utils
from S4_utils import convert_nsq_data
from energy_cutoff import mult

Array = NDArray[Any]


class TwoPionIntegral:
    """Divergent part of the loop integral with one independent momentum."""

    def __init__(self, mL: float, EMAX: float = 20, alpha: float = 1.0) -> None:
        self.mL: float = mL
        self.EMAX: float = EMAX
        self.alpha: float = alpha
        self.tpiL: float = 2.0 * np.pi / mL
        self.prefactor: float = 1.0 / (2 * (mL**3))

        self.levels: Array
        self.energies: Array
        self.coeffs: Array
        self.levels, self.energies, self.coeffs = self.get_energy_levels()

    def get_energy_levels(self) -> tuple[Array, Array, Array]:
        levels = np.load("data/N_100_ordered_pairs.npy", allow_pickle=True)
        energies = np.sum((levels * self.tpiL**2 + 1) ** 0.5, axis=-1)
        energies = energies[energies <= self.EMAX]

        levels = levels[: len(energies), :].astype(int)
        prod_omegas = np.prod(2 * (levels * self.tpiL**2 + 1) ** 0.5, axis=-1)
        coeffs = np.array(self.prefactor / prod_omegas, dtype=np.float32)
        return levels, energies, coeffs

    def __call__(self, E: ArrayLike) -> Any:
        E = np.atleast_1d(E)
        energy_differences = E[:, None] - self.energies[None, :]
        cutoff = np.exp(self.alpha * energy_differences)
        result = np.sum(cutoff * self.coeffs / energy_differences, axis=-1)
        return result[0] if np.size(E) == 1 else result

    def remainders(self, max_lvl: int | None = None) -> Array:
        n_levels = max_lvl or len(np.where(self.energies < 4)[0])
        mask = ~np.eye(n_levels, dtype=bool)

        energy_differences = (
            self.energies[:n_levels, None] - self.energies[None, :n_levels]
        )
        np.fill_diagonal(energy_differences, 1.0)
        coeffs_matrix = mask * np.broadcast_to(
            self.coeffs[:n_levels], (n_levels, n_levels)
        )

        cutoff = mask * np.exp(self.alpha * energy_differences)
        add = self.alpha * self.coeffs[:n_levels]
        return add + np.sum(cutoff * coeffs_matrix / energy_differences, axis=-1)

    def plot(
        self,
        Erange: ArrayLike,
        yrange: Sequence[float] | None = None,
        filename: str = "",
    ) -> None:
        integral_values = self(Erange)
        _, ax = plt.subplots(1, 1)
        ax.plot(Erange, integral_values.real)
        ax.set_ylabel(r"$Im\mathcal{I}_2(E, \mathbf{P}=\mathbf{0})$")
        ax.set_xlabel("E")
        if yrange is not None:
            ax.set_ylim(yrange)

        if filename == "":
            filename = f"plots/I2_mL{self.mL}_EMAX{self.EMAX}_alpha{self.alpha}.pdf"
        utils.call_PDF(filename)


class FourPionIntegral:
    """Divergent part of the loop integral with three independent momenta."""

    def __init__(
        self,
        mL: float,
        g22: float = 0,
        EMAX: float = 20,
        alpha: float = 1.0,
        **kwargs: Any,
    ) -> None:
        self.mL: float = mL
        self.EMAX: float = EMAX
        self.alpha: float = alpha
        self.tpiL: float = 2.0 * np.pi / mL
        self.prefactor: float = 1.0 / (24 * (mL**9))

        self.levels: Array
        self.energies: Array
        self.coeffs: Array
        self.levels, self.energies, self.coeffs = self.get_energy_levels()
        if g22 != 0:
            self.shift_levels(g22)

    def get_energy_levels(self, **kwargs: Any) -> tuple[Array, Array, Array]:
        levels = np.load("data/N_8_ordered_quads.npy", allow_pickle=True)
        energies = np.sum((levels * self.tpiL**2 + 1) ** 0.5, axis=-1)
        energies = energies[energies <= self.EMAX]

        levels = levels[: len(energies), :].astype(int)
        prod_omegas = np.prod(2 * (levels * self.tpiL**2 + 1) ** 0.5, axis=-1)
        c_kks = np.array([mult(levels[i, :]) for i in range(levels.shape[0])])
        coeffs = np.array(self.prefactor * c_kks / prod_omegas, dtype=np.float32)
        return levels, energies, coeffs

    def shift_levels(self, g22: float, max_lvl: int = 50) -> None:
        self.g22: float = g22
        self.shifts: Array = self.load_shifts(max_lvl=max_lvl)
        self.unshifted_energies: Array = self.energies.copy()

        g22L3 = g22 / (self.mL**3)
        n_levels = len(self.shifts)
        self.energies[:n_levels] = (
            self.unshifted_energies[:n_levels] + g22L3 * self.shifts
        )

    def __call__(
        self,
        E: ArrayLike,
        MPI: bool = True,
        block_size: int = 1000,
        debug: bool = False,
    ) -> Any:
        E = np.atleast_1d(E)
        if MPI and np.size(E) > 1:
            comm = MPI_.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            local_E = np.array_split(E, size)[rank]
            local_result = np.zeros_like(local_E, dtype=np.float64)

            for start in range(0, len(self.energies), block_size):
                end = start + block_size
                energy_block = self.energies[start:end]
                coeff_block = self.coeffs[start:end]
                energy_differences = local_E[:, None] - energy_block[None, :]
                cutoff = np.exp(self.alpha * energy_differences)
                local_result += np.sum(
                    cutoff * coeff_block / energy_differences, axis=1
                )

            gathered = comm.gather(local_result, root=0)
            if rank == 0:
                return np.concatenate(gathered)
            return None

        energy_differences = E[:, None] - self.energies[None, :]
        cutoff = np.exp(self.alpha * energy_differences)
        result = np.sum(cutoff * self.coeffs / energy_differences, axis=-1)
        if debug:
            pdb.set_trace()
        return result[0] if np.size(E) == 1 else result

    def load_shifts(self, max_lvl: int = 50) -> Array:
        return convert_nsq_data(mL=self.mL, max_lvl=max_lvl)

    def Sigma4(self, E: ArrayLike, g44: float) -> Any:
        E = np.atleast_1d(E)
        s4 = self(E)
        return s4 / (1 - g44 * s4)

    def plot(
        self,
        Erange: ArrayLike,
        yrange: Sequence[float] | None = None,
        filename: str = "",
        **kwargs: Any,
    ) -> None:
        integral_values = self(Erange, **kwargs)
        _, ax = plt.subplots(1, 1)
        ax.plot(Erange, integral_values.real)
        ax.set_ylabel(r"$\mathcal{I}_4(E, \mathbf{P}=\mathbf{0})$")
        ax.set_xlabel("E/m")
        if yrange is not None:
            ax.set_ylim(yrange)

        if filename == "":
            filename = f"plots/I4_mL{self.mL}_EMAX{self.EMAX}_alpha{self.alpha}.pdf"
        utils.call_PDF(filename)


def main() -> None:
    mL = 6
    Erange = np.linspace(2, 6, 3000)
    TwoPionIntegral(mL=mL)
    FourPionIntegral(mL=mL)


if __name__ == "__main__":
    main()
