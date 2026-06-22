import pdb
import mpi4py.MPI as MPI_

import numpy as np
import matplotlib.pyplot as plt
import utils
from S4_utils import convert_nsq_data
from energy_cutoff import mult


class TwoPionIntegral:
    """
    Calculates the divergent part of the loop integral involving two
    particles with 1 independent momentum.
    """

    def __init__(self, mL, EMAX=20, alpha=1.0):
        self.mL = mL
        self.EMAX, self.alpha = EMAX,  alpha
        self.tpiL = 2.0*np.pi/mL
        self.prefactor = 1.0/(2*(mL**3))

        self.levels, self.energies, self.coeffs = self.get_energy_levels()

    def get_energy_levels(self):
        levels = np.load("data/N_100_ordered_pairs.npy", allow_pickle=True)
        energies = np.sum((levels*self.tpiL**2+1)**0.5, axis=-1)
        energies = energies[energies <= self.EMAX]

        levels = levels[:len(energies), :].astype(int)
        prod_omegas = np.prod(2*(levels*self.tpiL**2+1)**0.5, axis=-1)
        coeffs = np.array(
            (self.prefactor/prod_omegas), dtype=np.float32)

        return levels, energies, coeffs

    def __call__(self, E):
        E = np.atleast_1d(E)
        Ediff = E[:, None] - self.energies[None, :]
        resid = np.broadcast_to(self.coeffs, Ediff.shape)
        cutoff = np.exp(self.alpha*Ediff)
        result = np.sum(cutoff*resid/Ediff, axis=-1)
        return result[0] if np.size(E) == 1 else result

    def remainders(self, max_lvl=None):
        N_lvl = max_lvl or len(np.where(self.energies < 4)[0])
        mask = ~np.eye(N_lvl, dtype=bool)

        E = self.energies[:N_lvl, None] - self.energies[None, :N_lvl]
        np.fill_diagonal(E, 1.0)
        coeffs_matrix = mask * \
            np.broadcast_to(self.coeffs[:N_lvl], (N_lvl, N_lvl))

        cutoff = mask*np.exp(self.alpha * E)
        add = self.alpha * self.coeffs[:N_lvl]

        result = add + np.sum(cutoff * coeffs_matrix / E, axis=-1)
        return result

    def plot(self, Erange, yrange=None, filename=""):
        Irange = self(Erange)
        fig, ax = plt.subplots(1, 1)
        ax.plot(Erange, Irange.real)
        ax.set_ylabel(r"$Im\mathcal{I}_2(E, \mathbf{P})$")
        ax.set_xlabel("E")
        if yrange is not None:
            ax.set_ylim(yrange)

        if filename == "":
            filename = f"plots/I2_mL{self.mL}_EMAX{
                self.EMAX}_alpha{self.alpha}.pdf"
        utils.call_PDF(filename)


class FourPionIntegral:
    """
    Calculates the divergent part of the loop integral involving four
    particles with 3 independent momenta.
    """

    def __init__(self, mL, g22=0, EMAX=20, alpha=1.0, **kwargs):
        self.mL = mL
        self.EMAX, self.alpha = EMAX, alpha
        self.tpiL = 2.0*np.pi/mL
        self.prefactor = 1.0/(24*(mL**9))

        self.levels, self.energies, self.coeffs = self.get_energy_levels()
        self.shifts = self.load_shifts(**kwargs)
        self.unshifted_energies = self.energies.copy()
        self.shift_levels(g22)

    def get_energy_levels(self, **kwargs):
        levels = np.load("data/N_8_ordered_quads.npy", allow_pickle=True)
        energies = np.sum((levels*self.tpiL**2+1)**0.5, axis=-1)
        energies = energies[energies <= self.EMAX]

        levels = levels[:len(energies), :].astype(int)
        prod_omegas = np.prod(2*(levels*self.tpiL**2+1)**0.5, axis=-1)

        C_kks = np.array([mult(levels[i, :]) for i in range(levels.shape[0])])
        coeffs = np.array(
            (self.prefactor*C_kks/prod_omegas), dtype=np.float32)

        return levels, energies, coeffs

    def shift_levels(self, g22, max_lvl=50):
        self.g22 = g22
        g22L3 = g22/(self.mL**3)
        N_lvls = len(self.shifts)
        self.energies[:N_lvls] = self.unshifted_energies[:N_lvls] + \
            g22L3*self.shifts

    def __call__(self, E, MPI=True, block_size=1000):
        E = np.atleast_1d(E)
        if MPI and np.size(E) > 1:
            comm = MPI_.COMM_WORLD
            rank = comm.Get_rank()
            size = comm.Get_size()

            local_E = np.array_split(E, size)[rank]
            local_result = np.zeros_like(local_E, dtype=np.float64)

            for start in range(0, len(self.energies), block_size):
                end = start + block_size
                eps_block = self.energies[start:end]
                coeff_block = self.coeffs[start:end]
                Ediff = local_E[:, None] - eps_block[None, :]
                cutoff = np.exp(self.alpha * Ediff)
                local_result += np.sum(cutoff * coeff_block / Ediff, axis=1)

            gathered = comm.gather(local_result, root=0)
            if rank == 0:
                return np.concatenate(gathered)
            else:
                return None

        Ediff = E[:, None] - self.energies[None, :]
        resid = np.broadcast_to(self.coeffs, Ediff.shape)
        cutoff = np.exp(self.alpha*Ediff)
        result = np.sum(cutoff*resid/Ediff, axis=-1)
        return result[0] if np.size(E) == 1 else result

    def load_shifts(self, max_lvl=50):
        q_tildes = convert_nsq_data(mL=self.mL, max_lvl=max_lvl)
        return q_tildes

    def Sigma4(self, E, g44):
        E = np.atleast_1d(E)
        S4 = self(E)
        return S4/(1-g44*S4)

    def plot(self, Erange, yrange=None, filename="", **kwargs):
        Irange = self(Erange, **kwargs)
        fig, ax = plt.subplots(1, 1)
        ax.plot(Erange, Irange.real)
        ax.set_ylabel(r"$\mathcal{I}_4(E, \mathbf{P})$")
        ax.set_xlabel("E/m")
        if yrange is not None:
            ax.set_ylim(yrange)

        if filename == "":
            filename = f"plots/I4_mL{self.mL}_EMAX{
                self.EMAX}_alpha{self.alpha}.pdf"
        utils.call_PDF(filename)


if __name__ == "__main__":
    mL = 6
    Erange = np.linspace(2, 6, 3000)
    I2 = TwoPionIntegral(mL=mL)
    I4 = FourPionIntegral(mL=mL)
