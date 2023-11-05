#  Copyright (c) Ergodic LLC 2023
#  research@ergodic.io

from jax import numpy as jnp
from jax.lax import scan
import equinox as eqx


class TridiagonalSolver(eqx.Module):
    num_unroll: int

    def __init__(self, cfg):
        super(TridiagonalSolver, self).__init__()
        self.num_unroll = 16

    @staticmethod
    def compute_primes(last_primes, x):
        """
        This function is a single iteration of the forward pass in the non-in-place Thomas
        tridiagonal algorithm

        :param last_primes:
        :param x:
        :return:
        """

        last_cp, last_dp = last_primes
        a, b, c, d = x
        cp = c / (b - a * last_cp)
        dp = (d - a * last_dp) / (b - a * last_cp)
        new_primes = jnp.stack((cp, dp))
        return new_primes, new_primes

    @staticmethod
    def backsubstitution(last_x, x):
        """
        This function is a single iteration of the backward pass in the non-in-place Thomas
        tridiagonal algorithm

        :param last_x:
        :param x:
        :return:
        """
        cp, dp = x
        new_x = dp - cp * last_x
        return new_x, new_x

    def __call__(self, a, b, c, d):
        """
        Solves a tridiagonal matrix system with diagonals a, b, c and RHS vector d.

        This uses the non-in-place Thomas tridiagonal algorithm.

        The NumPy version, on the other hand, uses the in-place algorithm.

        :param a: (2D float array (nx, nv)) represents the subdiagonal of the linear operator
        :param b: (2D float array (nx, nv)) represents the main diagonal of the linear operator
        :param c: (2D float array (nx, nv)) represents the super diagonal of the linear operator
        :param d: (2D float array (nx, nv)) represents the right hand side of the linear operator
        :return:
        """

        diags_stacked = jnp.stack([arr.transpose((1, 0)) for arr in (a, b, c, d)], axis=1)
        _, primes = scan(self.compute_primes, jnp.zeros((2, *a.shape[:-1])), diags_stacked, unroll=self.num_unroll)
        _, sol = scan(self.backsubstitution, jnp.zeros(a.shape[:-1]), primes[::-1], unroll=self.num_unroll)
        return sol[::-1].transpose((1, 0))
