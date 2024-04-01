from typing import ClassVar

import numpy as np

from .base import AbstractStratonovichSolver
from .srk import (
    AbstractSRK,
    AdditiveCoeffs,
    SpaceTimeLevyAreaTableau,
    StochasticButcherTableau,
)


cfs_w = AdditiveCoeffs(
    a=np.array([0.0, 5 / 6]),
    b=np.array(1.0),
)

cfs_hh = AdditiveCoeffs(
    a=np.array([1.0, 1.0]),
    b=np.array(0.0),
)

cfs_bm = SpaceTimeLevyAreaTableau[AdditiveCoeffs](
    coeffs_w=cfs_w,
    coeffs_hh=cfs_hh,
)

_tab = StochasticButcherTableau(
    c=np.array([5 / 6]),
    b_sol=np.array([0.4, 0.6]),
    b_error=np.array([-0.6, 0.6]),
    a=[np.array([5 / 6])],
    cfs_bm=cfs_bm,
)


class ShARK(AbstractSRK, AbstractStratonovichSolver):
    r"""Shifted Additive-noise Runge-Kutta method for additive SDEs.

    Makes two evaluations of the drift and diffusion per step and has a strong order
    1.5.

    This is the recommended choice for SDEs with additive noise.

    See also [`diffrax.SRA1`][], which is very similar.

    ??? cite "Reference"

        This solver is based on equation (6.1) in

        ```bibtex
        @article{foster2023high,
            title={High order splitting methods for SDEs satisfying a commutativity
                   condition},
            author={James Foster and Goncalo dos Reis and Calum Strange},
            year={2023},
            journal={arXiv:2210.17543},
        }
        ```
    """

    tableau: ClassVar[StochasticButcherTableau] = _tab

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 1.5
