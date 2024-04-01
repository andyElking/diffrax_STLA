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
    a=np.array([0.5]),
    b=np.array(1.0),
)

cfs_hh = AdditiveCoeffs(
    a=np.array([1.0]),
    b=np.array(0.0),
)

cfs_bm = SpaceTimeLevyAreaTableau[AdditiveCoeffs](
    coeffs_w=cfs_w,
    coeffs_hh=cfs_hh,
)

_tab = StochasticButcherTableau(
    c=np.array([]),
    b_sol=np.array([1.0]),
    b_error=None,
    a=[],
    cfs_bm=cfs_bm,
)


class SEA(AbstractSRK, AbstractStratonovichSolver):
    r"""Shifted Euler method for SDEs with additive noise.

    Makes one evaluation of the drift and diffusion per step and has a strong order 1.
    Compared to [`diffrax.Euler`][], it has a better constant factor in the global
    error, and an improved local error of $O(h^2)$ instead of $O(h^{1.5})$.

    This solver is useful for solving additive-noise SDEs with as few drift and
    diffusion evaluations per step as possible.

    ??? cite "Reference"

        This solver is based on equation (5.8) in

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
        return 1

    def strong_order(self, terms):
        return 1
