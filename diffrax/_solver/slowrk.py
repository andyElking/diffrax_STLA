from typing import ClassVar

import numpy as np

from .base import AbstractStratonovichSolver
from .srk import (
    AbstractSRK,
    GeneralCoeffs,
    SpaceTimeLevyAreaTableau,
    StochasticButcherTableau,
)


# TODO: maybe this solver should use a custom implementation, not going via
# `AbstractSRK`? Its tableaus have huge amounts of sparsity due to its odd structure.


cfs_w = GeneralCoeffs(
    a=(
        np.array([0.0]),
        np.array([0.0, 0.5]),
        np.array([0.0, 0.0, 0.5]),
        np.array([0.0, 0.0, 0.0, 1.0]),
        np.array([0.0, 0.0, 0.0, 0.75, 0.0]),
        np.array([0.0, 0.0, 0.5, 0.0, 0.0, 0.0]),
    ),
    b=np.array([0.0, 1 / 6, 1 / 3, 1 / 3, 1 / 6, 0.0, 0.0]),
    b_error=None,
)

cfs_hh = GeneralCoeffs(
    a=(
        np.array([0.0]),
        np.array([0.0, 0.0]),
        np.array([0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0]),
        np.array([0.0, 0.0, 0.0, 1.5, 0.0]),
        np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ),
    b=np.array([0.0, 0.0, 0.0, 2.0, 0.0, 0.0, -2.0]),
    b_error=None,
)

cfs_bm = SpaceTimeLevyAreaTableau[GeneralCoeffs](
    coeffs_w=cfs_w,
    coeffs_hh=cfs_hh,
)

_tab = StochasticButcherTableau(
    c=np.array([0.5, 0.5, 0.5, 0.5, 0.75, 1.0]),
    b_sol=np.array([1 / 3, 0.0, 0.0, 0.0, 0.0, 2 / 3, 0.0]),
    b_error=None,
    a=[
        np.array([0.5]),
        np.array([0.5, 0.0]),
        np.array([0.5, 0.0, 0.0]),
        np.array([0.5, 0.0, 0.0, 0.0]),
        np.array([0.75, 0.0, 0.0, 0.0, 0.0]),
        np.array([1.0, 0.0, 0.0, 0.0, 0.0, 0.0]),
    ],
    cfs_bm=cfs_bm,
    ignore_stage_f=np.array([False, True, True, True, True, False, True]),
    ignore_stage_g=np.array([True, False, False, False, False, True, False]),
)


class SlowRK(AbstractSRK, AbstractStratonovichSolver):
    r"""SLOW-RK method for commutative-noise Stratonovich SDEs.

    Makes two evaluations of the drift and five evaluations of the diffusion per step.
    Applied to SDEs with commutative noise, it converges strongly with order 1.5.
    Can be used for SDEs with non-commutative noise, but then it only converges
    strongly with order 0.5.

    This solver is an excellent choice for Stratonovich SDEs with commutative noise.

    ??? cite "Reference"

        This solver is based on equation (6.2) from

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
        return 0.5
