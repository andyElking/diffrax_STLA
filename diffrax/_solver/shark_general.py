from typing import ClassVar

import numpy as np

from .base import AbstractStratonovichSolver
from .srk import (
    AbstractSRK,
    GeneralCoeffs,
    SpaceTimeLevyAreaTableau,
    StochasticButcherTableau,
)


cfs_w = GeneralCoeffs(
    a=(np.array([0.0]), np.array([0.0, 5 / 6])),
    b=np.array([0.0, 0.4, 0.6]),
    b_error=None,
)

cfs_hh = GeneralCoeffs(
    a=(np.array([1.0]), np.array([1.0, 0.0])),
    b=np.array([0.0, 1.2, -1.2]),
    b_error=None,
)

cfs_bm = SpaceTimeLevyAreaTableau[GeneralCoeffs](
    coeffs_w=cfs_w,
    coeffs_hh=cfs_hh,
)

_tab = StochasticButcherTableau(
    c=np.array([0.0, 5 / 6]),
    b_sol=np.array([0.0, 0.4, 0.6]),
    b_error=None,
    a=[np.array([0.0]), np.array([0.0, 5 / 6])],
    cfs_bm=cfs_bm,
    ignore_stage_f=np.array([True, False, False]),
)


class GeneralShARK(AbstractSRK, AbstractStratonovichSolver):
    r"""ShARK method for Stratonovich SDEs.

    As compared to [`diffrax.ShARK`][] this can handle any SDE, not only those with
    additive noise.

    Makes two evaluations of the drift and three evaluations of the diffusion per step.
    For additive SDEs this has strong order 1.5. For general SDEs this has strong order
    0.5, but with a good coefficient in front (similar to making three steps of
    [`diffrax.Heun`][]).

    ??? cite "Reference"

        This solver is based on equation (6.1) from

        ```bibtex
        @misc{foster2023high,
          title={High order splitting methods for SDEs satisfying
            a commutativity condition},
          author={James Foster and Goncalo dos Reis and Calum Strange},
          year={2023},
          eprint={2210.17543},
          archivePrefix={arXiv},
          primaryClass={math.NA}
        ```
    """

    tableau: ClassVar[StochasticButcherTableau] = _tab

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 1.5
