from typing import ClassVar

import numpy as np

from .base import AbstractStratonovichSolver
from .srk import AbstractSRK, GeneralSpaceTimeLACoeffs, StochasticButcherTableau


cfs_bm = GeneralSpaceTimeLACoeffs(
    a_w=(np.array([0.0]), np.array([0.0, 5 / 6])),
    b_w=np.array([0.0, 0.4, 0.6]),
    a_hh=(np.array([1.0]), np.array([1.0, 0.0])),
    b_hh=np.array([0.0, 1.2, -1.2]),
    b_error=None,
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
    r"""A generalised version of the ShARK method which now works for
    any SDE, not only those with additive noise.
    Applied to SDEs with additive noise, it still has strong order 1.5.
    Uses three evaluations of the vector field per step.

    Based on equation $(6.1)$ in

    ??? cite "Reference"

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
