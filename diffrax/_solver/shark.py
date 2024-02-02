from typing import ClassVar

import numpy as np

from .base import AbstractStratonovichSolver
from .srk import AbstractSRK, StochasticButcherTableau


_tab = StochasticButcherTableau(
    c=np.array([5 / 6]),
    b_sol=np.array([0.4, 0.6]),
    b_error=np.array([-0.6, 0.6]),
    a=[np.array([5 / 6])],
    cW=np.array([0.0, 5 / 6]),
    cH=np.array([1.0, 1.0]),
    bW=np.array(1.0),
    bH=np.array(0.0),
)


class ShARK(AbstractSRK, AbstractStratonovichSolver):
    r"""Shifted Additive-noise Runge-Kutta method for SDEs by James Foster.
    Applied to SDEs with additive noise, it has strong order 1.5.
    Uses two evaluations of the vector field per step.

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
