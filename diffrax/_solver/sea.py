import numpy as np

from .srk import AbstractSRK, StochasticButcherTableau


tab = StochasticButcherTableau(
    c=np.array([]),
    b_sol=np.array([1.0]),
    b_error=None,
    a=[],
    cW=np.array([0.5]),
    cH=np.array([1.0]),
    bW=np.array(1.0),
    bH=np.array(0.0),
)


class SEA(AbstractSRK):
    r"""Shifted Euler method for SDEs with additive noise.
     It has a local error of $O(h^2)$ compared to
     standard Euler-Maruyama, which has $O(h^{1.5})$.
     Uses one evaluation of the vector field per step and
     has order 1 for additive noise SDEs.

    Based on equation $(5.8)$ in
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

    tableau = tab

    def __init__(self):
        super().__init__()

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 1
