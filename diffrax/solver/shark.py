import numpy as np

from .ansr import AbstractANSR, StochasticButcherTableau


tab = StochasticButcherTableau(
    c=np.array([5 / 6]),
    b_sol=np.array([0.4, 0.6]),
    b_error=None,
    a=[np.array([5 / 6])],
    cw=np.array([0.0, 5 / 6]),
    ch=np.array([1.0, 1.0]),
    cw_last=1.0,
    ch_last=0.0,
)


class ShARK(AbstractANSR):
    r"""Shifted Additive-noise Runge-Kutta method for SDEs by James Foster.
    Applied to SDEs with additive noise, it converges strongly with order 1.5.

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

    tableau = tab

    def __init__(self):
        pass

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 1.5
