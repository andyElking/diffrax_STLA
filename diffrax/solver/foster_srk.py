import numpy as np

from .srk import AbstractSRK, StochasticButcherTableau


x1 = (3 - np.sqrt(3)) / 6
x2 = np.sqrt(3) / 3

tab = StochasticButcherTableau(
    c=np.array([0.5, 1.0]),
    b_sol=np.array([x1, x2, x1]),
    b_error=None,
    a=[np.array([0.5]), np.array([0.0, 1.0])],
    aW=[np.array([0.5]), np.array([0.0, 1.0])],
    aH=[np.array([np.sqrt(3)]), np.array([0.0, 0.0])],
    bW=np.array([x1, x2, x1]),
    bH=np.array([1.0, 0.0, -1.0]),
    additive_noise=False,
)


class FosterSRK(AbstractSRK):
    r"""A Stochastic Runge-Kutta method based on equation Definition $1.6$ from

    ??? cite "Reference"

        ```bibtex
        @misc{foster2023convergence,
            title={On the convergence of adaptive approximations
            for stochastic differential equations},
            author={James Foster},
            year={2023},
            eprint={2311.14201},
            archivePrefix={arXiv},
            primaryClass={math.NA}
        }
        ```

    """

    tableau = tab

    def __init__(self):
        pass

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 1.5
