import numpy as np

from .ansr import ANSR, StochasticButcherTableau


tab = StochasticButcherTableau(
    c=np.array([3 / 4]),
    b=np.array([1 / 3, 2 / 3]),
    a=[np.array([3 / 4])],
    cw=np.array([0.0, 3 / 4]),
    ch=np.array([0.0, 1.5]),
    cw_last=1.0,
    ch_last=0.0,
)


class SRA1(ANSR):
    """Based on the SRA1 method from
    A. Rößler, Runge–Kutta methods for the strong approximation
    of solutions of stochastic differential equations,
    SIAM Journal on Numerical Analysis, 8 (2010), pp. 922–952.
    """

    tableau = tab

    def __init__(self):
        pass

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 1.5
