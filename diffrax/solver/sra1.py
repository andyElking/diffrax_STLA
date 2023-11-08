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
    """Shifted Additive-noise Runge-Kutta method for SDEs.
    When applied to SDEs with additive noise, it converges
    strongly with order 1.5.

    Based on equation (6.1) in
    Foster, J., dos Reis, G., & Strange, C. (2023).
    High order splitting methods for SDEs satisfying a commutativity condition.
    arXiv [Math.NA] http://arxiv.org/abs/2210.17543
    """

    tableau = tab

    def __init__(self):
        pass

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 1.5
