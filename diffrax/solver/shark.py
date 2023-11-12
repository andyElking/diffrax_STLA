import numpy as np

from .ansr import AbstractANSR, StochasticButcherTableau


tab = StochasticButcherTableau(
    c=np.array([5 / 6]),
    b=np.array([0.4, 0.6]),
    a=[np.array([5 / 6])],
    cw=np.array([0.0, 5 / 6]),
    ch=np.array([1.0, 1.0]),
    cw_last=1.0,
    ch_last=0.0,
)


class ShARK(AbstractANSR):
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
