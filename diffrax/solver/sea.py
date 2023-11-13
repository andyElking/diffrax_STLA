import numpy as np

from .ansr import AbstractANSR, StochasticButcherTableau


tab = StochasticButcherTableau(
    c=np.array([]),
    b_sol=np.array([1.0]),
    b_error=None,
    a=[],
    cw=np.array([0.5]),
    ch=np.array([1.0]),
    cw_last=1.0,
    ch_last=0.0,
)


class SEA(AbstractANSR):
    """Shifted Euler method for SDEs with additive noise.
     It has a local error of O(h^2) compared to
     standard Euler-Maruyama, which has O(h^1.5).

    Based on equation (5.8) in
    Foster, J., dos Reis, G., & Strange, C. (2023).
    High order splitting methods for SDEs satisfying a commutativity condition.
    arXiv [Math.NA] http://arxiv.org/abs/2210.17543
    """

    tableau = tab

    def __init__(self):
        pass

    def order(self, terms):
        return 1

    def strong_order(self, terms):
        return 0.5
