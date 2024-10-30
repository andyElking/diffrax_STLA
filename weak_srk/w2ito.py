import numpy as np
from diffrax import AbstractItoSolver

from .weak_srk import AbstractWeakSRK, WeakTableau


_tab1 = WeakTableau(
    a0=(
        np.array([0.5]),
        np.array([-1.0, 2.0]),
    ),
    a1=(
        np.array([0.25]),
        np.array([0.25, 0.0]),
    ),
    b0=(
        np.array([(6 - np.sqrt(6)) / 10]),
        np.array([(3 + 2 * np.sqrt(6)) / 5, 0.0]),
    ),
    b1=(
        np.array([0.5]),
        np.array([-0.5, 0.0]),
    ),
    b2=(
        np.array([1.0]),
        np.array([0.0, 0.0]),
    ),
    alpha=np.array([1 / 6, 2 / 3, 1 / 6]),
    beta0=np.array([-1.0, 1.0, 1.0]),
    beta1=np.array([2.0, 0.0, -2.0]),
)


class W2Ito1(AbstractWeakSRK, AbstractItoSolver):
    tableau = _tab1

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 0.5


_tab2 = WeakTableau(
    a0=(
        np.array([0.5]),
        np.array([0.0, 0.5]),
        np.array([0.0, 0.0, 1.0]),
    ),
    a1=(
        np.array([0.5]),
        np.array([0.5, 0.0]),
        np.array([0.5, 0.0, 0.0]),
    ),
    b0=(
        np.array([0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 1.0, 0.0]),
    ),
    b1=(
        np.array([0.0]),
        np.array([0.0, 0.5]),
        np.array([0.0, -0.5, 0.0]),
    ),
    b2=(
        np.array([0.0]),
        np.array([0.0, 1.0]),
        np.array([0.0, 0.0, 0.0]),
    ),
    alpha=np.array([1 / 6, 1 / 3, 1 / 3, 1 / 6]),
    beta0=np.array([0.0, -1.0, 1.0, 1.0]),
    beta1=np.array([0.0, 2.0, 0.0, -2.0]),
)


class W2Ito2(AbstractWeakSRK, AbstractItoSolver):
    tableau = _tab2

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 0.5
