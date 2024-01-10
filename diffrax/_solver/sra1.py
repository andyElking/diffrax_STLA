import numpy as np

from .srk import AbstractSRK, StochasticButcherTableau


tab = StochasticButcherTableau(
    c=np.array([3 / 4]),
    b_sol=np.array([1 / 3, 2 / 3]),
    b_error=np.array([-2 / 3, 2 / 3]),
    a=[np.array([3 / 4])],
    cW=np.array([0.0, 3 / 4]),
    cH=np.array([0.0, 1.5]),
    bW=np.array(1.0),
    bH=np.array(0.0),
)


class SRA1(AbstractSRK):
    r"""Based on the SRA1 method by Andreas Rößler.
    Works only for SDEs with additive noise, applied to whichit has strong order 1.5.
    Uses two evaluations of the vector field per step.

    ??? cite "Reference"

        ```bibtex
        @article{doi:10.1137/09076636X,
            author = {R\"{o}\ss{}ler, Andreas},
            title = {Runge–Kutta Methods for the Strong Approximation of
                Solutions of Stochastic Differential Equations},
            journal = {SIAM Journal on Numerical Analysis},
            volume = {48},
            number = {3},
            pages = {922-952},
            year = {2010},
            doi = {10.1137/09076636X},
            URL = {https://doi.org/10.1137/09076636X},
            eprint = {https://doi.org/10.1137/09076636X}
        ```
    """

    tableau = tab

    def __init__(self):
        super().__init__()

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 1.5