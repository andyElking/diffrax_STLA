import abc
from typing import Optional
import equinox as eqx
from dataclasses import field

from ..custom_types import Array, PyTree, Scalar
from ..path import AbstractPath


class LevyVal(eqx.Module):
    h: Scalar = field(default=0.0)
    W: PyTree[Array] = field(default=None)
    J: Optional[PyTree[Array]] = field(default=None)
    H: Optional[PyTree[Array]] = field(default=None)

    def wh(self):
        return self.W, self.H

    def wj(self):
        return self.W, self.J


class AbstractBrownianPath(AbstractPath):
    "Abstract base class for all Brownian paths."

    @abc.abstractmethod
    def evaluate(
            self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True, use_levy: bool =False
    ) -> PyTree[Array]:
        r"""Samples a Brownian increment $w(t_1) - w(t_0)$.

        Each increment has distribution $\mathcal{N}(0, t_1 - t_0)$.

        **Arguments:**

        - `t0`: Start of interval.
        - `t1`: End of interval.
        - `left`: Ignored. (This determines whether to treat the path as
            left-continuous or right-continuous at any jump points, but Brownian
            motion has no jump points.)

        **Returns:**

        A pytree of JAX arrays corresponding to the increment $w(t_1) - w(t_0)$.

        Some subclasses may allow `t1=None`, in which case just the value $w(t_0)$ is
        returned.
        """
