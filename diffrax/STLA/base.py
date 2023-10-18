import abc
from typing import Optional
import equinox as eqx

from ..custom_types import Array, PyTree, Scalar
from ..path import AbstractPath


class BMInc(eqx.Module):
    h: Scalar
    W: PyTree[Array]
    J: Optional[PyTree[Array]]
    H: Optional[PyTree[Array]]


class AbstractSTLAPath(AbstractPath):
    "Abstract base class for all Brownian paths."

    @abc.abstractmethod
    def evaluate(
            self, t0: Scalar, t1: Optional[Scalar] = None, left: bool = True, use_hh: bool = False
    ) -> BMInc:
        r"""Samples a Brownian increment $w(t_1) - w(t_0)$.

        Each increment has distribution $\mathcal{N}(0, t_1 - t_0)$.

        **Arguments:**

        - `t0`: Start of interval.
        - `t1`: End of interval.
        - `left`: Ignored. (This determines whether to treat the path as
            left-continuous or right-continuous at any jump points, but Brownian
            motion has no jump points.)

        **Returns:**

        A pytree of JAX arrays corresponding to the increment $w(t_1) - w(t_0)$ and
        a pytree of the same shape corresponding to the STLA $H_{t_0, t_1}$.

        Some subclasses may allow `t1=None`, in which case just the value $w(t_0)$ is
        returned.
        """
