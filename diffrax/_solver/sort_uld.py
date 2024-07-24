import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from equinox.internal import ω
from jaxtyping import Array, ArrayLike, PyTree

from .._custom_types import (
    AbstractSpaceTimeTimeLevyArea,
    RealScalarLike,
)
from .._local_interpolation import LocalLinearInterpolation
from .._term import _LangevinArgs, LangevinTerm, LangevinX
from .langevin_srk import AbstractCoeffs, AbstractLangevinSRK, SolverState


# UBU evaluates at l = (3 -sqrt(3))/6, at r = (3 + sqrt(3))/6 and at 1,
# so we need 3 versions of each coefficient


class _SORTCoeffs(AbstractCoeffs):
    beta_half: PyTree[ArrayLike]
    a_half: PyTree[ArrayLike]
    b_half: PyTree[ArrayLike]
    beta1: PyTree[ArrayLike]
    a1: PyTree[ArrayLike]
    b1: PyTree[ArrayLike]
    aa: PyTree[ArrayLike]
    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(self, beta_half, a_half, b_half, beta1, a1, b1, aa):
        self.beta_half = beta_half
        self.a_half = a_half
        self.b_half = b_half
        self.beta1 = beta1
        self.a1 = a1
        self.b1 = b1
        self.aa = aa
        all_leaves = jtu.tree_leaves(
            [
                self.beta_half,
                self.a_half,
                self.b_half,
                self.beta1,
                self.a1,
                self.b1,
                self.aa,
            ]
        )
        self.dtype = jnp.result_type(*all_leaves)


class SORT(AbstractLangevinSRK[_SORTCoeffs, None]):
    r"""The Shifted-ODE Runge-Kutta Three method
    designed by James Foster. Only works for Underdamped Langevin Diffusion
    of the form

    $$d x_t = v_t dt$$

    $$d v_t = - gamma v_t dt - u ∇f(x_t) dt + (2gammau)^(1/2) dW_t$$

    where $v$ is the velocity, $f$ is the potential, $gamma$ is the friction, and
    $W$ is a Brownian motion.
    """

    term_structure = LangevinTerm
    interpolation_cls = LocalLinearInterpolation
    taylor_threshold: RealScalarLike = eqx.field(static=True)
    _coeffs_structure = jtu.tree_structure(
        _SORTCoeffs(
            beta_half=np.array(0.0),
            a_half=np.array(0.0),
            b_half=np.array(0.0),
            beta1=np.array(0.0),
            a1=np.array(0.0),
            b1=np.array(0.0),
            aa=np.array(0.0),
        )
    )
    minimal_levy_area = AbstractSpaceTimeTimeLevyArea

    def __init__(self, taylor_threshold: RealScalarLike = 0.1):
        r"""**Arguments:**

        - `taylor_threshold`: If the product `h*gamma` is less than this, then
        the Taylor expansion will be used to compute the coefficients.
        Otherwise they will be computed directly. When using float32, the
        empirically optimal value is 0.1, and for float64 about 0.01.
        """
        self.taylor_threshold = taylor_threshold

    def order(self, terms):
        return 3

    def strong_order(self, terms):
        return 3.0

    def _directly_compute_coeffs_leaf(self, h, c) -> _SORTCoeffs:
        del self
        # c is a leaf of gamma
        # compute the coefficients directly (as opposed to via Taylor expansion)
        al = c * h
        beta_half = jnp.exp(-al / 2)
        beta1 = jnp.exp(-al)
        a_half = (1 - beta_half) / c
        a1 = (1 - beta1) / c
        b_half = (beta_half + al / 2 - 1) / (al * c)
        b1 = (beta1 + al - 1) / (al * c)
        aa = a1 / h

        return _SORTCoeffs(
            beta_half=beta_half,
            a_half=a_half,
            b_half=b_half,
            beta1=beta1,
            a1=a1,
            b1=b1,
            aa=aa,
        )

    def _tay_coeffs_single(self, c: Array) -> _SORTCoeffs:
        del self
        # c is a leaf of gamma
        zero = jnp.zeros_like(c)
        one = jnp.ones_like(c)
        c2 = jnp.square(c)
        c3 = c2 * c
        c4 = c3 * c
        c5 = c4 * c

        beta_half = jnp.stack(
            [one, -c / 2, c2 / 8, -c3 / 48, c4 / 384, -c5 / 3840], axis=-1
        )
        beta1 = jnp.stack([one, -c, c2 / 2, -c3 / 6, c4 / 24, -c5 / 120], axis=-1)

        a_half = jnp.stack(
            [zero, one / 2, -c / 8, c2 / 48, -c3 / 384, c4 / 3840], axis=-1
        )
        a1 = jnp.stack([zero, one, -c / 2, c2 / 6, -c3 / 24, c4 / 120], axis=-1)
        # aa = a1/h
        aa = jnp.stack([one, -c / 2, c2 / 6, -c3 / 24, c4 / 120, -c5 / 720], axis=-1)

        # b_half is not exactly b(1/2 h), but 1/2 * b(1/2 h)
        b_half = jnp.stack(
            [zero, one / 8, -c / 48, c2 / 384, -c3 / 3840, c4 / 46080], axis=-1
        )
        b1 = jnp.stack([zero, one / 2, -c / 6, c2 / 24, -c3 / 120, c4 / 720], axis=-1)
        return _SORTCoeffs(
            beta_half=beta_half,
            a_half=a_half,
            b_half=b_half,
            beta1=beta1,
            a1=a1,
            b1=b1,
            aa=aa,
        )

    @staticmethod
    def _compute_step(
        h: RealScalarLike,
        levy: AbstractSpaceTimeTimeLevyArea,
        x0: LangevinX,
        v0: LangevinX,
        langevin_args: _LangevinArgs,
        coeffs: _SORTCoeffs,
        st: SolverState,
    ) -> tuple[LangevinX, LangevinX, LangevinX, None]:
        w: LangevinX = levy.W
        hh: LangevinX = levy.H
        kk: LangevinX = levy.K

        gamma, u, f = langevin_args

        rho_w_k = (st.rho**ω * (w**ω - 12 * kk**ω)).ω
        uh = (u**ω * h).ω

        f0 = st.prev_f
        v1 = (v0**ω + st.rho**ω * (hh**ω + 6 * kk**ω)).ω
        x1 = (
            x0**ω
            + coeffs.a_half**ω * v1**ω
            + coeffs.b_half**ω * (-(uh**ω) * f0**ω + rho_w_k**ω)
        ).ω
        f1 = f(x1)
        x_out = (
            x0**ω
            + coeffs.a1**ω * v1**ω
            + coeffs.b1**ω * (-(uh**ω) * (1 / 3 * f0**ω + 2 / 3 * f1**ω) + rho_w_k**ω)
        ).ω
        f_out = f(x_out)
        v_out = (
            coeffs.beta1**ω * v1**ω
            - uh**ω
            * (
                coeffs.beta1**ω / 6 * f0**ω
                + 2 / 3 * coeffs.beta_half**ω * f1**ω
                + 1 / 6 * f_out**ω
            )
            + coeffs.aa**ω * rho_w_k**ω
            - st.rho**ω * (hh**ω - 6 * kk**ω)
        ).ω

        return x_out, v_out, f_out, None
