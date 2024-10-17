from typing import Optional

import equinox as eqx
import jax.numpy as jnp
import jax.tree_util as jtu
from equinox.internal import ω
from jaxtyping import ArrayLike, PyTree

from .._custom_types import (
    AbstractSpaceTimeTimeLevyArea,
    RealScalarLike,
)
from .._local_interpolation import LocalLinearInterpolation
from .._term import (
    UnderdampedLangevinLeaf,
    UnderdampedLangevinX,
)
from .foster_langevin_srk import (
    AbstractCoeffs,
    AbstractFosterLangevinSRK,
    UnderdampedLangevinArgs,
)


# For an explanation of the coefficients, see foster_langevin_srk.py
class _nUBUCoeffs(AbstractCoeffs):
    ech_half: PyTree[ArrayLike]
    a_half: PyTree[ArrayLike]
    ech: PyTree[ArrayLike]
    a1: PyTree[ArrayLike]
    b1: PyTree[ArrayLike]
    aa: PyTree[ArrayLike]
    chh: PyTree[ArrayLike]
    ckk: PyTree[ArrayLike]
    lw: PyTree[ArrayLike]
    lhh: PyTree[ArrayLike]
    lkk: PyTree[ArrayLike]

    dtype: jnp.dtype = eqx.field(static=True)

    def __init__(self, ech_half, a_half, ech, a1, b1, aa, chh, ckk, lw, lhh, lkk):
        self.ech_half = ech_half
        self.a_half = a_half
        self.ech = ech
        self.a1 = a1
        self.b1 = b1
        self.aa = aa
        self.chh = chh
        self.ckk = ckk
        self.lw = lw
        self.lhh = lhh
        self.lkk = lkk
        all_leaves = jtu.tree_leaves(
            [
                self.ech_half,
                self.a_half,
                self.ech,
                self.a1,
                self.b1,
                self.aa,
                self.chh,
                self.ckk,
                self.lw,
                self.lhh,
                self.lkk,
            ]
        )
        self.dtype = jnp.result_type(*all_leaves)


class nUBU(AbstractFosterLangevinSRK[_nUBUCoeffs, None]):
    r"""The nUBU method by Daire O'Kane,
    based on the UBU method by Alfonso Álamo Zapatero.
    This is a second order solver for the Underdamped Langevin Diffusion.
    Uses one vector field evaluation per step.

    ??? cite "Reference"

        ```bibtex
        @inproceedings{Zapatero2017WordSF,
          title={Word series for the numerical integration of stochastic differential equations},
          author={Alfonso {\'A}lamo Zapatero},
          year={2017},
          url={https://api.semanticscholar.org/CorpusID:125699606}
        }
        ```

    """

    interpolation_cls = LocalLinearInterpolation
    minimal_levy_area = AbstractSpaceTimeTimeLevyArea
    taylor_threshold: RealScalarLike = eqx.field(static=True)
    _is_fsal = False

    def __init__(self, taylor_threshold: RealScalarLike = 0.1):
        r"""**Arguments:**

        - `taylor_threshold`: If the product `h*gamma` is less than this, then
        the Taylor expansion will be used to compute the coefficients.
        Otherwise they will be computed directly. When using float32, the
        empirically optimal value is 0.1, and for float64 about 0.01.
        """
        self.taylor_threshold = taylor_threshold

    def order(self, terms):
        return 2

    def strong_order(self, terms):
        return 2.0

    def _directly_compute_coeffs_leaf(
        self, h: RealScalarLike, c: UnderdampedLangevinLeaf
    ) -> _nUBUCoeffs:
        del self
        # c is a leaf of gamma
        # compute the coefficients directly (as opposed to via Taylor expansion)
        ch = c * h
        ech_half = jnp.exp(-ch / 2)
        ech = jnp.exp(-ch)
        a_half = (1 - ech_half) / c
        a1 = (1 - ech) / c
        b1 = (ech + ch - 1) / (ch * c)
        aa = a1 / h

        ch2 = ch**2
        ch3 = ch2 * ch
        chh_by_6 = (ech * (ch + 2) + ch - 2) / (ch2 * c)
        chh = 6 * chh_by_6
        ckk = 60 * (ech * (ch * (ch + 6) + 12) - ch * (ch - 6) - 12) / (ch3 * c)

        inv_ech_half = jnp.exp(ch / 2)
        lw = inv_ech_half * chh_by_6
        lhh = 6 * inv_ech_half * (ech * (ch * (ch + 4) + 6) + 2 * ch - 6) / (ch3 * c)
        lkk = (
            60
            * inv_ech_half
            * (ech * (ch * (ch * (ch + 8) + 30) + 48) - 2 * (ch * (ch - 9) + 24))
            / (ch3 * ch * c)
        )

        return _nUBUCoeffs(
            ech_half=ech_half,
            a_half=a_half,
            ech=ech,
            a1=a1,
            b1=b1,
            aa=aa,
            chh=chh,
            ckk=ckk,
            lw=lw,
            lhh=lhh,
            lkk=lkk,
        )

    def _tay_coeffs_single(self, c: UnderdampedLangevinLeaf) -> _nUBUCoeffs:
        del self
        # c is a leaf of gamma
        zero = jnp.zeros_like(c)
        one = jnp.ones_like(c)
        c2 = jnp.square(c)
        c3 = c2 * c
        c4 = c3 * c
        c5 = c4 * c

        # Coefficients of the Taylor expansion, starting from 5th power
        # to 0th power. The descending power order is because of jnp.polyval
        ech_half = jnp.stack(
            [-c5 / 3840, c4 / 384, -c3 / 48, c2 / 8, -c / 2, one], axis=-1
        )
        ech = jnp.stack([-c5 / 120, c4 / 24, -c3 / 6, c2 / 2, -c, one], axis=-1)
        a_half = jnp.stack(
            [c4 / 3840, -c3 / 384, c2 / 48, -c / 8, one / 2, zero], axis=-1
        )
        a1 = jnp.stack([c4 / 120, -c3 / 24, c2 / 6, -c / 2, one, zero], axis=-1)
        aa = jnp.stack([-c5 / 720, c4 / 120, -c3 / 24, c2 / 6, -c / 2, one], axis=-1)
        b1 = jnp.stack([c4 / 720, -c3 / 120, c2 / 24, -c / 6, one / 2, zero], axis=-1)
        chh = jnp.stack([c4 / 168, -c3 / 30, 3 * c2 / 20, -c / 2, one, zero], axis=-1)
        ckk = jnp.stack([5 * c4 / 168, -c3 / 7, c2 / 2, -c, zero, zero], axis=-1)

        lw = jnp.stack([c4 / 26880, zero, c2 / 240, zero, one / 6, zero], axis=-1)
        lhh = jnp.stack(
            [c4 / 8960, -c3 / 1120, c2 / 80, -c / 20, one / 2, zero], axis=-1
        )
        lkk = jnp.stack(
            [5 * c4 / 8064, -c3 / 112, 3 * c2 / 56, -c / 2, one, zero], axis=-1
        )

        correct_shape = jnp.shape(c) + (6,)
        assert (
            ech_half.shape
            == a_half.shape
            == ech.shape
            == a1.shape
            == b1.shape
            == aa.shape
            == chh.shape
            == ckk.shape
            == lw.shape
            == lhh.shape
            == lkk.shape
            == correct_shape
        )

        return _nUBUCoeffs(
            ech_half=ech_half,
            a_half=a_half,
            ech=ech,
            a1=a1,
            b1=b1,
            aa=aa,
            chh=chh,
            ckk=ckk,
            lw=lw,
            lhh=lhh,
            lkk=lkk,
        )

    def _compute_step(
        self,
        h: RealScalarLike,
        levy: AbstractSpaceTimeTimeLevyArea,
        x0: UnderdampedLangevinX,
        v0: UnderdampedLangevinX,
        underdamped_langevin_args: UnderdampedLangevinArgs,
        coeffs: _nUBUCoeffs,
        rho: UnderdampedLangevinX,
        prev_f: Optional[UnderdampedLangevinX],
    ) -> tuple[
        UnderdampedLangevinX,
        UnderdampedLangevinX,
        UnderdampedLangevinX,
        None,
    ]:
        del prev_f
        dtypes = jtu.tree_map(jnp.result_type, x0)
        w: UnderdampedLangevinX = jtu.tree_map(jnp.asarray, levy.W, dtypes)
        hh: UnderdampedLangevinX = jtu.tree_map(jnp.asarray, levy.H, dtypes)
        kk: UnderdampedLangevinX = jtu.tree_map(jnp.asarray, levy.K, dtypes)

        gamma, u, f = underdamped_langevin_args

        z = (
            x0**ω
            + coeffs.a_half**ω * v0**ω
            + rho**ω
            * (coeffs.lw**ω * w**ω + coeffs.lhh**ω * hh**ω + coeffs.lkk**ω * kk**ω)
        ).ω
        fz = f(z)
        fz_uh = (fz**ω * u**ω * h).ω
        hh_kk = (coeffs.chh**ω * hh**ω + coeffs.ckk**ω * kk**ω).ω
        v1 = (
            coeffs.ech**ω * v0**ω
            - coeffs.ech_half**ω * fz_uh**ω
            + rho**ω * (coeffs.aa**ω * w**ω - gamma**ω * hh_kk**ω)
        ).ω
        x1 = (
            x0**ω
            + coeffs.a1**ω * v0**ω
            - coeffs.a_half**ω * fz_uh**ω
            + rho**ω * (coeffs.b1**ω * w**ω + hh_kk**ω)
        ).ω

        return x1, v1, None, None
