import math

import diffrax
import jax
import jax.numpy as jnp
import jax.random as jrandom
import jax.tree_util as jtu
import pytest
import scipy.stats as stats


_vals = {
    int: [0, 2],
    float: [0.0, 2.0],
    jnp.int32: [jnp.array(0, dtype=jnp.int32), jnp.array(2, dtype=jnp.int32)],
    jnp.float32: [jnp.array(0.0, dtype=jnp.float32), jnp.array(2.0, dtype=jnp.float32)],
}


def _make_struct(shape, dtype):
    dtype = jax.dtypes.canonicalize_dtype(dtype)
    return jax.ShapeDtypeStruct(shape, dtype)


@pytest.mark.parametrize(
    "ctr", [diffrax.UnsafeBrownianPath, diffrax.VirtualBrownianTree]
)
@pytest.mark.parametrize("levy_area", ["space-time", "space-time-time"])
def test_shape_and_dtype(ctr, levy_area, getkey):
    t0 = 0
    t1 = 2

    shapes = (
        (),
        (0,),
        (1, 0),
        (2,),
        (3, 4),
        (1, 2, 3, 4),
        {
            "a": (1,),
            "b": (2, 3),
        },
        (
            (1, 2),
            (
                (3, 4),
                (5, 6),
            ),
        ),
    )

    dtypes = (
        None,
        None,
        None,
        jnp.float16,
        jnp.float32,
        jnp.float64,
        {"a": None, "b": jnp.float64},
        (jnp.float16, (jnp.float32, jnp.float64)),
    )

    def is_tuple_of_ints(obj):
        return isinstance(obj, tuple) and all(isinstance(x, int) for x in obj)

    for shape, dtype in zip(shapes, dtypes):
        # Shape to pass as input
        if dtype is not None:
            shape = jtu.tree_map(_make_struct, shape, dtype, is_leaf=is_tuple_of_ints)

        if ctr is diffrax.UnsafeBrownianPath:
            path = ctr(shape, getkey(), levy_area=levy_area)
            assert path.t0 is None
            assert path.t1 is None
        elif ctr is diffrax.VirtualBrownianTree:
            tol = 2**-5
            path = ctr(t0, t1, tol, shape, getkey(), levy_area=levy_area)
            assert path.t0 == 0
            assert path.t1 == 2
        else:
            assert False

        # Expected output shape
        if dtype is None:
            shape = jtu.tree_map(_make_struct, shape, dtype, is_leaf=is_tuple_of_ints)

        for _t0 in _vals.values():
            for _t1 in _vals.values():
                t0, _ = _t0
                _, t1 = _t1
                bm = path.evaluate(t0, t1, use_levy=True)
                out_w = bm.W
                out_h = bm.H
                out_w_shape = jtu.tree_map(
                    lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), out_w
                )
                out_h_shape = jtu.tree_map(
                    lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), out_h
                )
                if levy_area == "space-time-time":
                    out_k = bm.K
                    out_k_shape = jtu.tree_map(
                        lambda leaf: jax.ShapeDtypeStruct(leaf.shape, leaf.dtype), out_k
                    )
                    assert out_k_shape == shape
                assert out_h_shape == shape
                assert out_w_shape == shape


@pytest.mark.parametrize(
    "ctr", [diffrax.UnsafeBrownianPath, diffrax.VirtualBrownianTree]
)
@pytest.mark.parametrize("levy_area", ["space-time", "space-time-time"])
def test_statistics(ctr, levy_area):
    # Deterministic key for this test; not using getkey()
    key = jrandom.PRNGKey(5678)
    keys = jrandom.split(key, 10000)

    def _eval(key):
        if ctr is diffrax.UnsafeBrownianPath:
            path = ctr(shape=(), key=key, levy_area=levy_area)
        elif ctr is diffrax.VirtualBrownianTree:
            path = ctr(t0=0, t1=5, tol=2**-5, shape=(), key=key, levy_area=levy_area)
        else:
            assert False
        return path.evaluate(0, 3, use_levy=True)

    bm_inc = jax.vmap(_eval)(keys)
    values_w = bm_inc.W
    values_h = bm_inc.H
    assert values_w.shape == (10000,) and values_h.shape == (10000,)
    ref_dist_w = stats.norm(loc=0, scale=math.sqrt(3))
    _, pval_w = stats.kstest(values_w, ref_dist_w.cdf)
    ref_dist_h = stats.norm(loc=0, scale=math.sqrt(3 / 12))
    _, pval_h = stats.kstest(values_h, ref_dist_h.cdf)
    assert pval_w > 0.1
    assert pval_h > 0.1
    if levy_area == "space-time-time":
        values_k = bm_inc.K
        assert values_k.shape == (10000,)
        ref_dist_k = stats.norm(loc=0, scale=math.sqrt(3 / 720))
        _, pval_k = stats.kstest(values_k, ref_dist_k.cdf)
        assert pval_k > 0.1


@pytest.mark.parametrize("levy_area", ["space-time", "space-time-time"])
def test_conditional_statistics(levy_area):
    key = jrandom.PRNGKey(5678)
    bm_key, sample_key, permute_key = jrandom.split(key, 3)

    # Get >80 randomly selected points; not too close to avoid discretisation error.
    t0 = 0.3
    t1 = 8.7
    ts = jrandom.uniform(sample_key, shape=(100,), minval=t0, maxval=t1)
    sorted_ts = jnp.sort(ts)
    ts = []
    prev_ti = sorted_ts[0]
    for ti in sorted_ts[1:]:
        if ti < prev_ti + 2**-9:
            continue
        prev_ti = ti
        ts.append(ti)
    ts = jnp.stack(ts)
    assert len(ts) > 80
    ts = jrandom.permutation(permute_key, ts)

    # Get some random paths
    bm_keys = jrandom.split(bm_key, 100000)

    path = jax.vmap(
        lambda k: diffrax.VirtualBrownianTree(
            t0=t0, t1=t1, shape=(), tol=2**-12, key=k, levy_area=levy_area
        )
    )(bm_keys)

    # Sample some points
    out = []
    for ti in ts:
        vals = jax.vmap(lambda p: p.evaluate(t0, ti, use_levy=True))(path)
        out.append((ti, vals))
    out = sorted(out, key=lambda x: x[0])

    # Test their conditional statistics
    for i in range(1, len(ts) - 2):
        s, bm_s = out[i - 1]
        r, bm_r = out[i]
        u, bm_u = out[i + 1]

        w_s, h_s = bm_s.W, bm_s.H
        w_r, h_r = bm_r.W, bm_r.H
        w_u, h_u = bm_u.W, bm_u.H
        bh_s = s * h_s
        bh_r = r * h_r
        bh_u = u * h_u
        s = s - t0
        r = r - t0
        u = u - t0
        su = u - s
        sr = r - s
        ru = u - r
        sr3 = jnp.power(sr, 3)
        ru3 = jnp.power(ru, 3)
        su3 = jnp.power(su, 3)
        sr_ru_half = jnp.sqrt(sr * ru)
        if levy_area == "space-time":
            d = jnp.sqrt(sr3 + ru3)
            d_prime = 1 / (2 * su * d)
            a = d_prime * sr3 * sr_ru_half
            b = d_prime * ru3 * sr_ru_half
            c = jnp.sqrt(3 * sr3 * ru3) / (6 * d)

            u_bb_s = u * w_s - s * w_u  # bb_s = brownian bridge on [0,u] evaluated at s

            # bh_su = \bar{H}_{s,u} := (u-s) H_{s,u}
            bh_su = bh_u - bh_s - 0.5 * u_bb_s

            w_mean = w_s + (sr / su) * (w_u - w_s) + (6 * sr * ru / su3) * bh_su
            w_std = 2 * (a + b) / su
            normalised_w = (w_r - w_mean) / w_std
            h_mean = (
                (1 / r) * bh_s
                + (sr3 / (r * su3)) * bh_su
                + 0.5 * w_s
                - s / (2 * r) * w_mean
            )
            h_var = jnp.square(c / r) + jnp.square((a * u + s * b) / (r * su))
            h_std = jnp.sqrt(h_var)
            normalised_h = (h_r - h_mean) / h_std

            _, pval_w = stats.kstest(normalised_w, stats.norm.cdf)
            _, pval_h = stats.kstest(normalised_h, stats.norm.cdf)

            # Raise if the failure is statistically significant at 10%, subject to
            # multiple-testing correction.
            assert pval_w > 0.001
            assert pval_h > 0.001
        elif levy_area == "space-time-time":
            # TODO: implement this
            su5 = jnp.power(su, 5)
            sr2 = jnp.square(sr)
            ru2 = jnp.square(ru)

            k_s, k_r, k_u = bm_s.K, bm_r.K, bm_u.K
            bk_s = s**2 * k_s
            bk_r = r**2 * k_r
            bk_u = u**2 * k_u
            # u_bb_s := u * brownian bridge on [0,u] evaluated at s
            u_bb_s = u * w_s - s * w_u

            # Chen's relation for H
            bh_su = bh_u - bh_s - 0.5 * u_bb_s

            # Chen's relation for \bar{K}_{s,u} := (u-s)^2 * K_{s,u}
            bk_su = (
                bk_u - bk_s - su / 2 * bh_u + s / 2 * bh_su - (u - 2 * s) / 12 * u_bb_s
            )

            # compute the mean of (W_sr, H_sr, K_sr) conditioned on
            # (W_s, H_s, K_s, W_u, H_u, K_u)
            bb_mean = (
                6 * sr * ru / su3 * bh_su + 120 * sr * ru * (ru - sr) / 2 / su5 * bk_su
            )
            w_mean = sr / su * (w_u - w_s) + bb_mean
            h_mean = sr2 / su3 * bh_su + 30 * sr2 * ru / su5 * bk_su
            k_mean = sr3 / su3 * bk_su

            mean = jnp.array([w_mean, h_mean, k_mean])

            # now compute the covariance matrix of (W_sr, H_sr, K_sr) conditioned on
            # (W_s, H_s, K_s, W_u, H_u, K_u)
            # note that the covariance matrix is independent of the values of
            # (W_s, H_s, K_s, W_u, H_u, K_u), since those are already represented
            # in the mean.
            sr5 = jnp.power(sr, 5)

            ww_cov = (sr * ru * ((sr - ru) ** 4 + 4 * (sr2 * ru2))) / su5
            wh_cov = -(sr3 * ru * (sr2 - 3 * sr * ru + 6 * ru2)) / (2 * su5)
            wk_cov = (sr**4 * ru * (sr - ru)) / (12 * su5)
            hh_cov = sr / 12 * (1 - (sr3 * (sr2 + 2 * sr * ru + 16 * ru2)) / su5)
            hk_cov = -(sr5 * ru) / (24 * su5)
            kk_cov = sr / 720 * (1 - sr5 / su5)

            cov = jnp.array(
                [
                    [ww_cov, wh_cov, wk_cov],
                    [wh_cov, hh_cov, hk_cov],
                    [wk_cov, hk_cov, kk_cov],
                ]
            )

            # now compute the values of (W_sr, H_sr, K_sr), which are to be tested
            # against the normal distribution N(mean, cov)
            w_sr = w_r - w_s
            r_bb_s = r * w_s - s * w_r
            h_sr = 1 / sr * (bh_r - bh_s - 0.5 * r_bb_s)
            k_sr = (
                1
                / sr2
                * (
                    bk_r
                    - bk_s
                    - 0.5 * (sr * bh_r - s * bh_su)
                    - 1 / 12 * (r - 2 * s) * r_bb_s
                )
            )

            # now compare the distributions using the ks test
            normalised = jnp.array([w_sr, h_sr, k_sr]) - mean
            _, pval = stats.kstest(normalised, stats.multivariate_normal(mean, cov).cdf)

            assert pval > 0.001


def test_reverse_time():
    key = jrandom.PRNGKey(5678)
    bm_key, sample_key = jrandom.split(key, 2)
    bm = diffrax.VirtualBrownianTree(
        t0=0, t1=5, tol=2**-5, shape=(), key=bm_key, levy_area="space-time"
    )

    ts = jrandom.uniform(sample_key, shape=(100,), minval=0, maxval=5)

    vec_eval = jax.vmap(lambda t_prev, t: bm.evaluate(t_prev, t))

    fwd_increments = vec_eval(ts[:-1], ts[1:])
    back_increments = vec_eval(ts[1:], ts[:-1])

    assert jtu.tree_map(
        lambda fwd, bck: jnp.allclose(fwd, -bck), fwd_increments, back_increments
    )
