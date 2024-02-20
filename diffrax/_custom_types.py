import typing
from typing import Any, TYPE_CHECKING, Union

import equinox as eqx
import equinox.internal as eqxi
import jax.tree_util as jtu
import numpy as np
from jaxtyping import (
    AbstractDtype,
    Array,
    ArrayLike,
    Bool,
    Float,
    Int,
    PyTree,
    Shaped,
)


if TYPE_CHECKING:
    BoolScalarLike = Union[bool, Array, np.ndarray]
    FloatScalarLike = Union[float, Array, np.ndarray]
    IntScalarLike = Union[int, Array, np.ndarray]
elif getattr(typing, "GENERATING_DOCUMENTATION", False):
    # Skip the union with Array in docs.
    BoolScalarLike = bool
    FloatScalarLike = float
    IntScalarLike = int

    #
    # Because they appear in our docstrings, we also monkey-patch some non-Diffrax
    # types that have similar defined-in-one-place, exported-in-another behaviour.
    #

    jtu.Partial.__module__ = "jax.tree_util"

else:
    BoolScalarLike = Bool[ArrayLike, ""]
    FloatScalarLike = Float[ArrayLike, ""]
    IntScalarLike = Int[ArrayLike, ""]


RealScalarLike = Union[FloatScalarLike, IntScalarLike]

Y = PyTree[Shaped[ArrayLike, "?*y"], "Y"]
VF = PyTree[Shaped[ArrayLike, "?*vf"], "VF"]
Control = PyTree[Shaped[ArrayLike, "?*control"], "C"]
Args = PyTree[Any]

DenseInfo = dict[str, PyTree[Array]]
DenseInfos = dict[str, PyTree[Shaped[Array, "times ..."]]]
BufferDenseInfos = dict[str, PyTree[eqxi.MaybeBuffer[Shaped[Array, "times ..."]]]]
sentinel: Any = eqxi.doc_repr(object(), "sentinel")


class BrownianIncrement(eqx.Module):
    dt: PyTree
    W: PyTree


class SpaceTimeLevyArea(BrownianIncrement):
    dt: PyTree
    W: PyTree
    H: PyTree


class SpaceTimeTimeLevyArea(SpaceTimeLevyArea):
    dt: PyTree
    W: PyTree
    H: PyTree
    K: PyTree


def levy_tree_transpose(
    tree_shape, tree: PyTree[BrownianIncrement]
) -> BrownianIncrement:
    """Helper that takes a PyTree of BrownianIncrement and transposes
    into an BrownianIncrement of PyTrees.

    **Arguments:**

    - `tree_shape`: Corresponds to `outer_treedef` in `jax.tree_transpose`.
    - `tree`: the PyTree of BrownianIncrement to transpose.

    **Returns:**

    An `BrownianIncrement` of PyTrees.
    """
    inner_tree = jtu.tree_leaves(
        tree, is_leaf=lambda x: isinstance(x, BrownianIncrement)
    )[0]
    inner_tree_shape = jtu.tree_structure(inner_tree)
    return jtu.tree_transpose(
        outer_treedef=jtu.tree_structure(tree_shape),
        inner_treedef=inner_tree_shape,
        pytree_to_transpose=tree,
    )


del Array, ArrayLike, PyTree, Bool, Int, Shaped, Float, AbstractDtype
