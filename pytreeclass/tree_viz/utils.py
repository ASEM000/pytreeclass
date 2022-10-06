from __future__ import annotations

import sys
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import pytreeclass as pytc
from pytreeclass._src.tree_base import _tree_structure


def _sequential_tree_shape_eval(tree, array):
    """Evaluate shape propagation of assumed sequential modules"""
    dyanmic, static = _tree_structure(tree)

    # all dynamic/static leaves
    all_leaves = (*dyanmic.values(), *static.values())
    leaves = [leaf for leaf in all_leaves if pytc.is_treeclass(leaf)]

    shape = [jax.eval_shape(lambda x: x, array)]
    for leave in leaves:
        shape += [jax.eval_shape(leave, shape[-1])]
    return shape


def _reduce_count_and_size(leaf):
    """reduce params count and params size of a tree of leaves to be used in `tree_summary`"""

    def reduce_func(acc, node):
        lhs_count, lhs_size = acc
        rhs_count, rhs_size = _node_count_and_size(node)
        return (lhs_count + rhs_count, lhs_size + rhs_size)

    return jtu.tree_reduce(reduce_func, leaf, (complex(0, 0), complex(0, 0)))


def _node_count_and_size(node: Any) -> tuple[complex, complex]:
    """Calculate number and size of `trainable` and `non-trainable` parameters

    Returns:
        complex: Complex number of (inexact, exact) parameters for count/size
    """

    if isinstance(node, jnp.ndarray):

        if jnp.issubdtype(node, jnp.inexact):
            # inexact(trainable) array
            count = complex(int(jnp.array(node.shape).prod()), 0)
            size = complex(int(node.nbytes), 0)
        else:
            # non in-exact paramter
            count = complex(0, int(jnp.array(node.shape).prod()))
            size = complex(0, int(node.nbytes))

    elif isinstance(node, (float, complex)):
        # inexact non-array (array_like)
        count = complex(1, 0)
        size = complex(sys.getsizeof(node), 0)

    elif isinstance(node, int):
        # exact non-array
        count = complex(0, 1)
        size = complex(0, sys.getsizeof(node))
        return count, size
    else:
        count = complex(0, 0)
        size = complex(0, 0)

    return count, size
