from __future__ import annotations

import sys
from dataclasses import field

import jax
import jax.numpy as jnp
import numpy as np
from jax.tree_util import tree_reduce


def static_field(**kwargs):
    """ignore from pytree computations"""
    return field(**{**kwargs, **{"metadata": {"static": True}}})


def is_treeclass(model):
    return hasattr(model, "tree_fields")


def is_treeclass_leaf_bool(node):
    if isinstance(node, jnp.ndarray):
        return node.dtype == "bool"
    else:
        return isinstance(node, bool)


def is_treeclass_leaf(model):

    if is_treeclass(model):
        fields = model.__dataclass_fields__.values()

        return is_treeclass(model) and not any(
            [is_treeclass(model.__dict__[field.name]) for field in fields]
        )
    else:
        return False


def is_treeclass_equal(lhs, rhs):
    """assert all leaves are same . use jnp.all on jnp.arrays"""

    def assert_node(lhs_node, rhs_node):
        if isinstance(lhs_node, jnp.ndarray):
            return jnp.all(lhs_node == rhs_node)
        else:
            return lhs_node == rhs_node

    lhs_leaves = jax.tree_util.tree_leaves(lhs)
    rhs_leaves = jax.tree_util.tree_leaves(rhs)

    for lhs_node, rhs_node in zip(lhs_leaves, rhs_leaves):
        if not assert_node(lhs_node, rhs_node):
            return False
    return True


def sequential_model_shape_eval(model, array):
    leaves = jax.tree_util.tree_leaves(model, is_treeclass_leaf)
    shape = [jax.eval_shape(lambda x: x, array)]
    for leave in leaves:
        shape += [jax.eval_shape(leave, shape[-1])]
    return shape


def node_count_and_size(node):
    """calculate number and size of `trainable` and `non-trainable` parameters"""

    if isinstance(node, (jnp.ndarray, np.ndarray)):
        # inexact(trainable) array
        if jnp.issubdtype(node, jnp.inexact):
            count = complex(int(jnp.array(node.shape).prod()), 0)
            size = complex(int(node.nbytes), 0)

        # exact paramter
        else:
            count = complex(0, int(jnp.array(node.shape).prod()))
            size = complex(0, int(node.nbytes))

    # inexact non-array
    elif isinstance(node, (float, complex)):
        count = complex(1, 0)
        size = complex(sys.getsizeof(node), 0)

    # exact non-array
    elif isinstance(node, int):
        count = complex(0, 1)
        size = complex(sys.getsizeof(node), 0)

    # others
    else:
        count = complex(0, 0)
        size = complex(0, sys.getsizeof(node))

    return (count, size)


def reduce_count_and_size(leaf):
    """reduce params count and params size of a tree of leaves"""

    def reduce_func(acc, node):
        lhs_count, lhs_size = acc
        rhs_count, rhs_size = node_count_and_size(node)
        return (lhs_count + rhs_count, lhs_size + rhs_size)

    return tree_reduce(reduce_func, leaf, (complex(0, 0), complex(0, 0)))


def freeze_nodes(model):
    """inplace freezing"""
    if is_treeclass(model):
        object.__setattr__(model, "__frozen_treeclass__", True)
        for kw, leaf in model.__dataclass_fields__.items():
            freeze_nodes(model.__dict__[kw])
    return model


def unfreeze_nodes(model):
    """inplace freezing"""
    if is_treeclass(model):
        object.__setattr__(model, "__frozen_treeclass__", False)
        for kw, leaf in model.__dataclass_fields__.items():
            unfreeze_nodes(model.__dict__[kw])
    return model
