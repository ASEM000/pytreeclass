from __future__ import annotations

import math
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

    lhs_leaves = jax.tree_leaves(lhs)
    rhs_leaves = jax.tree_leaves(rhs)

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


def node_class_name(node):
    return node.__class__.__name__


def node_size(node):
    """get size of `trainable` and `non-trainable` parameters"""

    # store trainable in real , nontrainable in imag
    if isinstance(node, (jnp.ndarray, np.ndarray)):

        if jnp.issubdtype(node, jnp.inexact):
            return complex(int(node.nbytes), 0)

        else:
            return complex(0, int(node.nbytes))

    elif isinstance(node, (float, complex)):
        return complex(sys.getsizeof(node), 0)

    else:
        return complex(0, sys.getsizeof(node))


def node_count(node):
    """count number of `trainable` and `non-trainable` parameters"""
    if isinstance(node, (jnp.ndarray, np.ndarray)):

        if jnp.issubdtype(node, jnp.inexact):
            return complex(int(jnp.array(node.shape).prod()), 0)

        else:
            return complex(0, int(jnp.array(node.shape).prod()))

    elif isinstance(node, (float, complex)):
        return complex(1, 0)

    elif isinstance(node, int):
        return complex(0, 1)

    else:
        return complex(0, 0)


def node_format(node):
    """format shape and dtype of jnp.array"""

    if isinstance(node, (jnp.ndarray, jax.ShapeDtypeStruct)):
        replace_tuple = (
            ("int", "i"),
            ("float", "f"),
            ("complex", "c"),
            ("(", "["),
            (")", "]"),
            (" ", ""),
        )

        formatted_string = f"{node.dtype}{jnp.shape(node)!r}"

        # trunk-ignore
        for lhs, rhs in replace_tuple:
            formatted_string = formatted_string.replace(lhs, rhs)
        return formatted_string

    else:
        return f"{node!r}"


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


def format_size(node_size, newline=False):
    """return formatted size from inexact(exact) complex number"""
    mark = "\n" if newline else ""
    order_kw = ["B", "KB", "MB", "GB"]

    # define order of magnitude
    real_size_order = int(math.log(node_size.real, 1024)) if node_size.real > 0 else 0
    imag_size_order = int(math.log(node_size.imag, 1024)) if node_size.imag > 0 else 0
    return (
        f"{(node_size.real)/(1024**real_size_order):.2f}{order_kw[real_size_order]}{mark}"
        f"({(node_size.imag)/(1024**imag_size_order):.2f}{order_kw[imag_size_order]})"
    )


def format_count(node_count, newline=False):
    mark = "\n" if newline else ""
    return f"{int(node_count.real):,}{mark}({int(node_count.imag):,})"


def summary_line(leaf):

    dynamic, static = leaf.tree_fields
    is_dynamic = not leaf.frozen

    if is_dynamic:
        name = f"{node_class_name(leaf)}"
        count, size = reduce_count_and_size(dynamic)
        return (name, count, size)

    else:
        name = f"{node_class_name(leaf)}\n(frozen)"
        count, size = reduce_count_and_size(static)
        return (name, count, size)


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
