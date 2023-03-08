from __future__ import annotations

from typing import Any, Callable, NamedTuple, Sequence

import jax.tree_util as jtu
from jax._src.tree_util import _registry

"""Extending `jax.tree_util` to support `tree_viz` and `tree_indexer` functionality"""

# the code style is heavilty influenced by `jax.tree_util`
# https://github.com/google/jax/blob/main/jax/_src/tree_util.py


# This code extends the `jax.tree_util` module to support `tree_viz` and `tree_indexer` functionality.
# It defines a `LeafTrace` namedtuple that identifies a node in a tree and provides the (1) name path,
# (2) type path, (3) index path, and a (4) hidden flag for each level of the tree. The module also registers node
# trace flatten functions for different Python data types using the `_TraceRegistryEntry` namedtuple.
# It provides functions such as `tree_leaves_with_trace`, `tree_flatten_with_trace`, and `tree_map_with_trace`
# that are similar to their `jax.tree_util` counterparts but also include the `LeafTrace` objects.
# moreover, the tracing can provide extra functionality to relate between differnt nodes in the tree through
# the `LeafTrace` objects.

# ** Example usage **
# >>> tree = {"a": {"b": ([1, 2])}}

# >>> for trace,leaf in tree_leaves_with_trace(tree):
# ...    print(trace.names)
# ("['a']", "['b']", '[0]')
# ("['a']", "['b']", '[1]')

# >>> for trace,leaf in tree_leaves_with_trace(tree):
# ...    print(trace.types)
# (<class 'dict'>, <class 'list'>, <class 'int'>)
# (<class 'dict'>, <class 'list'>, <class 'int'>)

# >>> for trace,leaf in tree_leaves_with_trace(tree):
# ...    print(trace.index)
# (0, 0, 1)
# (0, 0, 0)
# ** note ** the index is reversed for each level of the tree so that the last child is always 0


PyTree = Any

_trace_registry = {}


class _TraceRegistryEntry(NamedTuple):
    to_iter: Callable[..., Any]


class LeafTrace(NamedTuple):
    names: tuple[str, ...]  # name path in each level of the tree
    types: tuple[type, ...]  # types of nodes in each level
    index: tuple[int, ...]  # reversed index of the node in each level. last child:=0
    hidden: tuple[bool, ...]  # wheather to omit this trace at this level


EmptyTrace = LeafTrace((), (), (), ())


def register_pytree_node_trace(
    klass: type, trace_func: Callable[[Any], Sequence[LeafTrace]]
):
    if not isinstance(klass, type):
        raise TypeError(f"Expected `klass` to be a type, got {type(klass)}")
    if klass in _trace_registry:
        raise ValueError(f"Node trace flatten function for {klass} already registered")
    # register the node trace flatten function to the node trace registry
    _trace_registry[klass] = _TraceRegistryEntry(trace_func)


def flatten_one_trace_level(
    trace: LeafTrace,
    tree: PyTree,
    *,
    is_leaf: Callable[[Any], bool] | None,
    depth: int | None,
):
    # addition to `is_leaf` condtion , `depth`` is also useful for `tree_viz` utils
    if (is_leaf is not None and is_leaf(tree)) or (depth is not None and depth < 1):
        yield trace, tree
        return

    leaves_handler = _registry.get(type(tree))

    if leaves_handler:
        leaves, _ = leaves_handler.to_iter(tree)

        if type(tree) in _trace_registry:
            # trace handler for the current tree
            # defaults to `_registered_jax_trace_func`
            traces = _trace_registry[type(tree)].to_iter(tree)
        else:
            # in case where a class is registered in `jax` but not in `trace` registry
            # then fill the trace with arbitrary `LeafTrace` objects
            traces = _registered_jax_trace_func(tree)

        for leaf_trace, leaf in zip(traces, leaves):
            names = (*trace.names, *leaf_trace.names)
            types = (*trace.types, *leaf_trace.types)
            index = (*trace.index, *leaf_trace.index)
            hidden = (*trace.hidden, *leaf_trace.hidden)
            yield from flatten_one_trace_level(
                trace=LeafTrace(names, types, index, hidden),
                tree=leaf,
                is_leaf=is_leaf,
                depth=(depth - 1) if depth is not None else None,
            )

    # TODO: add support for namedtuple following `jax.tree_util` code

    # following code is adapted from jax.tree_util
    elif tree is not None:
        yield (trace, tree)


def tree_leaves_with_trace(
    tree: PyTree, is_leaf: Callable[[Any], bool] | None = None, depth: int | None = None
):
    """Similar to jax.tree_util.tree_leaves` but returns the `LeafTrace` objects too as well"""
    return list(flatten_one_trace_level(EmptyTrace, tree, is_leaf=is_leaf, depth=depth))


def tree_flatten_with_trace(tree: PyTree, is_leaf: Callable[[Any], bool] | None = None):
    """Similar to jax.tree_util.tree_flatten` but returns the `LeafTrace` objects too as well"""
    tree_def = jtu.tree_structure(tree, is_leaf=is_leaf)
    leaves_trace = tree_leaves_with_trace(tree, is_leaf=is_leaf)
    return leaves_trace, tree_def


def _sequence_trace_func(tree: Sequence) -> Sequence[LeafTrace]:
    names = ((f"[{i}]",) for i in range(len(tree)))
    types = ((type(value),) for value in tree)
    index = ((i,) for i in reversed(range(len(tree))))
    hidden = ((False,) for _ in range(len(tree)))
    return [LeafTrace(*x) for x in zip(names, types, index, hidden)]


def _dict_trace_func(tree: dict) -> Sequence[LeafTrace]:
    names = ((f"['{k}']",) for k in tree)
    types = ((type(tree[key]),) for key in tree)
    index = ((i,) for i in reversed(range(len(tree))))
    hidden = ((k.startswith("_"),) for k in tree)  # omit keys starting with `_`
    return [LeafTrace(*x) for x in zip(names, types, index, hidden)]


def _registered_jax_trace_func(tree: Any) -> Sequence[LeafTrace]:
    # fallback trace function in case no trace function is registered for a given
    # class in the `trace` registry
    # get leaves from the `jax` registry
    leaves = _registry.get(type(tree)).to_iter(tree)
    traces = []
    for i, leaf in enumerate(leaves):
        names = (f"leaf_{i}",)
        types = (type(leaf),)
        index = (len(leaves) - i - 1,)
        hidden = (False,)
        traces += [LeafTrace(names, types, index, hidden)]
    return traces


register_pytree_node_trace(tuple, _sequence_trace_func)
register_pytree_node_trace(list, _sequence_trace_func)
register_pytree_node_trace(dict, _dict_trace_func)


def tree_map_with_trace(
    func: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    """Similar to `jax.tree_util.tree_map` but func accepts `LeafTrace` as first argument"""
    traces_leaves, treedef = tree_flatten_with_trace(tree, is_leaf)
    traces_leaves = list(zip(*traces_leaves))
    traces_leaves += [treedef.flatten_up_to(r) for r in rest]
    return treedef.unflatten(func(*xs) for xs in zip(*traces_leaves))
