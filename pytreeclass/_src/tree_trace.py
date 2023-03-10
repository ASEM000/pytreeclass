from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Any, Callable, NamedTuple, Sequence

import jax.tree_util as jtu
from jax._src.tree_util import _registry

"""Extending `jax.tree_util` to support `tree_viz` and `tree_indexer` functionality"""

# the code style is heavilty influenced by `jax.tree_util`
# https://github.com/google/jax/blob/main/jax/_src/tree_util.py


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
# (0, 0, 0)
# (0, 0, 1)


PyTree = Any

_trace_registry = {}


class _TraceRegistryEntry(NamedTuple):
    to_iter: Callable[..., Any]


class LeafTrace(NamedTuple):
    names: Sequence[str]  # name of the node in each level
    types: Sequence[type]  # type of the node in each level
    index: Sequence[int]  # index of the node in the tree in each level
    width: Sequence[int]  # number of children in each level
    metas: Sequence[Any]  # metadata for each level for a node


EmptyTrace = ((), (), (), (), ())


def register_pytree_node_trace(
    klass: type, trace_func: Callable[[Any], Sequence[LeafTrace]]
):
    if not isinstance(klass, type):
        raise TypeError(f"Expected `klass` to be a type, got {type(klass)}.")
    if klass in _trace_registry:
        raise ValueError(f"Node trace flatten function for {klass} already registered.")
    # register the node trace flatten function to the node trace registry
    _trace_registry[klass] = _TraceRegistryEntry(trace_func)


def flatten_one_trace_level(
    tree_trace: LeafTrace,
    tree: PyTree,
    *,
    is_leaf: Callable[[Any], bool] | None,
    depth: int | None,
):
    # addition to `is_leaf` condtion , `depth`` is also useful for `tree_viz` utils
    # However, can not be used for any function that works with `treedef` objects
    if (is_leaf is not None and is_leaf(tree)) or (depth is not None and depth < 1):
        yield LeafTrace(*tree_trace), tree
        return

    leaves_handler = _registry.get(type(tree))

    if leaves_handler:
        leaves, _ = leaves_handler.to_iter(tree)

        # if type(tree) in _trace_registry:
        # trace handler for the current tree
        # defaults to `_jaxable_trace_func`
        traces = (
            _trace_registry[type(tree)].to_iter(tree)
            if type(tree) in _trace_registry
            else _jaxable_trace_func(tree)
        )

    elif isinstance(tree, tuple) and hasattr(tree, "_fields"):
        # this conforms to the `jax` convention for namedtuples
        leaves = (getattr(tree, field) for field in tree._fields)
        traces = _namedtuple_trace_func(tree)

    elif tree is not None:
        yield (LeafTrace(*tree_trace), tree)
        return

    for trace, leaf in zip(traces, leaves):
        yield from flatten_one_trace_level(
            tree_trace=(
                (*tree_trace[0], trace[0]),  # names
                (*tree_trace[1], trace[1]),  # types
                (*tree_trace[2], trace[2]),  # index
                (*tree_trace[3], trace[3]),  # width
                (*tree_trace[4], trace[4]),  # metas
            ),
            tree=leaf,
            is_leaf=is_leaf,
            # `None` depth is max depth
            depth=(depth - 1) if depth is not None else None,
        )


def tree_leaves_with_trace(
    tree: PyTree, is_leaf: Callable[[Any], bool] | None = None, depth: int | None = None
) -> Sequence[LeafTrace]:
    """Similar to jax.tree_util.tree_leaves` but returns the `LeafTrace` objects too as well"""
    return list(flatten_one_trace_level(EmptyTrace, tree, is_leaf=is_leaf, depth=depth))


def tree_flatten_with_trace(
    tree: PyTree, is_leaf: Callable[[Any], bool] | None = None
) -> tuple[Sequence[tuple[LeafTrace, Any]], jtu.PyTreeDef]:
    """Similar to jax.tree_util.tree_flatten` but returns the `LeafTrace` objects too as well"""
    tree_def = jtu.tree_structure(tree, is_leaf=is_leaf)
    traces_leaves = tree_leaves_with_trace(tree, is_leaf=is_leaf)
    return traces_leaves, tree_def


def _sequence_trace_func(
    tree: Sequence,
) -> Sequence[Sequence[str, type, int, int, Any]]:
    names = (f"[{i}]" for i in range(len(tree)))
    types = map(type, tree)
    index = range(len(tree))
    width = (len(tree) for _ in range(len(tree)))
    metas = (() for _ in range(len(tree)))
    return [*zip(names, types, index, width, metas)]


def _dict_trace_func(tree: dict) -> Sequence[Sequence[str, type, int, int, Any]]:
    names = (f"['{k}']" for k in tree)
    types = (type(tree[key]) for key in tree)
    index = range(len(tree))
    width = (len(tree) for _ in range(len(tree)))
    metas = ({"repr": not k.startswith("_")} for k in tree)
    return [*zip(names, types, index, width, metas)]


def _namedtuple_trace_func(tree: Any) -> Sequence[Sequence[str, type, int, int, Any]]:
    names = (f"['{field}']" for field in tree._fields)
    types = (type(getattr(tree, field)) for field in tree._fields)
    index = range(len(tree))
    width = (len(tree) for _ in range(len(tree)))
    metas = (() for _ in tree._fields)
    return [*zip(names, types, index, width, metas)]


def _jaxable_trace_func(tree: Any) -> Sequence[Sequence[str, type, int, int, Any]]:
    # fallback trace function in case no trace function is registered for a given
    # class in the `trace` registry
    # get leaves from the `jax` registry
    leaves, _ = _registry.get(type(tree)).to_iter(tree)
    names = (f"leaf_{i}" for i in range(len(leaves)))
    types = map(type, leaves)
    index = range(len(leaves))
    width = (len(leaves) for _ in range(len(leaves)))
    metas = (() for _ in range(len(leaves)))
    return [*zip(names, types, index, width, metas)]


# register trace functions for common types
register_pytree_node_trace(tuple, _sequence_trace_func)
register_pytree_node_trace(list, _sequence_trace_func)
register_pytree_node_trace(dict, _dict_trace_func)
register_pytree_node_trace(OrderedDict, _dict_trace_func)
register_pytree_node_trace(defaultdict, _dict_trace_func)


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
