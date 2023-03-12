from __future__ import annotations

from collections import OrderedDict, defaultdict
from typing import Any, Callable, NamedTuple, Sequence

import jax.tree_util as jtu
from jax._src.tree_util import _registry

"""Extending `jax.tree_util` to support `tree_viz` and `tree_indexer` functionality"""

# the code style is heavilty influenced by `jax.tree_util`
# https://github.com/google/jax/blob/main/jax/_src/tree_util.py

PyTree = Any

_trace_registry = {}


class _TraceRegistryEntry(NamedTuple):
    to_iter: Callable[..., Any]


TraceType = Any


def register_pytree_node_trace(
    klass: type,
    trace_func: Callable[[Any], list[tuple[str, Any, tuple[int, int], Any]]],
) -> None:
    """
    Args:
        klass: The class of the object to be traced.
        trace_func:
            A function that instance of type `klass` and returns a tuple of
            (name, type, index, metadata) for each leaf in the object.

    Example:
        >>> import jax
        >>> import pytreeclass as pytc

        >>> class UserList(list):
        ...     pass

        >>> def user_list_trace_func(tree:UserList):
        ...     # (1) define name for each leaf
        ...     names = (f"leaf{i}" for i in range(len(tree)))
        ...     # (2) define types for each leaf
        ...     types = (type(leaf) for leaf in tree)
        ...     # (3) define (index,children count) for each leaf
        ...     indices = ((i,len(tree)) for i in range(len(tree)))
        ...     # (4) define metadatas (if any) for each leaf
        ...     metadatas = (() for _ in range(len(tree)))
        ...     # return a list of tuples (name, type, index, metadata)
        ...     return [*zip(names, types, indices, metadatas)]

        >>> pytc.register_pytree_node_trace(UserList, user_list_trace_func)

    Raises:
        TypeError: if `klass` is not a type
        ValueError: if `klass` is already registered
    """
    if not isinstance(klass, type):
        msg = f"Expected `klass` to be a type, got {type(klass)}."
        raise TypeError(msg)
    if klass in _trace_registry:
        msg = f"Node trace flatten function for {klass} is already registered."
        raise ValueError(msg)
    # register the node trace flatten function to the node trace registry
    _trace_registry[klass] = _TraceRegistryEntry(trace_func)


def flatten_one_trace_level(
    tree_trace: TraceType,
    tree: PyTree,
    is_leaf: Callable[[Any], bool] | None,
    is_trace_leaf: Callable[[TraceType], bool] | None,
):
    # addition to `is_leaf` condtion , `depth`` is also useful for `tree_viz` utils
    # However, can not be used for any function that works with `treedef` objects
    if (is_leaf and is_leaf(tree)) or (is_trace_leaf and is_trace_leaf(tree_trace)):
        # wrap the trace tuple with a object
        yield tree_trace, tree
        return

    if type(tree) in _registry:
        # trace handler for the current tree
        leaves, _ = _registry[type(tree)].to_iter(tree)

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
        leaves = (getattr(tree, field) for field in tree._fields)  # type: ignore
        traces = _namedtuple_trace_func(tree)

    elif tree is not None:
        # wrap the trace tuple with a object
        yield tree_trace, tree
        return
    else:
        return

    for trace, leaf in zip(traces, leaves):
        leaf_trace = (
            (*tree_trace[0], trace[0]),  # names
            (*tree_trace[1], trace[1]),  # types
            (*tree_trace[2], trace[2]),  # indices
            (*tree_trace[3], trace[3]),  # metadatas
        )

        yield from flatten_one_trace_level(leaf_trace, leaf, is_leaf, is_trace_leaf)


def tree_leaves_with_trace(
    tree: PyTree,
    *,
    is_leaf: Callable[[Any], bool] | None = None,
    is_trace_leaf: Callable[[Any], bool] | None = None,
) -> Sequence[tuple[TraceType, Any]]:
    """Similar to jax.tree_util.tree_leaves` but returns  object, leaf pairs"""
    trace = ((type(tree).__name__,), (type(tree),), ((0, 1),), ())  # type: ignore
    return list(flatten_one_trace_level(trace, tree, is_leaf, is_trace_leaf))


def tree_flatten_with_trace(
    tree: PyTree, *, is_leaf: Callable[[Any], bool] | None = None
) -> tuple[Sequence[tuple[TraceType, Any]], jtu.PyTreeDef]:
    """Similar to jax.tree_util.tree_flatten` but returns the objects too as well"""
    treedef = jtu.tree_structure(tree, is_leaf=is_leaf)
    traces_leaves = tree_leaves_with_trace(tree, is_leaf=is_leaf)
    return traces_leaves, treedef


def _sequence_trace_func(
    tree: Sequence,
) -> list[tuple[Any, Any, tuple[int, int], Any]]:
    names = (f"[{i}]" for i in range(len(tree)))
    types = map(type, tree)
    indices = ((i, len(tree)) for i in range(len(tree)))
    metadatas = (() for _ in range(len(tree)))
    return [*zip(names, types, indices, metadatas)]


def _dict_trace_func(tree: dict) -> list[tuple[str, Any, tuple[int, int], Any]]:
    names = (f"['{k}']" for k in tree)
    types = (type(tree[key]) for key in tree)
    indices = ((i, len(tree)) for i in range(len(tree)))
    metadatas = ({"repr": not k.startswith("_")} for k in tree)
    return [*zip(names, types, indices, metadatas)]


def _namedtuple_trace_func(tree: Any) -> list[tuple[str, type, tuple[int, int], Any]]:
    names = (f"['{field}']" for field in tree._fields)
    types = (type(getattr(tree, field)) for field in tree._fields)
    indices = ((i, len(tree)) for i in range(len(tree)))
    metadatas = (() for _ in tree._fields)
    return [*zip(names, types, indices, metadatas)]


def _jaxable_trace_func(tree: Any) -> list[tuple[str, Any, tuple[int, int], Any]]:
    # fallback trace function in case no trace function is registered for a given
    # class in the `trace` registry
    # get leaves from the `jax` registry
    leaves, _ = _registry.get(type(tree)).to_iter(tree)  # type: ignore
    names = (f"leaf_{i}" for i in range(len(leaves)))
    types = map(type, leaves)
    indices = ((i, len(leaves)) for i in range(len(leaves)))
    metadatas = (() for _ in range(len(leaves)))
    return [*zip(names, types, indices, metadatas)]


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
    """Similar to `jax.tree_util.tree_map` but func accepts `trace` as first argument"""
    traces_leaves, treedef = tree_flatten_with_trace(tree, is_leaf=is_leaf)
    traces_leaves = list(zip(*traces_leaves))
    traces_leaves += [treedef.flatten_up_to(r) for r in rest]
    return treedef.unflatten(func(*xs) for xs in zip(*traces_leaves))
