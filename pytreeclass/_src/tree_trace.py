from __future__ import annotations

import functools as ft
from collections import OrderedDict, defaultdict
from typing import Any, Callable, NamedTuple, Sequence

import jax.tree_util as jtu
from jax._src.tree_util import _registry

"""Extending `jax.tree_util` to support `tree_viz` and `tree_indexer` functionality"""

# the code style is heavilty influenced by `jax.tree_util`
# https://github.com/google/jax/blob/main/jax/_src/tree_util.py

PyTree = Any
TraceType = Any


def _jaxable_trace_func(tree: Any) -> list[TraceType]:
    # fallback trace function in case no trace function is registered for a given
    # class in the `trace` registry
    # get leaves from the `jax` registry
    leaves, _ = _registry.get(type(tree)).to_iter(tree)  # type: ignore
    # TODO: fetch from `jax` key registry once min jax version>=0.4.6
    names = (f"leaf_{i}" for i in range(len(leaves)))
    types = map(type, leaves)
    indices = range(len(leaves))
    return [*zip(names, types, indices)]


class _TraceRegistryEntry(NamedTuple):
    to_iter: Callable[..., Any]


_trace_registry = defaultdict(lambda: _TraceRegistryEntry(_jaxable_trace_func))


def _validate_trace_func(
    trace_func: Callable[[Any], TraceType]
) -> Callable[[Any], TraceType]:
    # validate the trace function to make sure it returns the correct format
    # validation is only performed once
    @ft.wraps(trace_func)
    def wrapper(tree):
        if wrapper.has_run is True:
            return trace_func(tree)

        for trace in (traces := trace_func(tree)):
            # check if the trace has the correct format
            if not isinstance(trace, (list, tuple)):
                msg = "Trace return type is not defined properly for "
                msg += f"class=`{type(tree).__name__}`."
                msg += f"Expected a list or tuple, got {trace}"
                raise TypeError(msg)

            if len(trace) != 3:
                msg = "Trace length is not defined properly for "
                msg += f"class=`{type(tree).__name__}`."
                msg += "Expected 3 entries, in the order of"
                msg += f"(name, type, index), got {trace}"
                raise ValueError(msg)

            if not isinstance(trace[0], (str, type(None))):
                msg = "Trace name entry is not defined properly for "
                msg += f"class=`{type(tree).__name__}`."
                msg += f" Expected a string or `None`, got {trace[0]}"
                raise TypeError(msg)

            if not isinstance(trace[1], type):
                msg = "Trace type entry is not defined properly for "
                msg += f"class=`{type(tree).__name__}`."
                msg += f" Expected a type, got {trace[1]}"
                raise TypeError(msg)

            if not isinstance(trace[2], (int, type(None))):
                msg = "Trace index entry is not defined properly for "
                msg += f"class=`{type(tree).__name__}`."
                msg += f" Expected an integer or `None`, got {trace[2]}"
                raise TypeError(msg)

        wrapper.has_run = True
        return traces

    wrapper.has_run = False
    return wrapper


def register_pytree_node_trace(
    klass: type,
    trace_func: Callable[[Any], list[tuple[str, Any, tuple[int, int], Any]]],
) -> None:
    """
    Args:
        klass: The class of the object to be traced.
        trace_func:A function that takes an instance of type `klass` and defines the flatten rule
            for the object (name, type, index) for each leaf in the object.

    Example:
        >>> import jax
        >>> import pytreeclass as pytc
        >>> class UserList(list):
        ...     pass
        >>> def user_list_trace_func(tree:UserList):
        ...     # (1) define name for each leaf if exists, `None` otherwise
        ...     names = (None,) * len(tree)
        ...     # (2) define types for each leaf
        ...     types = (type(leaf) for leaf in tree)
        ...     # (3) index for each leaf in the level if exists, `None` otherwise
        ...     indices = range(len(tree))
        ...     # return a list of tuples (name, type, index)
        ...     return [*zip(names, types, indices)]
        >>> pytc.register_pytree_node_trace(UserList, user_list_trace_func)

    Note:
        The `trace_func` should return a list of tuples in the order of
        (name, type, index) for each leaf in the object.
        The format of the trace is validated on the first call and will raise
        `TypeError` or `ValueError` if the format is not correct.

    Raises:
        TypeError: if input is not a type
        ValueError: if `klass` is already registered
    """
    if not isinstance(klass, type):
        msg = f"Expected `klass` to be a type, got {type(klass)}."
        raise TypeError(msg)
    if klass in _trace_registry:
        msg = f"Node trace flatten function for {klass} is already registered."
        raise ValueError(msg)
    # register the node trace flatten function to the node trace registry
    _trace_registry[klass] = _TraceRegistryEntry(_validate_trace_func(trace_func))


def _sequence_trace_func(tree: Sequence) -> list[TraceType]:
    names = (None,) * len(tree)
    types = map(type, tree)
    indices = range(len(tree))
    return [*zip(names, types, indices)]


def _dict_trace_func(tree: dict) -> list[TraceType]:
    names = (f"['{k}']" for k in tree)
    types = (type(tree[key]) for key in tree)
    indices = (None,) * len(tree)
    return [*zip(names, types, indices)]


def _namedtuple_trace_func(tree: Any) -> list[TraceType]:
    names = tree._fields
    types = (type(getattr(tree, field)) for field in tree._fields)
    indices = range(len(tree))
    return [*zip(names, types, indices)]


# register trace functions for common types
register_pytree_node_trace(tuple, _sequence_trace_func)
register_pytree_node_trace(list, _sequence_trace_func)
register_pytree_node_trace(dict, _dict_trace_func)
register_pytree_node_trace(OrderedDict, _dict_trace_func)
register_pytree_node_trace(defaultdict, _dict_trace_func)


def flatten_one_trace_level(
    tree_trace: TraceType,
    tree: PyTree,
    is_leaf: Callable[[Any], bool] | None,
    is_trace_leaf: Callable[[TraceType], bool] | None,
):
    if (is_leaf and is_leaf(tree)) or (is_trace_leaf and is_trace_leaf(tree_trace)):
        # is_leaf is a predicate function that determines whether a value is a leaf
        # is_trace_leaf is a predicate function that determines whether a trace is a leaf
        yield tree_trace, tree
        return

    if type(tree) in _registry:
        # trace handler for the current tree
        leaves, _ = _registry[type(tree)].to_iter(tree)

        # trace handler for the current tree
        # defaults to `_jaxable_trace_func`
        traces = _trace_registry[type(tree)].to_iter(tree)

    elif isinstance(tree, tuple) and hasattr(tree, "_fields"):
        # this conforms to the `jax` convention for namedtuples
        leaves = (getattr(tree, field) for field in tree._fields)  # type: ignore
        traces = _namedtuple_trace_func(tree)

    else:
        yield tree_trace, tree
        return

    for rhs_trace, leaf in zip(traces, leaves):
        leaf_trace = (
            (*tree_trace[0], rhs_trace[0]),  # names
            (*tree_trace[1], rhs_trace[1]),  # types
            (*tree_trace[2], rhs_trace[2]),  # indices
        )
        yield from flatten_one_trace_level(leaf_trace, leaf, is_leaf, is_trace_leaf)


def tree_leaves_with_trace(
    tree: PyTree,
    *,
    is_leaf: Callable[[Any], bool] | None = None,
    is_trace_leaf: Callable[[Any], bool] | None = None,
) -> Sequence[tuple[TraceType, Any]]:
    r"""Similar to jax.tree_util.tree_leaves` but returns  object, leaf pairs.

    Args:
        tree: The tree to be flattened.
        is_leaf: A predicate function that determines whether a value is a leaf.
        is_trace_leaf: A predicate function that determines whether a trace is a leaf.

    Returns:
        A list of (trace, leaf) pairs.

    Example:
        >>> import pytreeclass as pytc
        >>> tree = [1, [2, [3]]]
        >>> traces, _ = zip(*pytc.tree_leaves_with_trace(tree))
    """
    return list(flatten_one_trace_level(((), (), ()), tree, is_leaf, is_trace_leaf))


def tree_flatten_with_trace(
    tree: PyTree,
    *,
    is_leaf: Callable[[Any], bool] | None = None,
) -> tuple[Sequence[tuple[TraceType, Any]], jtu.PyTreeDef]:
    """Similar to jax.tree_util.tree_flatten` but returns the objects too as well

    Args:
        tree: The tree to be flattened.
        is_leaf: A predicate function that determines whether a value is a leaf.

    Returns:
        A pair (leaves, treedef) where leaves is a list of (trace, leaf) pairs and
        treedef is a PyTreeDef object that can be used to reconstruct the tree.
    """
    treedef = jtu.tree_structure(tree, is_leaf=is_leaf)
    traces_leaves = tree_leaves_with_trace(tree, is_leaf=is_leaf)
    return traces_leaves, treedef


def tree_map_with_trace(
    func: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    r"""Similar to `jax.tree_util.tree_map` but func accepts `trace` as first argument

    Args:
        func: A function that takes a trace and a leaf and returns a new leaf.
        tree: The tree to be mapped over.
        rest: Additional trees to be mapped over.
        is_leaf: A predicate function that determines whether a value is a leaf.

    Returns:
        A new tree with the same structure as tree.

    Example:
        >>> import jax
        >>> import pytreeclass as pytc
        >>> tree = {"a": [1, 2], "b": 4, "c": [5, 6]}
        >>> # the tree above is visualized as:
        >>> # value tree:
        >>> #          |
        >>> #    ---------------
        >>> #    |      |      |
        >>> #    |      |      |
        >>> #  ------   4   -------
        >>> #  |    |       |     |
        >>> #  1    2       5     6

        >>> # named tree:
        >>> #          |
        >>> #    ---------------
        >>> #    |      |      |
        >>> # `['a']` `['b']``['c']`
        >>> #    |             |
        >>> #  ------       -------
        >>> #  |    |       |     |
        >>> #`[0]``[1]`   `[0]` `[1]`

        >>> # type tree:
        >>> #          |
        >>> #    ---------------
        >>> #    |      |      |
        >>> #  <list> <int> <list>
        >>> #    |             |
        >>> #  ------       -------
        >>> #  |    |       |     |
        >>> # <int><int>   <int><int>

        >>> # index tree:
        >>> #          |
        >>> #    ---------------
        >>> #    |      |      |
        >>> #    0      1      2
        >>> #    |             |
        >>> #  ------       -------
        >>> #  |    |       |     |
        >>> #  0    1       0     1

        >>> def map_func(trace, leaf):
        ...    names, _, __ = trace
        ...    if "['a']" in names:
        ...        return leaf + 100
        ...    return leaf
        >>> pytc.tree_map_with_trace(map_func, tree)
        {'a': [101, 102], 'b': 4, 'c': [5, 6]}

        >>> def map_func(trace, leaf):
        ...    _, types, __ = trace
        ...    if list in types:
        ...        return leaf + 100
        ...    return leaf
        >>> pytc.tree_map_with_trace(map_func, tree)
        {'a': [101, 102], 'b': 4, 'c': [105, 106]}
    """
    traces_leaves, treedef = tree_flatten_with_trace(tree, is_leaf=is_leaf)
    traces_leaves = list(zip(*traces_leaves))
    traces_leaves += [treedef.flatten_up_to(r) for r in rest]
    return treedef.unflatten(func(*xs) for xs in zip(*traces_leaves))
