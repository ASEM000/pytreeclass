from __future__ import annotations

import functools as ft
from collections import OrderedDict, defaultdict
from typing import Any, Callable, Iterable, NamedTuple, Sequence

import jax.tree_util as jtu
from jax._src.tree_util import _registry

"""Extending `jax.tree_util` to support `tree_viz` and `tree_indexer` functionality"""

# the code style is heavilty influenced by `jax.tree_util`
# https://github.com/google/jax/blob/main/jax/_src/tree_util.py

PyTree = Any
TraceType = Any

_trace_registry: dict[type, _TraceRegistryEntry] = dict()


class _TraceRegistryEntry(NamedTuple):
    to_iter: Callable[..., Any]


def _validate_trace_func(
    trace_func: Callable[[Any], TraceType]
) -> Callable[[Any], TraceType]:
    # validate the trace function to make sure it returns the correct format
    # validation is only performed once
    @ft.wraps(trace_func)
    def wrapper(tree):
        if wrapper.is_validated:
            return trace_func(tree)

        for trace in (traces := trace_func(tree)):
            # check if the trace has the correct format
            if not isinstance(trace, (list, tuple)):
                msg = f"Trace return type is not defined properly for "
                msg += f"class=`{type(tree).__name__}`."
                msg += f"Expected a list or tuple, got {trace}"
                raise TypeError(msg)

            if len(trace) != 4:
                msg = f"Trace length is not defined properly for "
                msg += f"class=`{type(tree).__name__}`."
                msg += "Expected 4 entries, in the order of"
                msg += f"(name, type, index, metadata), got {trace}"
                raise ValueError(msg)

            if not isinstance(trace[0], str):
                msg = f"Trace name entry is not defined properly for "
                msg += f"class=`{type(tree).__name__}`."
                msg += f" Expected a string, got {trace[0]}"
                raise TypeError(msg)

            if not isinstance(trace[1], type):
                msg = f"Trace type entry is not defined properly for "
                msg += f"class=`{type(tree).__name__}`."
                msg += f" Expected a type, got {trace[1]}"
                raise TypeError(msg)

            if not isinstance(trace[2], int):
                msg = f"Trace index entry is not defined properly for "
                msg += f"class=`{type(tree).__name__}`."
                msg += f" Expected an integer, got {trace[2]}"
                raise TypeError(msg)

        wrapper.is_validated = True
        return traces

    wrapper.is_validated = False
    return wrapper


def register_pytree_node_trace(
    klass: type,
    trace_func: Callable[[Any], list[tuple[str, Any, tuple[int, int], Any]]],
) -> None:
    """
    Args:
        klass: The class of the object to be traced.
        trace_func:
            A function that takes an instance of type `klass` and defines the flatten rule
            for the object (name, type, index, metadata) for each leaf in the object.

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
        ...     # (3) index for each leaf in the level
        ...     indices = range(len(tree))
        ...     # (4) define metadatas (if any) for each leaf
        ...     metadatas = (() for _ in range(len(tree)))
        ...     # return a list of tuples (name, type, index, metadata)
        ...     return [*zip(names, types, indices, metadatas)]
        >>> pytc.register_pytree_node_trace(UserList, user_list_trace_func)

    Note:
        The `trace_func` should return a list of tuples in the order of
        (name, type, index, metadata) for each leaf in the object.
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
        yield tree_trace, tree
        return

    for rhs_trace, leaf in zip(traces, leaves):
        leaf_trace = (
            (*tree_trace[0], rhs_trace[0]),  # names
            (*tree_trace[1], rhs_trace[1]),  # types
            (*tree_trace[2], rhs_trace[2]),  # indices
            (*tree_trace[3], rhs_trace[3]),  # metadatas
        )
        yield from flatten_one_trace_level(leaf_trace, leaf, is_leaf, is_trace_leaf)


def tree_leaves_with_trace(
    tree: PyTree,
    *,
    is_leaf: Callable[[Any], bool] | None = None,
    is_trace_leaf: Callable[[Any], bool] | None = None,
) -> Sequence[tuple[TraceType, Any]]:
    r"""Similar to jax.tree_util.tree_leaves` but returns  object, leaf pairs

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
        >>> # print(pytc.tree_repr(traces))
        >>> # (
        >>> #   (
        >>> #     ('list', '[0]'),                                                      # -> name path of leaf = 1
        >>> #     (<class 'list'>, <class 'int'>),                                      # -> type path of leaf = 1
        >>> #     (0, 0),                                                               # -> index path of leaf = 1
        >>> #     ({id:4951960512}, {id:4380344560})                                    # -> metadata path of leaf = 1
        >>> #   ),
        >>> #   (
        >>> #     ('list', '[1]', '[0]'),                                               # -> name path of leaf = 2
        >>> #     (<class 'list'>, <class 'list'>, <class 'int'>),                      # -> type path of leaf = 2
        >>> #     (0, 1, 0),                                                            # -> index path of leaf = 2
        >>> #     ({id:4951960512}, {id:4951876032}, {id:4380344592})                   # -> metadata path of leaf = 2
        >>> #   ),
        >>> #   (
        >>> #     ('list', '[1]', '[1]', '[0]'),                                        # -> name path of leaf = 3
        >>> #     (<class 'list'>, <class 'list'>, <class 'list'>, <class 'int'>),      # -> type path of leaf = 3
        >>> #     (0, 1, 1, 0),                                                         # -> index path of leaf = 3
        >>> #     ({id:4951960512}, {id:4951876032}, {id:4950290624}, {id:4380344624})  # -> metadata path of leaf = 3
        >>> #   )
        >>> # )

    Note:
        `metadata` path can hold any information about the object. PyTreeClass stores the object id
        for the common data structures like `list`, `tuple`, `dict`, `set`, `namedtuple`, and `treeclass` wrapped
        classes.
    """
    trace = ((type(tree).__name__,), (type(tree),), (0,), (dict(id=id(tree)),))  # type: ignore
    return list(flatten_one_trace_level(trace, tree, is_leaf, is_trace_leaf))


def tree_flatten_with_trace(
    tree: PyTree, *, is_leaf: Callable[[Any], bool] | None = None
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


def _sequence_trace_func(tree: Sequence) -> list[TraceType]:
    names = (f"[{i}]" for i in range(len(tree)))
    types = map(type, tree)
    indices = range(len(tree))
    metadatas = (dict(id=id(leaf)) for leaf in tree)
    return [*zip(names, types, indices, metadatas)]


def _dict_trace_func(tree: dict) -> list[TraceType]:
    names = (f"['{k}']" for k in tree)
    types = (type(tree[key]) for key in tree)
    indices = range(len(tree))
    metadatas = (dict(repr=not k.startswith("_"), id=id(tree[k])) for k in tree)
    return [*zip(names, types, indices, metadatas)]


def _namedtuple_trace_func(tree: Any) -> list[TraceType]:
    names = (f"['{field}']" for field in tree._fields)
    types = (type(getattr(tree, field)) for field in tree._fields)
    indices = range(len(tree))
    metadatas = (dict(id=id(getattr(tree, field))) for field in tree._fields)
    return [*zip(names, types, indices, metadatas)]


def _jaxable_trace_func(tree: Any) -> list[TraceType]:
    # fallback trace function in case no trace function is registered for a given
    # class in the `trace` registry
    # get leaves from the `jax` registry
    leaves, _ = _registry.get(type(tree)).to_iter(tree)  # type: ignore
    # TODO: fetch from `jax` key registry once min jax version>=0.4.6
    names = (f"leaf_{i}" for i in range(len(leaves)))
    types = map(type, leaves)
    indices = range(len(leaves))
    metadatas = (dict(id=id(leaf)) for leaf in leaves)
    return [*zip(names, types, indices, metadatas)]


# register trace functions for common types
_trace_registry[tuple] = _TraceRegistryEntry(_sequence_trace_func)
_trace_registry[list] = _TraceRegistryEntry(_sequence_trace_func)
_trace_registry[dict] = _TraceRegistryEntry(_dict_trace_func)
_trace_registry[OrderedDict] = _TraceRegistryEntry(_dict_trace_func)
_trace_registry[defaultdict] = _TraceRegistryEntry(_dict_trace_func)


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

        >>> # value tree:
        >>> #        tree
        >>> #          |
        >>> #    ---------------
        >>> #    |      |      |
        >>> #    |      |      |
        >>> #  ------   4   -------
        >>> #  |    |       |     |
        >>> #  1    2       5     6

        >>> # named tree:
        >>> #        `dict`
        >>> #          |
        >>> #    ---------------
        >>> #    |      |      |
        >>> #   'a'    'b'    'c'
        >>> #    |             |
        >>> #  ------       -------
        >>> #  |    |       |     |
        >>> #`[0]``[1]`   `[0]` `[1]`

        >>> # type tree:
        >>> #        dict
        >>> #          |
        >>> #    ---------------
        >>> #    |      |      |
        >>> #   list   int    list
        >>> #    |             |
        >>> #  ------       -------
        >>> #  |    |       |     |
        >>> # int  int     int   int

        >>> # index tree:
        >>> #          0
        >>> #          |
        >>> #    ---------------
        >>> #    |      |      |
        >>> #    0      1      2
        >>> #    |             |
        >>> #  ------       -------
        >>> #  |    |       |     |
        >>> #  0    1       0     1


        >>> def map_func(trace, leaf):
        ...    names, _, __, ___ = trace
        ...    if "['a']" in names:
        ...        return leaf + 100
        ...    return leaf
        >>> pytc.tree_map_with_trace(map_func, tree)
        {'a': [101, 102], 'b': 4, 'c': [5, 6]}

        >>> def map_func(trace, leaf):
        ...    _, types, __, ___ = trace
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
