from __future__ import annotations

import dataclasses as dc
from typing import Any, Callable, Hashable, Sequence, Tuple, TypeVar

import jax.tree_util as jtu
from jax._src.tree_util import _registry, _registry_with_keypaths
from jax.util import unzip2

"""Extending `jax.tree_util` to support `tree_viz` and `tree_indexer` functionality"""

# the code style is heavilty influenced by `jax.tree_util`
# https://github.com/google/jax/blob/main/jax/_src/tree_util.py

PyTree = Any

KeyEntry = TypeVar("KeyEntry", bound=Hashable)
TypeEntry = TypeVar("TypeEntry", bound=type)
TraceEntry = Tuple[KeyEntry, TypeEntry]
KeyPath = Tuple[KeyEntry, ...]
TypePath = Tuple[TypeEntry, ...]
TraceType = Tuple[KeyPath, TypePath]


@dc.dataclass(frozen=True)
class NamedSequenceKey:
    idx: int
    key: Hashable

    def __str__(self):
        return f".{self.key}"


def flatten_one_trace_level(
    trace: TraceType,
    tree: PyTree,
    is_leaf: Callable[[Any], bool] | None,
    is_trace_leaf: Callable[[TraceType], bool] | None,
):
    # similar to jax corresponding key path API but adds `is_trace_leaf` predicate and type path
    if (is_leaf and is_leaf(tree)) or (is_trace_leaf and is_trace_leaf(trace)):
        # is_leaf is a predicate function that determines whether a value is a leaf
        # is_trace_leaf is a predicate function that determines whether a trace is a leaf
        yield trace, tree
        return

    if type(tree) in _registry_with_keypaths:
        keys_leaves, _ = _registry_with_keypaths[type(tree)].flatten_with_keys(tree)
        keys, leaves = unzip2(keys_leaves)

    elif isinstance(tree, tuple) and hasattr(tree, "_fields"):
        # this conforms to the `jax` convention for namedtuples
        leaves = (getattr(tree, field) for field in tree._fields)  # type: ignore
        # use `NamedSequenceKey` to index by name and index unlike `jax` handler
        keys = tuple(NamedSequenceKey(idx, key) for idx, key in enumerate(tree._fields))  # type: ignore

    elif type(tree) in _registry:
        # no named handler for this type in key path
        leaves, _ = _registry[type(tree)].to_iter(tree)
        keys = tuple(jtu.GetAttrKey(f"leaf_{i}") for i, _ in enumerate(leaves))

    else:
        yield trace, tree
        return

    for key, leaf in zip(keys, leaves):
        yield from flatten_one_trace_level(
            ((*trace[0], key), (*trace[1], type(leaf))),
            leaf,
            is_leaf,
            is_trace_leaf,
        )


def tree_leaves_with_trace(
    tree: PyTree,
    *,
    is_leaf: Callable[[Any], bool] | None = None,
    is_trace_leaf: Callable[[TraceEntry], bool] | None = None,
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
    return list(flatten_one_trace_level(((), ()), tree, is_leaf, is_trace_leaf))


def tree_flatten_with_trace(
    tree: PyTree,
    *,
    is_leaf: Callable[[Any], bool] | None = None,
) -> tuple[Sequence[tuple[TraceType, Any]], jtu.PyTreeDef]:
    """Similar to jax.tree_util.tree_flatten` but returns key path, type path pairs.

    Args:
        tree: The tree to be flattened.
        is_leaf: A predicate function that determines whether a value is a leaf.

    Returns:
        A pair (leaves, treedef) where leaves is a list of (trace, leaf) pairs and
        treedef is a PyTreeDef object that can be used to reconstruct the tree.
    """
    treedef = jtu.tree_structure(tree, is_leaf=is_leaf)
    paths_leaves = tree_leaves_with_trace(tree, is_leaf=is_leaf)
    return paths_leaves, treedef


def tree_map_with_trace(
    func: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    r"""
    Similar to `jax.tree_util.tree_map_with_path` that accept a function that takes a two-item tuple
    for key path and type path.

    Args:
        func: A function that takes a trace and a leaf and returns a new leaf.
        tree: The tree to be mapped over.
        rest: Additional trees to be mapped over.
        is_leaf: A predicate function that determines whether a value is a leaf.

    Returns:
        A new tree with the same structure as tree.

    Example:
        >>> import jax.tree_util as jtu
        >>> import pytreeclass as pytc
        >>> tree = {"a": [1, 2], "b": 4, "c": [5, 6]}

        >>> # apply to "a" leaf
        >>> def map_func(trace, leaf):
        ...     names, _= trace
        ...     if jtu.DictKey("a") in names:
        ...         return leaf + 100
        ...     return leaf
        >>> pytc.tree_map_with_trace(map_func, tree)
        {'a': [101, 102], 'b': 4, 'c': [5, 6]}

        >>> # apply to any item with list in its type path
        >>> def map_func(trace, leaf):
        ...     _, types = trace
        ...     if list in types:
        ...         return leaf + 100
        ...     return leaf
        >>> pytc.tree_map_with_trace(map_func, tree)
        {'a': [101, 102], 'b': 4, 'c': [105, 106]}
    """
    paths_leaves, treedef = tree_flatten_with_trace(tree, is_leaf=is_leaf)
    paths_leaves = list(zip(*paths_leaves))
    paths_leaves += [treedef.flatten_up_to(r) for r in rest]
    return treedef.unflatten(func(*xs) for xs in zip(*paths_leaves))
