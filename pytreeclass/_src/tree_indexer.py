# this script defines the tree_indexer class e.g. `.at` method for pytree

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Sequence

import jax.numpy as jnp
import jax.tree_util as jtu
from jax.core import Tracer

import pytreeclass as pytc
from pytreeclass._src.tree_util import _tree_immutate, _tree_mutate, tree_copy

PyTree = Any


def _at_get(tree: PyTree, where: PyTree, is_leaf: Callable[[Any], bool]):
    def _lhs_get(lhs: Any, where: Any):
        """Get pytree node  value"""
        if isinstance(lhs, (Tracer, jnp.ndarray)):
            return lhs[jnp.where(where)]

        else:
            return lhs if where else None

    if not isinstance(where, type(tree)):
        raise NotImplementedError(f"Get where type = {type(where)} is not implemented.")

    lhs_leaves, lhs_treedef = jtu.tree_flatten(tree, is_leaf=is_leaf)
    where_leaves = jtu.tree_leaves(where, is_leaf=is_leaf)
    lhs_leaves = [
        _lhs_get(lhs=lhs_leaf, where=where_leaf)
        for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)
    ]

    return jtu.tree_unflatten(lhs_treedef, lhs_leaves)


def _at_set(
    tree: PyTree,
    where: PyTree,
    set_value: bool | int | float | complex | jnp.ndarray,
    is_leaf: Callable[[Any], bool],
):
    def _lhs_set(lhs: Any, where: Any, set_value: Any):
        """Set pytree node value."""
        if isinstance(lhs, (Tracer, jnp.ndarray)):
            if isinstance(set_value, (bool)):
                return set_value if jnp.all(where) else lhs

            elif isinstance(set_value, (int, float, complex, jnp.ndarray)):
                return jnp.where(where, set_value, lhs)
        else:
            return set_value if (where is True or where is None) else lhs

    if not isinstance(where, type(tree)):
        raise NotImplementedError(f"Set where type = {type(where)} is not implemented.")

    lhs_leaves, lhs_treedef = jtu.tree_flatten(tree, is_leaf=is_leaf)
    where_leaves = jtu.tree_leaves(where, is_leaf=is_leaf)
    lhs_leaves = [
        _lhs_set(lhs=lhs_leaf, where=where_leaf, set_value=set_value)
        for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)
    ]

    return jtu.tree_unflatten(lhs_treedef, lhs_leaves)


def _at_apply(
    tree: PyTree,
    where: PyTree,
    func: Callable[[Any], Any],
    is_leaf: Callable[[Any], bool],
):
    def _lhs_apply(lhs: Any, where: bool, func: Callable[[Any], Any]):
        """Set pytree node"""

        if isinstance(lhs, (Tracer, jnp.ndarray)):
            return jnp.where(where, func(lhs), lhs)

        else:
            return func(lhs) if (where is True or where is None) else lhs

    if not isinstance(where, type(tree)):
        raise NotImplementedError(
            f"Apply where type = {type(where)} is not implemented."
        )

    lhs_leaves, lhs_treedef = jtu.tree_flatten(tree, is_leaf=is_leaf)
    where_leaves = jtu.tree_leaves(where, is_leaf=is_leaf)
    lhs_leaves = [
        _lhs_apply(lhs=lhs_leaf, where=where_leaf, func=func)
        for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)
    ]

    return jtu.tree_unflatten(lhs_treedef, lhs_leaves)


""" .at[...].reduce() """


def _at_reduce(
    tree: PyTree,
    where: PyTree,
    func: Callable[[Any], Any],
    is_leaf: Callable[[Any], bool],
    initializer: Any,
):

    if not isinstance(where, type(tree)):
        raise NotImplementedError(
            f"Reduce tree type = {type(tree)} is not implemented."
        )

    return jtu.tree_reduce(func, tree.at[where].get(is_leaf=is_leaf), initializer)


@dataclass(eq=False, frozen=True)
class _pyTreeIndexer:
    tree: PyTree
    where: PyTree

    def __post_init__(self):
        assert all(
            pytc.is_treeclass_leaf_bool(leaf) for leaf in jtu.tree_leaves(self.where)
        ), f"All tree leaves must be boolean.Found {jtu.tree_leaves(self.where)}"

    def get(self, is_leaf: Callable[[Any], bool] = None):
        return _at_get(self.tree, self.where, is_leaf=is_leaf)

    def set(self, set_value, is_leaf: Callable[[Any], bool] = None):
        return _at_set(self.tree, self.where, set_value, is_leaf)

    def apply(self, func, is_leaf: Callable[[Any], bool] = None):
        return _at_apply(self.tree, self.where, func, is_leaf)

    def reduce(self, func, is_leaf: Callable[[Any], bool] = None, initializer=0):
        return _at_reduce(self.tree, self.where, func, is_leaf, initializer)

    def __repr__(self):
        return f"where={self.where!r}"

    def __str__(self):
        return f"where={self.where!s}"


def _getter(item: Any, path: Sequence[str]):
    """recursive getter"""
    # this function gets a certain attribute value based on a
    # sequence of strings.
    # for example _getter(item , ["a", "b", "c"]) is equivalent to item.a.b.c
    return (
        _getter(getattr(item, path[0]), path[1:])
        if len(path) > 1
        else getattr(item, path[0])
    )


def _setter(item: Any, path: Sequence[str], value: Any):
    """recursive setter"""
    # this function sets a certain attribute value based on a
    # sequence of strings.
    # for example _setter(item , ["a", "b", "c"], value) is equivalent to item.a.b.c = value

    def _setter_getter(item, path):
        return (
            _setter_getter(getattr(item, path[0]), path[1:])
            if len(path) > 1
            else (item, path[0])
        )

    parent, attr = _setter_getter(item, path)

    if hasattr(parent, attr):
        object.__setattr__(parent, attr, value)
    else:
        raise AttributeError(f"{attr} is not a valid attribute of {parent}")

    return _setter(item, path[:-1], parent) if len(path) > 1 else item


@dataclass(eq=False, frozen=True)
class _strIndexer:
    tree: PyTree
    where: str

    def get(self):
        # x.at["a"].get() returns x.a
        return _getter(self.tree, self.where.split("."))

    def set(self, set_value):
        # x.at["a"].set(value) returns a new tree with x.a = value
        return _setter(tree_copy(self.tree), self.where.split("."), set_value)

    def apply(self, func):
        return self.tree.at[self.where].set(func(self.tree.at[self.where].get()))

    def __call__(self, *args, **kwargs):
        # x.at[method_name]() -> returns value and new_tree
        tree = _tree_mutate(tree_copy(self.tree))
        method = getattr(tree, self.where)
        value = method(*args, **kwargs)
        tree = _tree_immutate(tree)
        return value, tree

    def __repr__(self):
        return f"where={self.where!r}"

    def __str__(self):
        return f"where={self.where!s}"


def _str_nested_indexer(tree, where):
    class _strNestedIndexer(_strIndexer):
        def __getitem__(nested_self, nested_where):
            return _strNestedIndexer(
                tree=tree, where=nested_self.where + "." + nested_where
            )

        def __getattr__(nested_self, name):
            # support nested `.at``
            # for example `.at[A].at[B]` represents model.A.B
            if name == "at":
                return _strNestedIndexer(tree=tree, where=nested_self.where)

            raise AttributeError(f"{name} is not a valid attribute of {nested_self}")

    return _strNestedIndexer(tree=tree, where=where)


def _pytree_nested_indexer(tree, where):
    class _pyTreeNestedIndexer(_pyTreeIndexer):
        def __getitem__(nested_self, nested_where):
            # here the case is .at[cond1].at[cond2] <-> .at[ cond1 and cond2 ]
            return _pyTreeNestedIndexer(
                tree=tree,
                where=(nested_self.where & nested_where),
            )

        def __getattr__(nested_self, name):
            # support for nested `.at`
            # e.g. `tree.at[tree>0].at[tree == str ]
            # corrsponds to (tree>0 and tree == str`)
            if name == "at":
                return _pyTreeNestedIndexer(tree=tree, where=nested_self.where)

            raise AttributeError(f"{name} is not a valid attribute of {nested_self}")

    return _pyTreeNestedIndexer(tree=tree, where=where)


class _treeIndexer:
    @property
    def at(self):
        class _atIndexer:
            def __getitem__(_, where):

                if isinstance(where, str):
                    return _str_nested_indexer(tree=self, where=where)

                elif isinstance(where, type(self)):
                    # indexing by boolean pytree
                    return _pytree_nested_indexer(tree=self, where=where)

                elif isinstance(where, type(Ellipsis)):
                    # Ellipsis as an alias for all elements
                    # model.at[model == model ] <--> model.at[...]
                    return self.at[self == self]

                msg = f"Indexing with type{type(where)} is not implemented."
                raise NotImplementedError(msg)

        return _atIndexer()
