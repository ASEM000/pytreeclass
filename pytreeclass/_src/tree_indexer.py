# this script defines the tree_indexer class e.g. `.at` method for pytree

from __future__ import annotations

import copy
import dataclasses as dc
import functools as ft
from collections.abc import Callable
from typing import Any, Sequence

import jax.numpy as jnp
import jax.tree_util as jtu
from jax.core import Tracer

import pytreeclass._src.dataclass_util as dcu
from pytreeclass._src.dataclass_util import _set_dataclass_frozen

PyTree = Any
EllipsisType = type(Ellipsis)


def _at_get(tree: PyTree, where: PyTree, is_leaf: Callable[[Any], bool]):
    def _lhs_get(lhs: Any, where: Any):
        """Get pytree node  value"""
        if not (dcu.is_leaf_bool(where) or where is None):
            raise TypeError(f"All tree leaves must be boolean.Found {(where)}")

        if isinstance(lhs, (Tracer, jnp.ndarray)):
            return lhs[jnp.where(where)]
        return lhs if where else None

    return jtu.tree_map(_lhs_get, tree, where, is_leaf=is_leaf)


def _at_set(
    tree: PyTree,
    where: PyTree,
    set_value: bool | int | float | complex | jnp.ndarray,
    is_leaf: Callable[[Any], bool],
):
    def _lhs_set(set_value: Any, lhs: Any, where: Any):
        """Set pytree node value."""
        if not (dcu.is_leaf_bool(where) or where is None):
            raise TypeError(f"All tree leaves must be boolean.Found {(where)}")

        if isinstance(lhs, (Tracer, jnp.ndarray)):
            # check if the set_value is a valid type to be broadcasted to ndarray
            # otherwise, do not broadcast the set_value to the lhs
            # but instead, set the set_value to the whole lhs
            if jnp.isscalar(set_value):
                return jnp.where(where, set_value, lhs)
            return set_value

        return set_value if (where is True or where is None) else lhs

    if isinstance(set_value, type(tree)):
        # set_value leaf is set to tree leaf according to where leaf
        return jtu.tree_map(_lhs_set, set_value, tree, where, is_leaf=is_leaf)

    # set_value is broadcasted to tree leaves
    _lhs_set = ft.partial(_lhs_set, set_value)
    return jtu.tree_map(_lhs_set, tree, where, is_leaf=is_leaf)


def _at_apply(
    tree: PyTree,
    where: PyTree,
    func: Callable[[Any], Any],
    is_leaf: Callable[[Any], bool],
):
    def _lhs_apply(lhs: Any, where: bool):
        """Set pytree node"""
        if not (dcu.is_leaf_bool(where) or where is None):
            raise TypeError(f"All tree leaves must be boolean.Found {(where)}")

        if isinstance(lhs, (Tracer, jnp.ndarray)):
            return jnp.where(where, func(lhs), lhs)
        return func(lhs) if (where is True or where is None) else lhs

    return jtu.tree_map(_lhs_apply, tree, where, is_leaf=is_leaf)


""" .at[...].reduce() """


def _at_reduce(
    tree: PyTree,
    where: PyTree,
    func: Callable[[Any], Any],
    is_leaf: Callable[[Any], bool],
    initializer: Any,
):
    return jtu.tree_reduce(func, tree.at[where].get(is_leaf=is_leaf), initializer)


@dc.dataclass(eq=False, frozen=True)
class _pyTreeIndexer:
    tree: PyTree
    where: PyTree

    def get(self, is_leaf: Callable[[Any], bool] = None):
        return _at_get(self.tree, self.where, is_leaf=is_leaf)

    def set(self, set_value, is_leaf: Callable[[Any], bool] = None):
        return _at_set(self.tree, self.where, set_value, is_leaf)

    def apply(self, func, is_leaf: Callable[[Any], bool] = None):
        return _at_apply(self.tree, self.where, func, is_leaf)

    def reduce(self, func, is_leaf: Callable[[Any], bool] = None, initializer=0):
        return _at_reduce(self.tree, self.where, func, is_leaf, initializer)

    def __repr__(self) -> str:
        return f"where={self.where!r}"

    def __str__(self) -> str:
        return f"where={self.where}"


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


@dc.dataclass(eq=False, frozen=True)
class _strIndexer:
    tree: PyTree
    where: str

    def get(self):
        # x.at["a"].get() returns x.a
        return _getter(self.tree, self.where.split("."))

    def set(self, set_value):
        # x.at["a"].set(value) returns a new tree with x.a = value
        return _setter(copy.copy(self.tree), self.where.split("."), set_value)

    def apply(self, func):
        return self.tree.at[self.where].set(func(self.tree.at[self.where].get()))

    def __call__(self, *a, **k):
        # x.at[method_name]() -> returns value and new_tree
        tree = _set_dataclass_frozen(copy.copy(self.tree), frozen=False)
        method = getattr(tree, self.where)
        value = method(*a, **k)
        tree = _set_dataclass_frozen(tree, frozen=True)
        return value, tree

    def __repr__(self) -> str:
        return f"where={self.where!r}"

    def __str__(self) -> str:
        return f"where={self.where}"


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

            raise AttributeError(
                f"{name} is not a valid attribute of {nested_self}\n"
                f"Did you mean to use .at[{name!r}]?"
            )

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

            raise AttributeError(
                f"{name} is not a valid attribute of {nested_self}\n"
                f"Did you mean to use .at[{name!r}]?"
            )

    return _pyTreeNestedIndexer(tree=tree, where=where)


def _at_indexer(tree):
    class _atIndexer:
        def __getitem__(_, where):

            if isinstance(where, str):
                return _str_nested_indexer(tree=tree, where=where)

            elif isinstance(where, type(tree)):
                # indexing by boolean pytree
                return _pytree_nested_indexer(tree=tree, where=where)

            elif isinstance(where, EllipsisType):
                # Ellipsis as an alias for all elements
                # model.at[model == model ] <--> model.at[...]
                return tree.at[tree == tree]

            raise NotImplementedError(
                f"Indexing with {type(where)} is not implemented.\n"
                "Example of supported indexing:\n\n"
                "@pytc.treeclass\n"
                f"class {tree.__class__.__name__}:\n"
                "    ...\n\n"
                f">>> tree = {tree.__class__.__name__}(...)\n"
                "# indexing by boolean pytree\n"
                ">>> tree.at[tree > 0].get()\n\n"
                "# indexing by string\n"
                ">>> tree.at[`field_name`].get()"
            )

    return _atIndexer()
