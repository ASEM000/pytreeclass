# this script defines the tree_indexer class e.g. `.at` method for pytree

from __future__ import annotations

import dataclasses as dc
import functools as ft
from collections.abc import Callable
from copy import copy
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu
from jax.core import Tracer

from pytreeclass._src.tree_freeze import _set_dataclass_frozen

PyTree = Any
EllipsisType = type(Ellipsis)


def _is_leaf_bool(node):
    """assert if a dataclass leaf is boolean (for boolen indexing)"""
    if hasattr(node, "dtype"):
        return node.dtype == "bool"
    return isinstance(node, bool)


def _at_get(tree: PyTree, where: PyTree, is_leaf: Callable[[Any], bool]):
    def _lhs_get(lhs: Any, where: Any):
        """Get pytree node  value"""
        if not (_is_leaf_bool(where) or where is None):
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
        # fuse the boolean check here
        if not (_is_leaf_bool(where) or where is None):
            raise TypeError(f"All tree leaves must be boolean.Found {(where)}")

        if isinstance(lhs, (Tracer, jnp.ndarray)):
            if jnp.isscalar(set_value):
                return jnp.where(where, set_value, lhs)
            return set_value if jnp.all(where) else lhs

        return set_value if (where is True or where is None) else lhs

    if isinstance(set_value, type(tree)):
        # set_value leaf is set to tree leaf according to where leaf
        # for example lhs_tree.at[where].set(rhs_tree) will set rhs_tree leaves to lhs_tree leaves
        return jtu.tree_map(_lhs_set, set_value, tree, where, is_leaf=is_leaf)

    # set_value is broadcasted to tree leaves
    # for example tree.at[where].set(1) will set all tree leaves to 1
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
        if not (_is_leaf_bool(where) or where is None):
            raise TypeError(f"All tree leaves must be boolean.Found {(where)}")

        if isinstance(lhs, (Tracer, jnp.ndarray)):
            # check if the `set_value` is a valid type for `jnp.where` (i.e. can replace single array content)
            # otherwise , do not broadcast the `set_value` to the lhs
            # but instead, set the `set_value` to the whole lhs if all array elements are True
            try:
                # the function should have a scalar output or array output with the same shape as the input
                # unlike `set`, I think instead of evaluating the function and check for the output shape
                # its better to try using `jnp.where` and catch the error
                return jnp.where(where, func(lhs), lhs)
            except TypeError:
                return func(lhs) if jnp.all(where) else lhs

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
class PyTreeIndexer:
    tree: PyTree
    where: PyTree

    def get(self, *, is_leaf: Callable[[Any], bool] = None):
        return _at_get(self.tree, self.where, is_leaf=is_leaf)

    def set(self, set_value, *, is_leaf: Callable[[Any], bool] = None):
        return _at_set(copy(self.tree), self.where, set_value, is_leaf)

    def apply(self, func, *, is_leaf: Callable[[Any], bool] = None):
        return _at_apply(copy(self.tree), self.where, func, is_leaf)

    def reduce(self, func, *, is_leaf: Callable[[Any], bool] = None, initializer=0):
        return _at_reduce(self.tree, self.where, func, is_leaf, initializer)

    def __repr__(self) -> str:
        return f"where={self.where!r}"

    def __str__(self) -> str:
        return f"where={self.where}"


def _getter(item: Any, path: list[str]):
    """recursive getter"""
    # this function gets a certain attribute value based on a
    # sequence of strings.
    # for example _getter(item , ["a", "b", "c"]) is equivalent to item.a.b.c
    return (
        _getter(getattr(item, path[0]), path[1:])
        if len(path) > 1
        else getattr(item, path[0])
    )


def _setter(item: Any, path: list[str], value: Any):
    """recursive setter"""
    # this function sets a certain attribute value based on a
    # sequence of strings.
    # for example _setter(item , ["a", "b", "c"], value) is equivalent to item.a.b.c = value

    def _setter_getter(item, path: list[str]):
        return (
            _setter_getter(getattr(item, path[0]), path[1:])
            if len(path) > 1
            else (item, path[0])
        )

    parent, attr = _setter_getter(item, path)

    if not hasattr(parent, attr):
        raise AttributeError(f"{attr} is not a valid attribute of {parent}")
    setattr(parent, attr, value)
    return _setter(item, path[:-1], parent) if len(path) > 1 else item


@dc.dataclass(eq=False, frozen=True)
class StrIndexer:
    tree: PyTree
    where: str

    def get(self):
        # x.at["a"].get() returns x.a
        return _getter(self.tree, self.where.split("."))

    def set(self, set_value):
        # x.at["a"].set(value) returns a new tree with x.a = value
        # unlike [...].set with mask, this **does not preserve** the tree structure
        tree = _set_dataclass_frozen(copy(self.tree), frozen=False)
        where = self.where.split(".")
        tree = _setter(tree, where, set_value)
        tree = _set_dataclass_frozen(tree, frozen=True)
        return copy(tree)

    def apply(self, func):
        # x.at["a"].apply(func) returns a new tree with x.a = func(x.a)

        # unlike [...].apply with mask, this does not preserve the tree structure
        value = func(self.tree.at[self.where].get())
        return self.tree.at[self.where].set(value)

    def __call__(self, *a, **k):
        # x.at[method_name]() -> returns value and new_tree
        tree = _set_dataclass_frozen(copy(self.tree), frozen=False)
        method = getattr(tree, self.where)
        value = method(*a, **k)
        tree = _set_dataclass_frozen(tree, frozen=True)
        return value, tree

    def __repr__(self) -> str:
        return f"where={self.where!r}"

    def __str__(self) -> str:
        return f"where={self.where}"


def _str_nested_indexer(tree, where):
    class StrNestedIndexer(StrIndexer):
        def __getitem__(nested_self, nested_where: list[str]):
            # check if nested_where is a valid attribute of the current node before
            # proceeding to the next level and to give a more informative error message
            if hasattr(_getter(tree, nested_self.where.split(".")), nested_where):
                nested_where = nested_self.where + "." + nested_where
                return StrNestedIndexer(tree=tree, where=nested_where)

            msg = f"{nested_where} is not a valid attribute of {_getter(tree, nested_self.where)}"
            raise AttributeError(msg)

        def __getattr__(nested_self, name):
            # support nested `.at``
            # for example `.at[A].at[B]` represents model.A.B
            if name == "at":
                # pass the current tree and the current path to the next `.at`
                return StrNestedIndexer(tree=tree, where=nested_self.where)

            raise AttributeError(
                f"{name} is not a valid attribute of {nested_self}\n"
                f"Did you mean to use .at[{name!r}]?"
            )

    return StrNestedIndexer(tree=tree, where=where)


def _pytree_nested_indexer(tree, where):
    class PyTreeNestedIndexer(PyTreeIndexer):
        def __getitem__(nested_self, nested_where):
            # here the case is .at[cond1].at[cond2] <-> .at[ cond1 and cond2 ]
            nested_where = nested_self.where & nested_where
            return PyTreeNestedIndexer(tree=tree, where=nested_where)

        def __getattr__(nested_self, name):
            # support for nested `.at`
            # e.g. `tree.at[tree>0].at[tree == str ]
            # corrsponds to (tree>0 and tree == str`)
            if name == "at":
                # pass the current where condition to the next level
                return PyTreeNestedIndexer(tree=tree, where=nested_self.where)

            raise AttributeError(
                f"{name} is not a valid attribute of {nested_self}\n"
                f"Did you mean to use .at[{name!r}]?"
            )

    return PyTreeNestedIndexer(tree=tree, where=where)


def _at_indexer(tree):
    class AtIndexer:
        def __init__(self):
            self._immutable_dataclass = True

        @property
        def __immutable_dataclass__(self):
            return self._immutable_dataclass

        @__immutable_dataclass__.setter
        def __immutable_dataclass__(self, value):
            self._immutable_dataclass = value

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

    return AtIndexer()


class _TreeAtIndexer:
    """Base class for indexing by string or boolean pytree.

    Example:
        >>> class Test(treeAtIndexer):
        ...     def __init__(self,a):
        ...         self.a =
        >>> test = Test(a=1)

        >>> # indexing by string
        >>> test.at["a"].get()
        1

        >>> # indexing by boolean pytree
        >>> test.at[jtu.tree_map(lambda _: True)].get()
        Test(a=1)
    """

    @property
    def at(self):
        return _at_indexer(self)
