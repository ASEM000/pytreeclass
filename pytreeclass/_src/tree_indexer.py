# this script defines the tree_indexer class e.g. `.at` method for pytree

from __future__ import annotations

import functools as ft
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, Sequence

import jax.numpy as jnp
import jax.tree_util as jtu
from jax.interpreters.partial_eval import DynamicJaxprTracer

from pytreeclass._src.dispatch import dispatch
from pytreeclass._src.tree_util import (
    _tree_immutate,
    _tree_mutate,
    is_treeclass_leaf_bool,
    tree_copy,
)

""" .at[...].get() """


def _at_get(tree, where, **kwargs):
    PyTree = type(tree)

    @dispatch(argnum="lhs")
    def _lhs_get(lhs: Any, where: Any, **kwargs):
        """Get pytree node  value

        Args:
            lhs (Any): Node value.
            where (Any): Conditional

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError(f"Get node type ={type(lhs)} is not implemented.")

    @_lhs_get.register(DynamicJaxprTracer)
    @_lhs_get.register(jnp.ndarray)
    def _(
        lhs: jnp.ndarray | DynamicJaxprTracer, where: Any, array_as_leaves: bool = True
    ):
        return (
            (lhs[jnp.where(where)])
            if array_as_leaves
            else (lhs if jnp.all(where) else None)
        )

    @_lhs_get.register(int)
    @_lhs_get.register(float)
    @_lhs_get.register(complex)
    @_lhs_get.register(tuple)
    @_lhs_get.register(list)
    @_lhs_get.register(str)
    def _(lhs: int | float | complex | tuple | list | str, where: Any, **kwargs):
        # set None to non-chosen non-array values
        return lhs if where else None

    @dispatch(argnum="where")
    def _where_get(tree: Any, where: Any, **kwargs):
        raise NotImplementedError(f"Get where type = {type(where)} is not implemented.")

    @_where_get.register(type(tree))
    def _(tree: PyTree, where: PyTree, is_leaf=None, **kwargs):
        lhs_leaves, lhs_treedef = jtu.tree_flatten(tree, is_leaf=is_leaf)
        where_leaves, where_treedef = jtu.tree_flatten(where, is_leaf=is_leaf)
        lhs_leaves = [
            _lhs_get(lhs=lhs_leaf, where=where_leaf, **kwargs)
            for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)
        ]

        return jtu.tree_unflatten(lhs_treedef, lhs_leaves)

    return _where_get(tree=tree, where=where, **kwargs)


""" .at[...].set() """


def _at_set(tree, where, set_value, **kwargs):
    # Dispatch table
    # ----------------------------------------------------------------#
    # lhs           | where(rhs)    | set_value     | operation       #
    # ----------------------------------------------------------------#
    # jnp.ndarray,  |               |               | set_value       #
    # JaxprTracer   | bool          | bool          | if jnp.all(where)
    #               |               |               | else set_value  #
    # ----------------------------------------------------------------#
    # jnp.ndarray,  |               |               | set_value       #
    # JaxprTracer   | bool          | bool          | if jnp.all(where)
    #               |               |               | else set_value  #
    # ----------------------------------------------------------------#
    #               |               |               |[array_as_leaves]#
    #               |               |               | jnp.where(where,#
    #               |               |int, float,    |   set_value, lhs)
    # jnp.ndarray   | bool          |jnp.ndarray    | [otherwise]     #
    # JaxprTracer   |               |complex,       | set_value       #
    #               |               |               | if jnp.all(where)
    #               |               |               | else set_value  #
    #               |               |               |                 #
    # ----------------------------------------------------------------#

    PyTree = type(tree)

    @dispatch(argnum="lhs")
    def _lhs_set(lhs: Any, where: Any, set_value: Any, **kwargs):
        """Set pytree node value.

        Args:
            lhs: Node value.
            where: Conditional.
            set_value: Set value of shape 1.

        Returns:
            Modified node value.
        """
        raise NotImplementedError(f"Set node type = {type(lhs)} is unknown.")

    @_lhs_set.register(DynamicJaxprTracer)
    @_lhs_set.register(jnp.ndarray)
    @dispatch(argnum="set_value")
    def _set_value_set(
        lhs: jnp.ndarray | DynamicJaxprTracer, where: Any, set_value: Any
    ):
        """Multi dispatched on lhs and where type"""
        # lhs is numeric node
        # set_value in not acceptable set_value type for numeric node
        # thus array_as_leaves is not an optional keyword argument
        # An example is setting an array to None
        # then the entire array is set to None if all array elements are
        # satisfied by the where condition
        return set_value if jnp.all(where) else lhs

    @_set_value_set.register(bool)
    def _(lhs: jnp.ndarray | DynamicJaxprTracer, where: Any, set_value: bool):
        # in python isinstance(True/False,int) is True
        # without this dispatch, it will be handled with the int dispatch
        return set_value if jnp.all(where) else lhs

    @_set_value_set.register(int)
    @_set_value_set.register(float)
    @_set_value_set.register(complex)
    @_set_value_set.register(jnp.ndarray)
    def _(
        lhs: jnp.ndarray | DynamicJaxprTracer,
        where: Any,
        set_value: int | float | complex | jnp.ndarray,
        array_as_leaves: bool = True,
    ):
        # lhs is numeric node
        # set_value in acceptable set_value type for a numeric node
        # For some reason python isinstance(True,int) is True ?
        return (
            jnp.where(where, set_value, lhs)
            if array_as_leaves
            else (set_value if jnp.all(where) else lhs)
        )

    @_lhs_set.register(int)
    @_lhs_set.register(float)
    @_lhs_set.register(complex)
    @_lhs_set.register(tuple)
    @_lhs_set.register(list)
    @_lhs_set.register(str)
    @_lhs_set.register(type(None))
    def _(
        lhs: int | float | complex | tuple | list | str | None,
        where: Any,
        set_value: Any,
        **kwargs,
    ):
        # where == None can be obtained by
        # is_leaf = lambda x : x is None
        return set_value if (where in [True, None]) else lhs

    @dispatch(argnum="where")
    def _where_set(tree: PyTree, where: PyTree, set_value: Any, is_leaf=None, **kwargs):
        raise NotImplementedError(f"Set where type = {type(where)} is not implemented.")

    @_where_set.register(type(tree))
    def _(tree: PyTree, where: PyTree, set_value: Any, is_leaf=None, **kwargs):
        lhs_leaves, lhs_treedef = jtu.tree_flatten(tree, is_leaf=is_leaf)
        where_leaves, rhs_treedef = jtu.tree_flatten(where, is_leaf=is_leaf)
        lhs_leaves = [
            _lhs_set(lhs=lhs_leaf, where=where_leaf, set_value=set_value, **kwargs)
            for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)
        ]

        return jtu.tree_unflatten(lhs_treedef, lhs_leaves)

    return _where_set(tree=tree, where=where, set_value=set_value, **kwargs)


""" .at[...].apply() """


def _at_apply(tree, where, func, **kwargs):

    PyTree = type(tree)

    @dispatch(argnum="lhs")
    def _lhs_apply(lhs: Any, where: bool, func: Callable[[Any], Any], **kwargs):
        """Set pytree node

        Args:
            lhs (Any): Node value.
            where (bool): Conditional
            func (Callable[[Any], Any]): Callable

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError(f"Apply node type= {type(lhs)} is not implemented.")

    @_lhs_apply.register(DynamicJaxprTracer)
    @_lhs_apply.register(jnp.ndarray)
    def _(
        lhs: jnp.ndarray | DynamicJaxprTracer,
        where: Any,
        func: Any,
        array_as_leaves: bool = True,
    ):
        # array_as_leaves defins whether the condition/where is applied to each array element
        return (
            jnp.where(where, func(lhs), lhs)
            if array_as_leaves
            else (func(lhs) if jnp.all(where) else lhs)
        )

    @_lhs_apply.register(int)
    @_lhs_apply.register(float)
    @_lhs_apply.register(complex)
    @_lhs_apply.register(tuple)
    @_lhs_apply.register(list)
    @_lhs_apply.register(str)
    @_lhs_apply.register(type(None))
    def _(
        lhs: int | float | complex | tuple | list | str | None,
        where: Any,
        func: Any,
        **kwargs,
    ):
        return func(lhs) if (where in [True, None]) else lhs

    @dispatch(argnum="where")
    def _where_apply(tree: PyTree, where: Any, func: Any, **kwargs):
        raise NotImplementedError(
            f"Apply where type = {type(where)} is not implemented."
        )

    @_where_apply.register(type(tree))
    def _(tree: PyTree, where: PyTree, func: Any, is_leaf: bool = None, **kwargs):

        lhs_leaves, lhs_treedef = jtu.tree_flatten(tree, is_leaf=is_leaf)
        where_leaves, rhs_treedef = jtu.tree_flatten(where, is_leaf=is_leaf)
        lhs_leaves = [
            _lhs_apply(lhs=lhs_leaf, where=where_leaf, func=func, **kwargs)
            for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)
        ]

        return jtu.tree_unflatten(lhs_treedef, lhs_leaves)

    return _where_apply(tree=tree, where=where, func=func, **kwargs)


""" .at[...].reduce() """


def _at_reduce(tree, where, func, **kwargs):
    @dispatch(argnum="where")
    def _where_reduce(tree, where, func, **kwargs):
        raise NotImplementedError(
            f"Reduce where type = {type(where)} is not implemented."
        )

    @_where_reduce.register(type(tree))
    def _(tree, where, func, initializer=0, **kwargs):
        return jtu.tree_reduce(func, tree.at[where].get(), initializer)

    return _where_reduce(tree=tree, where=where, func=func, **kwargs)


PyTree = Any


@dataclass(eq=False, frozen=True)
class _pyTreeIndexer:
    tree: PyTree
    where: PyTree

    def __post_init__(self):
        assert all(
            is_treeclass_leaf_bool(leaf) for leaf in jtu.tree_leaves(self.where)
        ), f"All tree leaves must be boolean.Found {jtu.tree_leaves(self.where)}"

    def get(self, **kwargs):
        return ft.partial(_at_get, where=self.where)(tree=self.tree, **kwargs)

    def set(self, set_value, **kwargs):
        return ft.partial(_at_set, where=self.where)(
            tree=self.tree, set_value=set_value, **kwargs
        )

    def apply(self, func, **kwargs):
        return ft.partial(_at_apply, where=self.where)(
            tree=self.tree, func=func, **kwargs
        )

    def reduce(self, func, **kwargs):
        return ft.partial(_at_reduce, where=self.where)(
            tree=self.tree, func=func, **kwargs
        )

    def __repr__(self):
        return f"where={self.where!r}"

    def __str__(self):
        return f"where={self.where!s}"


def _getter(item: Any, path: Sequence[str]):
    """ "recursive getter"""
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

    def apply(self, func, **kwargs):
        return self.tree.at[self.where].set(func(self.tree.at[self.where].get()))

    def __call__(self, *args, **kwargs):
        # x.at[method_name]() -> returns value and new_tree
        new_self = _tree_mutate(tree_copy(self.tree))
        method = getattr(new_self, self.where)
        value = method(*args, **kwargs)
        new_self = _tree_immutate(new_self)
        return value, new_self

    def __repr__(self):
        return f"where={self.where!r}"

    def __str__(self):
        return f"where={self.where!s}"


class _treeIndexer:
    @property
    def at(self):
        class _atIndexer:
            @dispatch(argnum=1)
            def __getitem__(_, *args):
                raise NotImplementedError(
                    f"Indexing with type{tuple(type(arg) for arg in args)} is not implemented."
                )

            @__getitem__.register(type(self))
            def _(_, where):
                """indexing by boolean pytree"""

                class _pyTreeNestedIndexer(_pyTreeIndexer):
                    # subclass to preserve the tree state(i.e. self)
                    # during recursive calls
                    def __getitem__(nested_self, nested_where):
                        # here the case is .at[cond1].at[cond2] <-> .at[ cond1 and cond2 ]
                        return _pyTreeNestedIndexer(
                            tree=self,
                            where=(nested_self.where & nested_where),
                        )

                    def __getattr__(nested_self, name):
                        # support for nested `.at`
                        # e.g. `tree.at[tree>0].at[tree == str ]
                        # corrsponds to (tree>0 and tree == str`)
                        if name != "at":
                            raise AttributeError(
                                f"{name} is not a valid attribute of {nested_self}"
                            )
                        return _pyTreeNestedIndexer(tree=self, where=nested_self.where)

                return _pyTreeNestedIndexer(tree=self, where=where)

            @__getitem__.register(str)
            def _(_, where):
                class _strNestedIndexer(_strIndexer):
                    # subclass to preserve the tree state(i.e. self)
                    # during recursive calls
                    def __getitem__(nested_self, nested_where):
                        return _strNestedIndexer(
                            tree=self, where=nested_self.where + "." + nested_where
                        )

                    def __getattr__(nested_self, name):
                        # support nested `.at``
                        # for example `.at[A].at[B]` represents model.A.B
                        if name != "at":
                            raise AttributeError(
                                f"{name} is not a valid attribute of {nested_self}"
                            )
                        return _strNestedIndexer(tree=self, where=nested_self.where)

                return _strNestedIndexer(tree=self, where=where)

            @__getitem__.register(type(Ellipsis))
            def _(_, where):
                """Ellipsis as an alias for all elements"""
                # model.at[model == model ] <==> model.at[...]
                return self.at[self == self]

        return _atIndexer()
