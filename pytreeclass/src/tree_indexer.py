from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_leaves, tree_unflatten

from pytreeclass.src.decorator_util import dispatch
from pytreeclass.src.tree_util import is_treeclass_leaf_bool


def _node_setter(lhs: Any, where: bool, set_value):
    """Set pytree node value.

    Args:
        lhs: Node value.
        where: Conditional.
        set_value: Set value of shape 1.

    Returns:
        Modified node value.
    """
    if isinstance(lhs, jnp.ndarray):
        return jnp.where(where, set_value, lhs)
    else:
        return set_value if where else lhs


def _node_getter(lhs, where):
    # not jittable as size can changes
    # does not change pytreestructure ,

    if isinstance(lhs, jnp.ndarray):
        return lhs[where]
    else:
        # set None to non-chosen non-array values
        return lhs if where else None


def _node_applier(lhs: Any, where: bool, func: Callable[[Any], Any]):
    if isinstance(lhs, jnp.ndarray):
        return jnp.where(where, func(lhs), lhs)
    else:
        return func(lhs) if where else lhs


def _param_indexing_getter(model, *where: tuple[str, ...]):

    modelCopy = copy.copy(model)

    for field in model.__dataclass_fields__.values():
        value = modelCopy.__dict__[field.name]
        excluded_by_type = isinstance(value, str)
        excluded_by_meta = ("static" in field.metadata) and field.metadata["static"] is True  # fmt: skip
        excluded = excluded_by_type or excluded_by_meta
        if field.name not in where and not excluded:
            modelCopy.__dict__[field.name] = None

    return modelCopy


def _slice_indexing_getter(model, where: int | slice):

    resolved_where = (
        range(*where.indices(len(model.__dataclass_fields__)))
        if isinstance(where, slice)
        else (where,)
    )
    modelCopy = copy.copy(model)

    for i, field in enumerate(model.__dataclass_fields__.values()):
        value = modelCopy.__dict__[field.name]
        excluded_by_type = isinstance(value, str)
        excluded_by_meta = ("static" in field.metadata) and field.metadata["static"] is True  # fmt: skip
        excluded = excluded_by_type or excluded_by_meta
        if i not in resolved_where and not excluded:
            modelCopy.__dict__[field.name] = None

    return modelCopy


def _param_indexing_setter(model, set_value, *where: tuple[str]):

    assert isinstance(
        set_value, (float, int, complex, jnp.ndarray)
    ), f"Invalid set_value type = {type(set_value)}."

    modelCopy = copy.copy(model)
    for field in model.__dataclass_fields__.values():
        value = modelCopy.__dict__[field.name]
        excluded_by_type = isinstance(value, str)
        excluded_by_meta = ("static" in field.metadata) and field.metadata["static"] is True  # fmt: skip
        excluded = excluded_by_meta or excluded_by_type
        if field.name in where and not excluded:
            modelCopy.__dict__[field.name] = _node_setter(value, True, set_value)
    return modelCopy


def _param_indexing_applier(model, func, *where: tuple[str]):
    assert isinstance(func, Callable), f"Invalid func type = {type(func)}."

    modelCopy = copy.copy(model)
    for field in model.__dataclass_fields__.values():
        value = modelCopy.__dict__[field.name]
        excluded_by_type = isinstance(value, str)
        excluded_by_meta = ("static" in field.metadata) and field.metadata["static"] is True  # fmt: skip
        excluded = excluded_by_meta or excluded_by_type
        if field.name in where and not excluded:
            modelCopy.__dict__[field.name] = _node_applier(value, True, func)
    return modelCopy


def _slice_indexing_setter(model, set_value, where: int | slice):

    assert isinstance(
        set_value, (float, int, complex, jnp.ndarray)
    ), f"Invalid set_value type = {type(set_value)}."

    resolved_where = (
        range(*where.indices(len(model.__dataclass_fields__)))
        if isinstance(where, slice)
        else (where,)
    )

    modelCopy = copy.copy(model)
    for i, field in enumerate(model.__dataclass_fields__.values()):
        value = modelCopy.__dict__[field.name]

        excluded_by_type = isinstance(value, str)
        excluded_by_meta = ("static" in field.metadata) and field.metadata["static"] is True  # fmt: skip
        excluded = excluded_by_meta or excluded_by_type
        if i in resolved_where and not excluded:
            modelCopy.__dict__[field.name] = _node_setter(value, True, set_value)
    return modelCopy


def _slice_indexing_applier(model, func, where: int | slice):
    assert isinstance(func, Callable), f"Invalid func type = {type(func)}."

    resolved_where = (
        range(*where.indices(len(model.__dataclass_fields__)))
        if isinstance(where, slice)
        else (where,)
    )

    modelCopy = copy.copy(model)
    for i, field in enumerate(model.__dataclass_fields__.values()):
        value = modelCopy.__dict__[field.name]

        excluded_by_type = isinstance(value, str)
        excluded_by_meta = ("static" in field.metadata) and field.metadata["static"] is True  # fmt: skip
        excluded = excluded_by_meta or excluded_by_type
        if i in resolved_where and not excluded:
            modelCopy.__dict__[field.name] = _node_applier(value, True, func)
    return modelCopy


def _boolean_indexing_getter(model, where):
    lhs_leaves, lhs_treedef = model.flatten_leaves
    where_leaves, where_treedef = tree_flatten(where)
    lhs_leaves = [
        _node_getter(lhs_leaf, where_leaf)
        for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)
    ]

    return tree_unflatten(lhs_treedef, lhs_leaves)


def _boolean_indexing_setter(model, set_value, where):
    lhs_leaves, lhs_treedef = tree_flatten(model)
    where_leaves, rhs_treedef = tree_flatten(where)
    lhs_leaves = [
        _node_setter(lhs_leaf, where_leaf, set_value=set_value,)
        for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)  # fmt: skip
    ]

    return tree_unflatten(lhs_treedef, lhs_leaves)


def _boolean_indexing_applier(model, func, where):
    lhs_leaves, lhs_treedef = tree_flatten(model)
    where_leaves, rhs_treedef = tree_flatten(where)
    lhs_leaves = [
        _node_applier(lhs_leaf, where_leaf, func=func)
        for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)  # fmt: skip
    ]

    return tree_unflatten(lhs_treedef, lhs_leaves)


class _treeIndexerMethods:
    def add(getter_setter_self, set_value):
        return getter_setter_self.apply(lambda x: x + set_value)

    def multiply(getter_setter_self, set_value):
        return getter_setter_self.apply(lambda x: x * set_value)

    def divide(getter_setter_self, set_value):
        return getter_setter_self.apply(lambda x: x / set_value)

    def power(getter_setter_self, set_value):
        return getter_setter_self.apply(lambda x: x**set_value)

    def min(getter_setter_self, set_value):
        return getter_setter_self.apply(lambda x: jnp.minimum(x, set_value))

    def max(getter_setter_self, set_value):
        return getter_setter_self.apply(lambda x: jnp.maximum(x, set_value))


class treeIndexer:
    @property
    def at(self):
        class indexer:
            @dispatch(argnum=1)
            def __getitem__(inner_self, *args):
                raise NotImplementedError(
                    f"Indexing with type{tuple(type(arg) for arg in args)} is not implemented."
                )

            @__getitem__.register(str)
            @__getitem__.register(tuple)
            def _(inner_self, *args):
                """indexing by param name"""
                flatten_args = tree_leaves(args)
                if not all(isinstance(arg, str) for arg in flatten_args):
                    raise ValueError("Invalid indexing argument")

                class _getterSetterIndexer(_treeIndexerMethods):
                    def get(getter_setter_self):
                        return _param_indexing_getter(self, *flatten_args)

                    def set(getter_setter_self, set_value):
                        if self.frozen:
                            raise ValueError("Cannot set to a frozen treeclass.")
                        return _param_indexing_setter(self, set_value, *flatten_args)

                    def apply(getter_setter_self, func):
                        if self.frozen:
                            raise ValueError("Cannot apply to a frozen treeclass.")
                        return _param_indexing_applier(self, func, *flatten_args)

                return _getterSetterIndexer()

            @__getitem__.register(type(self))
            def _(inner_self, arg):
                """indexing by boolean pytree"""

                if not all(is_treeclass_leaf_bool(leaf) for leaf in tree_leaves(arg)):
                    raise ValueError("All model leaves must be boolean.")

                class _getterSetterIndexer(_treeIndexerMethods):
                    def get(getter_setter_self):
                        return _boolean_indexing_getter(self, arg)

                    def set(getter_setter_self, set_value):
                        if self.frozen:
                            raise ValueError("Cannot set to a frozen treeclass.")
                        return _boolean_indexing_setter(self, set_value, arg)

                    def apply(getter_setter_self, func):
                        if self.frozen:
                            raise ValueError("Cannot apply to a frozen treeclass.")
                        return _boolean_indexing_applier(self, func, arg)

                return _getterSetterIndexer()

            @__getitem__.register(int)
            @__getitem__.register(slice)
            def _(inner_self, arg):
                """indexing by slice/int on tree leaves"""

                class _getterSetterIndexer(_treeIndexerMethods):
                    def get(getter_setter_self):
                        return _slice_indexing_getter(self, arg)

                    def set(getter_setter_self, set_value):
                        if self.frozen:
                            raise ValueError("Cannot set to a frozen treeclass.")
                        return _slice_indexing_setter(self, set_value, arg)

                    def apply(getter_setter_self, func):
                        if self.frozen:
                            raise ValueError("Cannot apply to a frozen treeclass.")
                        return _slice_indexing_applier(self, func, arg)

                return _getterSetterIndexer()

        return indexer()

    def __getitem__(self, *args):
        """alias for .at[].get()"""
        return self.at.__getitem__(*args).get()
