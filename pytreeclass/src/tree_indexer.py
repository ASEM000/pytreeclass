from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_leaves, tree_map, tree_unflatten

from pytreeclass.src.decorator_util import dispatch
from pytreeclass.src.tree_util import is_treeclass, is_treeclass_leaf_bool


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
    elif is_treeclass(lhs):
        return tree_map(lambda x: set_value if where else x, lhs)
    else:
        return set_value if where else lhs


def _node_getter(lhs, where):
    # not jittable as size can changes
    # does not change pytreestructure ,

    if isinstance(lhs, jnp.ndarray):
        return lhs[jnp.where(where)]
    elif is_treeclass(lhs):
        return tree_map(
            lambda x: x
            if where
            else (jnp.array([]) if isinstance(x, jnp.ndarray) else None),
            lhs,
        )
    else:
        # set None to non-chosen non-array values
        return lhs if where else None


def _node_applier(lhs: Any, where: bool, func: Callable[[Any], Any]):
    if isinstance(lhs, jnp.ndarray):
        return jnp.where(where, func(lhs), lhs)
    elif is_treeclass(lhs):
        return tree_map(lambda x: func(x) if where else x, lhs)
    else:
        return func(lhs) if where else lhs


def _non_boolean_indexing_getter(model, *where: tuple[str | int, ...]):

    if model.frozen:
        return model

    modelCopy = copy.copy(model)

    for i, field in enumerate(model.__dataclass_fields__.values()):
        value = modelCopy.__dict__[field.name]
        excluded_by_type = isinstance(value, str)
        excluded_by_meta = ("static" in field.metadata) and field.metadata["static"] is True  # fmt: skip
        excluded = excluded_by_type or excluded_by_meta

        if not excluded and not (i in where or field.name in where):
            modelCopy.__dict__[field.name] = _node_getter(value, False)

    return modelCopy


def _non_boolean_indexing_setter(model, set_value, *where: tuple[str | int, ...]):

    if model.frozen:
        return model

    modelCopy = copy.copy(model)

    for i, field in enumerate(model.__dataclass_fields__.values()):
        value = modelCopy.__dict__[field.name]
        excluded_by_type = isinstance(value, str)
        excluded_by_meta = ("static" in field.metadata) and field.metadata["static"] is True  # fmt: skip
        excluded = excluded_by_type or excluded_by_meta

        if not excluded and (i in where or field.name in where):
            modelCopy.__dict__[field.name] = _node_setter(value, True, set_value)

    return modelCopy


def _non_boolean_indexing_applier(model, set_value, *where: tuple[str | int, ...]):

    if model.frozen:
        return model

    modelCopy = copy.copy(model)

    for i, field in enumerate(model.__dataclass_fields__.values()):
        value = modelCopy.__dict__[field.name]
        excluded_by_type = isinstance(value, str)
        excluded_by_meta = ("static" in field.metadata) and field.metadata["static"] is True  # fmt: skip
        excluded = excluded_by_type or excluded_by_meta

        if not excluded and (i in where or field.name in where):
            modelCopy.__dict__[field.name] = _node_applier(value, True, set_value)

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

            @__getitem__.register(int)
            @__getitem__.register(str)
            @__getitem__.register(tuple)
            @__getitem__.register(range)
            def _(inner_self, *args):
                """Non-boolean indexing"""

                # Normalize indices
                flatten_args = [
                    (arg + len(self.__dataclass_fields__) if arg < 0 else arg)
                    if isinstance(arg, int)
                    else arg
                    for arg in tree_leaves(args)
                ]

                class _getterSetterIndexer(_treeIndexerMethods):
                    def get(getter_setter_self):
                        return _non_boolean_indexing_getter(self, *flatten_args)

                    def set(getter_setter_self, set_value):
                        if self.frozen:
                            raise ValueError("Cannot set to a frozen treeclass.")
                        return _non_boolean_indexing_setter(
                            self, set_value, *flatten_args
                        )

                    def apply(getter_setter_self, func):
                        if self.frozen:
                            raise ValueError("Cannot apply to a frozen treeclass.")
                        return _non_boolean_indexing_applier(self, func, *flatten_args)

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

            @__getitem__.register(slice)
            def _(inner_self, arg):
                return inner_self.__getitem__(
                    *range(*arg.indices(len(self.__dataclass_fields__)))
                )

        return indexer()

    def __getitem__(self, *args):
        """alias for .at[].get()"""
        return self.at.__getitem__(*args).get()
        # trunk-ignore(flake8/E501)
