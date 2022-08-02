from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
from jax.tree_util import tree_flatten, tree_leaves, tree_map, tree_unflatten

import pytreeclass
from pytreeclass.src.decorator_util import dispatch
from pytreeclass.src.tree_util import is_treeclass_leaf_bool


@dispatch(argnum=0)
def _node_setter(lhs: Any, where: bool, set_value):
    """Set pytree node value.

    Args:
        lhs: Node value.
        where: Conditional.
        set_value: Set value of shape 1.

    Returns:
        Modified node value.
    """
    raise NotImplementedError("lhs type is unknown.")


@_node_setter.register(jnp.ndarray)
def _(lhs, where, set_value):
    return jnp.where(where, set_value, lhs)


@_node_setter.register(pytreeclass.src.tree_base.treeBase)
def _(lhs, where, set_value):
    return tree_map(lambda x: _node_setter(x, where, set_value), lhs)


@_node_setter.register(int)
@_node_setter.register(float)
@_node_setter.register(complex)
def _(lhs, where, set_value):
    return set_value if where else lhs


@dispatch(argnum=0)
def _node_at_getter(lhs: Any, where: Any):
    """Get pytree node  value

    Args:
        lhs (Any): Node value.
        where (Any): Conditional

    Raises:
        NotImplementedError:
    """
    # not jittable as size can changes
    # does not change pytreestructure ,
    raise NotImplementedError(f"lhs type ={type(lhs)} is not implemented.")


@_node_at_getter.register(jnp.ndarray)
def _(lhs, where):
    return lhs[jnp.where(where)]


@_node_at_getter.register(pytreeclass.src.tree_base.treeBase)
def _(lhs, where):
    return tree_map(lambda x: _node_at_getter(x, where), lhs)


@_node_at_getter.register(int)
@_node_at_getter.register(float)
@_node_at_getter.register(complex)
def _(lhs, where):
    # set None to non-chosen non-array values
    return lhs if where else None


@dispatch(argnum=0)
def _node_applier(lhs: Any, where: bool, func: Callable[[Any], Any]):
    """Set pytree node

    Args:
        lhs (Any): Node value.
        where (bool): Conditional
        func (Callable[[Any], Any]): Callable

    Raises:
        NotImplementedError:
    """
    raise NotImplementedError(f"lhs type= {type(lhs)} is not implemented.")


@_node_applier.register(jnp.ndarray)
def _(lhs, where, func):
    return jnp.where(where, func(lhs), lhs)


@_node_applier.register(pytreeclass.src.tree_base.treeBase)
def _(lhs, where, func):
    return tree_map(lambda x: _node_applier(x, where, func), lhs)


@_node_applier.register(int)
@_node_applier.register(float)
@_node_applier.register(complex)
def _(lhs, where, func):
    return func(lhs) if where else lhs


@dispatch(argnum=1)
def _at_getter(model, where):
    raise NotImplementedError(f"Where type = {type(where)} is not implemented.")


@_at_getter.register(int)
@_at_getter.register(str)
@_at_getter.register(tuple)
def _(model, *where):
    modelCopy = copy.copy(model)

    for i, field in enumerate(model.__dataclass_fields__.values()):
        value = modelCopy.__dict__[field.name]
        excluded_by_type = isinstance(value, str)
        excluded_by_meta = ("static" in field.metadata) and field.metadata["static"] is True  # fmt: skip
        excluded = excluded_by_type or excluded_by_meta

        if not excluded and not (i in where or field.name in where):
            modelCopy.__dict__[field.name] = _node_at_getter(value, False)

    return modelCopy


@_at_getter.register(pytreeclass.src.tree_base.treeBase)
def _(model, where):
    lhs_leaves, lhs_treedef = model.flatten_leaves
    where_leaves, where_treedef = tree_flatten(where)
    lhs_leaves = [
        _node_at_getter(lhs_leaf, where_leaf)
        for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)
    ]

    return tree_unflatten(lhs_treedef, lhs_leaves)


@dispatch(argnum=2)
def _at_setter(model, set_value, where):
    raise NotImplementedError(f"Where type = {type(where)} is not implemented.")


@_at_setter.register(int)
@_at_setter.register(str)
@_at_setter.register(tuple)
def _(model, set_value, *where: tuple[str | int, ...]):
    modelCopy = copy.copy(model)

    for i, field in enumerate(model.__dataclass_fields__.values()):
        value = modelCopy.__dict__[field.name]
        excluded_by_type = isinstance(value, str)
        excluded_by_meta = ("static" in field.metadata) and field.metadata["static"] is True  # fmt: skip
        excluded = excluded_by_type or excluded_by_meta

        if not excluded and (i in where or field.name in where):
            modelCopy.__dict__[field.name] = _node_setter(value, True, set_value)

    return modelCopy


@_at_setter.register(pytreeclass.src.tree_base.treeBase)
def _(model, set_value, where):
    lhs_leaves, lhs_treedef = tree_flatten(model)
    where_leaves, rhs_treedef = tree_flatten(where)
    lhs_leaves = [
        _node_setter(lhs_leaf, where_leaf, set_value,)
        for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)  # fmt: skip
    ]

    return tree_unflatten(lhs_treedef, lhs_leaves)


@dispatch(argnum=2)
def _at_applier(model, set_value, where):
    raise NotImplementedError(f"Where type = {type(where)} is not implemented.")


@_at_applier.register(int)
@_at_applier.register(str)
@_at_applier.register(tuple)
def _(model, set_value, *where):
    modelCopy = copy.copy(model)

    for i, field in enumerate(model.__dataclass_fields__.values()):
        value = modelCopy.__dict__[field.name]
        excluded_by_type = isinstance(value, str)
        excluded_by_meta = ("static" in field.metadata) and field.metadata["static"] is True  # fmt: skip
        excluded = excluded_by_type or excluded_by_meta

        if not excluded and (i in where or field.name in where):
            modelCopy.__dict__[field.name] = _node_applier(value, True, set_value)

    return modelCopy


@_at_applier.register(pytreeclass.src.tree_base.treeBase)
def _(model, func, where):
    lhs_leaves, lhs_treedef = tree_flatten(model)
    where_leaves, rhs_treedef = tree_flatten(where)
    lhs_leaves = [
        _node_applier(lhs_leaf, where_leaf, func=func)
        for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)  # fmt: skip
    ]

    return tree_unflatten(lhs_treedef, lhs_leaves)


class treeIndexerMethods:
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

                class getterSetterIndexer(treeIndexerMethods):
                    def get(getter_setter_self):
                        return _at_getter(self, *flatten_args)

                    def set(getter_setter_self, set_value):
                        if self.frozen:
                            raise ValueError("Cannot set to a frozen treeclass.")
                        return _at_setter(self, set_value, *flatten_args)

                    def apply(getter_setter_self, func):
                        if self.frozen:
                            raise ValueError("Cannot apply to a frozen treeclass.")
                        return _at_applier(self, func, *flatten_args)

                return getterSetterIndexer()

            @__getitem__.register(type(self))
            def _(inner_self, arg):
                """indexing by boolean pytree"""

                if not all(is_treeclass_leaf_bool(leaf) for leaf in tree_leaves(arg)):
                    raise ValueError("All model leaves must be boolean.")

                class getterSetterIndexer(treeIndexerMethods):
                    def get(getter_setter_self):
                        return _at_getter(self, arg)

                    def set(getter_setter_self, set_value):
                        if self.frozen:
                            raise ValueError("Cannot set to a frozen treeclass.")
                        return _at_setter(self, set_value, arg)

                    def apply(getter_setter_self, func):
                        if self.frozen:
                            raise ValueError("Cannot apply to a frozen treeclass.")
                        return _at_applier(self, func, arg)

                return getterSetterIndexer()

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
