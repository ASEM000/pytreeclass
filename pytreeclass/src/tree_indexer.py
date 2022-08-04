from __future__ import annotations

from collections.abc import Callable
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu

import pytreeclass
from pytreeclass.src.decorator_util import dispatch
from pytreeclass.src.tree_util import is_treeclass_leaf_bool

""" Getter """


@dispatch(argnum=0)
def _node_get(lhs: Any, where: Any, *args, **kwargs):
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


@_node_get.register(jnp.ndarray)
def _(lhs, where, array_as_leaves: bool = True):
    return (
        (lhs[jnp.where(where)])
        if array_as_leaves
        else (lhs if jnp.all(where) else None)
    )


@_node_get.register(pytreeclass.src.tree_base.treeBase)
def _(lhs, where, *args, **kwargs):
    return jtu.tree_map(lambda x: _node_get(x, where), lhs, *args, **kwargs)


@_node_get.register(int)
@_node_get.register(float)
@_node_get.register(complex)
def _(lhs, where, *args, **kwargs):
    # set None to non-chosen non-array values
    return lhs if where else None


@dispatch(argnum=1)
def _at_get(model, where, *args, **kwargs):
    raise NotImplementedError(f"Where type = {type(where)} is not implemented.")


@_at_get.register(pytreeclass.src.tree_base.treeBase)
def _(model, where, *args, **kwargs):
    lhs_leaves, lhs_treedef = jtu.tree_flatten(model)
    where_leaves, where_treedef = jtu.tree_flatten(where)
    lhs_leaves = [
        _node_get(lhs_leaf, where_leaf, *args, **kwargs)
        for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)
    ]

    return jtu.tree_unflatten(lhs_treedef, lhs_leaves)


""" Setter """


@dispatch(argnum=0)
def _node_set(lhs: Any, where: bool, set_value, *args, **kwargs):
    """Set pytree node value.

    Args:
        lhs: Node value.
        where: Conditional.
        set_value: Set value of shape 1.

    Returns:
        Modified node value.
    """
    raise NotImplementedError("lhs type is unknown.")


@_node_set.register(jnp.ndarray)
def _(lhs, where, set_value, array_as_leaves: bool = True):
    return (
        jnp.where(where, set_value, lhs)
        if array_as_leaves
        else (set_value if jnp.all(where) else lhs)
    )


@_node_set.register(pytreeclass.src.tree_base.treeBase)
def _(lhs, where, set_value, *args, **kwargs):
    return jtu.tree_map(lambda x: _node_set(x, where, set_value, *args, **kwargs), lhs)


@_node_set.register(int)
@_node_set.register(float)
@_node_set.register(complex)
def _(lhs, where, set_value, *args, **kwargs):
    return set_value if where else lhs


@dispatch(argnum=2)
def _at_set(model, set_value, where, *args, **kwargs):
    raise NotImplementedError(f"Where type = {type(where)} is not implemented.")


@_at_set.register(pytreeclass.src.tree_base.treeBase)
def _(model, set_value, where, *args, **kwargs):
    lhs_leaves, lhs_treedef = jtu.tree_flatten(model)
    where_leaves, rhs_treedef = jtu.tree_flatten(where)
    lhs_leaves = [
        _node_set(lhs_leaf, where_leaf, set_value, *args, **kwargs)
        for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)  # fmt: skip
    ]

    return jtu.tree_unflatten(lhs_treedef, lhs_leaves)


""" Apply """


@dispatch(argnum=0)
def _node_apply(lhs: Any, where: bool, func: Callable[[Any], Any], *args, **kwargs):
    """Set pytree node

    Args:
        lhs (Any): Node value.
        where (bool): Conditional
        func (Callable[[Any], Any]): Callable

    Raises:
        NotImplementedError:
    """
    raise NotImplementedError(f"lhs type= {type(lhs)} is not implemented.")


@_node_apply.register(jnp.ndarray)
def _(lhs, where, func, array_as_leaves: bool = True):
    return (
        jnp.where(where, func(lhs), lhs)
        if array_as_leaves
        else (func(lhs) if jnp.all(where) else lhs)
    )


@_node_apply.register(pytreeclass.src.tree_base.treeBase)
def _(lhs, where, func, *args, **kwargs):
    return jtu.tree_map(lambda x: _node_apply(x, where, func, *args, **kwargs), lhs)


@_node_apply.register(int)
@_node_apply.register(float)
@_node_apply.register(complex)
def _(lhs, where, func, *args, **kwargs):
    return func(lhs) if where else lhs


@dispatch(argnum=2)
def _at_apply(model, set_value, where, *args, **kwargs):
    raise NotImplementedError(f"Where type = {type(where)} is not implemented.")


@_at_apply.register(pytreeclass.src.tree_base.treeBase)
def _(model, func, where, *args, **kwargs):
    lhs_leaves, lhs_treedef = jtu.tree_flatten(model)
    where_leaves, rhs_treedef = jtu.tree_flatten(where)
    lhs_leaves = [
        _node_apply(lhs_leaf, where_leaf, func, *args, **kwargs)
        for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)
    ]

    return jtu.tree_unflatten(lhs_treedef, lhs_leaves)


""" Reduce apply """


@dispatch(argnum=2)
def _at_reduce_apply(model, set_value, where, *args, **kwargs):
    raise NotImplementedError(f"Where type = {type(where)} is not implemented.")


@_at_reduce_apply.register(pytreeclass.src.tree_base.treeBase)
def _(tree, func, where, reduce_op=jnp.add, **kwargs):
    mask = tree.at[where].get()
    mask = mask.at[mask == mask].apply(func, array_as_leaves=False)
    return jtu.tree_reduce(lambda acc, cur: reduce_op(acc, cur), mask)


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

            @__getitem__.register(type(self))
            def _(inner_self, arg):
                """indexing by boolean pytree"""

                if not all(
                    is_treeclass_leaf_bool(leaf) for leaf in jtu.tree_leaves(arg)
                ):
                    raise ValueError("All model leaves must be boolean.")

                class getterSetterIndexer(treeIndexerMethods):
                    def get(getter_setter_self, **kwargs):
                        return _at_get(self, arg, **kwargs)

                    def set(getter_setter_self, set_value, **kwargs):
                        if self.frozen:
                            raise ValueError("Cannot set to a frozen treeclass.")
                        return _at_set(self, set_value, arg, **kwargs)

                    def apply(getter_setter_self, func, **kwargs):
                        if self.frozen:
                            raise ValueError("Cannot apply to a frozen treeclass.")
                        return _at_apply(self, func, arg, **kwargs)

                    def reduce_apply(getter_setter_self, func, **kwargs):
                        if self.frozen:
                            raise ValueError("Cannot apply to a frozen treeclass.")
                        return _at_reduce_apply(self, func, arg, **kwargs)

                return getterSetterIndexer()

        return indexer()

    def __getitem__(self, *args):
        """alias for .at[].get()"""
        return self.at.__getitem__(*args).get()
        # trunk-ignore(flake8/E501)
