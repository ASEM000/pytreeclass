from __future__ import annotations

import functools as ft
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

import pytreeclass.src.tree_util as ptu
from pytreeclass.src.decorator_util import dispatch
from pytreeclass.src.tree_util import static_value

""" Getter """


def _at_get(tree, where, **kwargs):
    @dispatch(argnum=0)
    def _node_get(lhs: Any, where: Any, **kwargs):
        """Get pytree node  value

        Args:
            lhs (Any): Node value.
            where (Any): Conditional

        Raises:
            NotImplementedError:
        """
        # not jittable as size can changes
        # does not change pytreestructure ,
        raise NotImplementedError(f"Get node type ={type(lhs)} is not implemented.")

    @_node_get.register(jax.interpreters.partial_eval.DynamicJaxprTracer)
    @_node_get.register(jnp.ndarray)
    def _(lhs, where, array_as_leaves: bool = True):
        return (
            (lhs[jnp.where(where)])
            if array_as_leaves
            else (lhs if jnp.all(where) else None)
        )

    @_node_get.register(int)
    @_node_get.register(float)
    @_node_get.register(complex)
    @_node_get.register(tuple)
    @_node_get.register(list)
    @_node_get.register(str)
    def _(lhs, where, **kwargs):
        # set None to non-chosen non-array values
        return lhs if where else None

    @dispatch(argnum=1)
    def __at_get(tree, where, **kwargs):
        raise NotImplementedError(f"Get where type = {type(where)} is not implemented.")

    @__at_get.register(type(tree))
    def _(tree, where, **kwargs):

        assert all(
            ptu.is_treeclass_leaf_bool(leaf) for leaf in jtu.tree_leaves(where)
        ), f"All tree leaves must be boolean.Found {jtu.tree_leaves(where)}"

        lhs_leaves, lhs_treedef = jtu.tree_flatten(tree)
        where_leaves, where_treedef = jtu.tree_flatten(where)
        lhs_leaves = [
            _node_get(lhs_leaf, where_leaf, **kwargs)
            for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)
        ]

        return jtu.tree_unflatten(lhs_treedef, lhs_leaves)

    return __at_get(tree, where, **kwargs)


""" Setter """


def _at_set(tree, where, set_value, **kwargs):
    @dispatch(argnum=0)
    def _node_set(lhs: Any, where: bool, set_value, **kwargs):
        """Set pytree node value.

        Args:
            lhs: Node value.
            where: Conditional.
            set_value: Set value of shape 1.

        Returns:
            Modified node value.
        """
        raise NotImplementedError(f"Set node type = {type(lhs)} is unknown.")

    @_node_set.register(jax.interpreters.partial_eval.DynamicJaxprTracer)
    @_node_set.register(jnp.ndarray)
    @dispatch(argnum=2)
    def _array_node(lhs, where, set_value):
        """Multi dispatched on lhs and where type"""
        # lhs is numeric node
        # set_value in not acceptable set_value type for numeric node
        # thus array_as_leaves is not an optional keyword argument
        # An example is setting an array to None
        # then the entire array is set to None if all array elements are
        # satisfied by the where condition
        return set_value if jnp.all(where) else lhs

    @_array_node.register(int)
    @_array_node.register(float)
    @_array_node.register(complex)
    @_array_node.register(jnp.ndarray)
    def _(lhs, where, set_value, array_as_leaves: bool = True):
        # lhs is numeric node
        # set_value in acceptable set_value type for a numeric node
        return (
            jnp.where(where, set_value, lhs)
            if array_as_leaves
            else (set_value if jnp.all(where) else lhs)
        )

    @_node_set.register(int)
    @_node_set.register(float)
    @_node_set.register(complex)
    @_node_set.register(tuple)
    @_node_set.register(list)
    @_node_set.register(str)
    def _(lhs, where, set_value, **kwargs):
        return set_value if where else lhs

    @dispatch(argnum=1)
    def __at_set(tree, where, set_value, **kwargs):
        raise NotImplementedError(f"Set where type = {type(where)} is not implemented.")

    @__at_set.register(type(tree))
    def _(tree, where, set_value, **kwargs):

        assert all(
            ptu.is_treeclass_leaf_bool(leaf) for leaf in jtu.tree_leaves(where)
        ), f"All tree leaves must be boolean.Found {jtu.tree_leaves(where)}"

        lhs_leaves, lhs_treedef = jtu.tree_flatten(tree)
        where_leaves, rhs_treedef = jtu.tree_flatten(where)
        lhs_leaves = [
            _node_set(lhs_leaf, where_leaf, set_value, **kwargs)
            for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)
        ]

        return jtu.tree_unflatten(lhs_treedef, lhs_leaves)

    return __at_set(tree, where, set_value, **kwargs)


""" Apply """


def _at_apply(tree, where, func, **kwargs):
    @dispatch(argnum=0)
    def _node_apply(lhs: Any, where: bool, func: Callable[[Any], Any], **kwargs):
        """Set pytree node

        Args:
            lhs (Any): Node value.
            where (bool): Conditional
            func (Callable[[Any], Any]): Callable

        Raises:
            NotImplementedError:
        """
        raise NotImplementedError(f"Apply node type= {type(lhs)} is not implemented.")

    @_node_apply.register(jax.interpreters.partial_eval.DynamicJaxprTracer)
    @_node_apply.register(jnp.ndarray)
    def _(lhs, where, func, array_as_leaves: bool = True):
        return (
            jnp.where(where, func(lhs), lhs)
            if array_as_leaves
            else (func(lhs) if jnp.all(where) else lhs)
        )

    @_node_apply.register(int)
    @_node_apply.register(float)
    @_node_apply.register(complex)
    @_node_apply.register(tuple)
    @_node_apply.register(list)
    @_node_apply.register(str)
    def _(lhs, where, func, **kwargs):
        return func(lhs) if where else lhs

    @dispatch(argnum=1)
    def __at_apply(tree, where, func, **kwargs):
        raise NotImplementedError(
            f"Apply where type = {type(where)} is not implemented."
        )

    @__at_apply.register(type(tree))
    def _(tree, where, func, **kwargs):

        assert all(
            ptu.is_treeclass_leaf_bool(leaf) for leaf in jtu.tree_leaves(where)
        ), f"All tree leaves must be boolean.Found {jtu.tree_leaves(where)}"

        lhs_leaves, lhs_treedef = jtu.tree_flatten(tree)
        where_leaves, rhs_treedef = jtu.tree_flatten(where)
        lhs_leaves = [
            _node_apply(lhs_leaf, where_leaf, func, **kwargs)
            for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)
        ]

        return jtu.tree_unflatten(lhs_treedef, lhs_leaves)

    return __at_apply(tree, where, func, **kwargs)


""" Reduce """


def _at_reduce(tree, where, func, **kwargs):
    @dispatch(argnum=1)
    def __at_reduce(tree, where, func, **kwargs):
        raise NotImplementedError(
            f"Reduce where type = {type(where)} is not implemented."
        )

    @__at_reduce.register(type(tree))
    def _(tree, where, func, initializer=0, **kwargs):
        return jtu.tree_reduce(func, tree.at[where].get(), initializer)

    return __at_reduce(tree, where, func, **kwargs)


""" Static"""


def _at_static(tree, where, **kwargs):
    @dispatch(argnum=1)
    def __at_static(tree, where, **kwargs):
        raise NotImplementedError(
            f"Static where type = {type(where)} is not implemented."
        )

    @__at_static.register(type(tree))
    def _(tree, where, **kwargs):
        return tree.at[where].apply(static_value, array_as_leaves=False)

    return __at_static(tree, where, **kwargs)


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

    def reduce_sum(getter_setter_self):
        return getter_setter_self.reduce(lambda x, y: x + jnp.sum(y))

    def reduce_product(getter_setter_self):
        return getter_setter_self.reduce(lambda x, y: x * jnp.prod(y), initializer=1)

    def reduce_max(getter_setter_self):
        return getter_setter_self.reduce(
            lambda x, y: jnp.maximum(x, jnp.max(y)), initializer=-jnp.inf
        )

    def reduce_min(getter_setter_self):
        return getter_setter_self.reduce(
            lambda x, y: jnp.minimum(x, jnp.min(y)), initializer=+jnp.inf
        )


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

                class getterSetterIndexer(treeIndexerMethods):
                    def get(getter_setter_self, **kwargs):
                        return ft.partial(_at_get, where=arg)(tree=self, **kwargs)

                    def set(getter_setter_self, set_value, **kwargs):
                        if self.frozen:
                            raise ValueError("Cannot set to a frozen treeclass.")
                        return ft.partial(_at_set, where=arg)(
                            tree=self, set_value=set_value, **kwargs
                        )

                    def apply(getter_setter_self, func, **kwargs):
                        if self.frozen:
                            raise ValueError("Cannot apply to a frozen treeclass.")
                        return ft.partial(_at_apply, where=arg)(
                            tree=self, func=func, **kwargs
                        )

                    def reduce(getter_setter_self, func, **kwargs):
                        if self.frozen:
                            raise ValueError("Cannot reduce to a frozen treeclass.")
                        return ft.partial(_at_reduce, where=arg)(
                            tree=self, func=func, **kwargs
                        )

                    def static(getter_setter_self, **kwargs):
                        if self.frozen:
                            raise ValueError(
                                "Cannot apply static to a frozen treeclass."
                            )
                        return ft.partial(_at_static, where=arg)(tree=self, **kwargs)

                return getterSetterIndexer()

            @__getitem__.register(type(Ellipsis))
            def _(inner_self, arg):
                """Ellipsis as an alias for all elements"""
                return self.at.__getitem__(self == self)

        return indexer()
