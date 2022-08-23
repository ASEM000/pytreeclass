from __future__ import annotations

import functools as ft
from collections.abc import Callable
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from pytreeclass.src.decorator_util import dispatch
from pytreeclass.src.tree_util import (
    _freeze_nodes,
    _unfreeze_nodes,
    is_treeclass_leaf_bool,
    static_value,
    tree_copy,
)

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
    def _(tree, where, is_leaf=None, **kwargs):
        lhs_leaves, lhs_treedef = jtu.tree_flatten(tree, is_leaf=is_leaf)
        where_leaves, where_treedef = jtu.tree_flatten(where, is_leaf=is_leaf)
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

    @_array_node.register(bool)
    def _(lhs, where, set_value):
        # in python isinstance(True/False,int) is True
        # without this dispatch, it will be handled with the int dispatch
        return set_value if jnp.all(where) else lhs

    @_array_node.register(int)
    @_array_node.register(float)
    @_array_node.register(complex)
    @_array_node.register(jnp.ndarray)
    def _(lhs, where, set_value, array_as_leaves: bool = True):
        # lhs is numeric node
        # set_value in acceptable set_value type for a numeric node
        # For some reason python isinstance(True,int) is True ?
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
    @_node_set.register(type(None))
    def _(lhs, where, set_value, **kwargs):
        # where == None can be obtained by
        # is_leaf = lambda x : x is None
        return set_value if (where in [True, None]) else lhs

    @dispatch(argnum=1)
    def __at_set(tree, where, set_value, is_leaf=None, **kwargs):
        raise NotImplementedError(f"Set where type = {type(where)} is not implemented.")

    @__at_set.register(type(tree))
    def _(tree, where, set_value, is_leaf=None, **kwargs):
        lhs_leaves, lhs_treedef = jtu.tree_flatten(tree, is_leaf=is_leaf)
        where_leaves, rhs_treedef = jtu.tree_flatten(where, is_leaf=is_leaf)
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
    @_node_apply.register(type(None))
    def _(lhs, where, func, **kwargs):
        return func(lhs) if (where in [True, None]) else lhs

    @dispatch(argnum=1)
    def __at_apply(tree, where, func, **kwargs):
        raise NotImplementedError(
            f"Apply where type = {type(where)} is not implemented."
        )

    @__at_apply.register(type(tree))
    def _(tree, where, func, is_leaf=None, **kwargs):

        lhs_leaves, lhs_treedef = jtu.tree_flatten(tree, is_leaf=is_leaf)
        where_leaves, rhs_treedef = jtu.tree_flatten(where, is_leaf=is_leaf)
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
    def __at_static(tree, where, **kwargs):
        return tree.at[where].apply(static_value, array_as_leaves=False)

    return __at_static(tree, where, **kwargs)


class treeIndexer:
    @property
    def at(self):
        class indexer:
            @dispatch(argnum=1)
            def __getitem__(mask_self, *args):
                raise NotImplementedError(
                    f"Indexing with type{tuple(type(arg) for arg in args)} is not implemented."
                )

            @__getitem__.register(type(self))
            def _(mask_self, arg):
                """indexing by boolean pytree"""

                assert all(
                    is_treeclass_leaf_bool(leaf) for leaf in jtu.tree_leaves(arg)
                ), f"All tree leaves must be boolean.Found {jtu.tree_leaves(arg)}"

                class opIndexer:
                    def get(op_self, **kwargs):
                        return ft.partial(_at_get, where=arg)(tree=self, **kwargs)

                    def set(op_self, set_value, **kwargs):
                        return ft.partial(_at_set, where=arg)(
                            tree=self, set_value=set_value, **kwargs
                        )

                    def apply(op_self, func, **kwargs):
                        return ft.partial(_at_apply, where=arg)(
                            tree=self, func=func, **kwargs
                        )

                    def reduce(op_self, func, **kwargs):
                        return ft.partial(_at_reduce, where=arg)(
                            tree=self, func=func, **kwargs
                        )

                    def static(op_self, **kwargs):
                        return ft.partial(_at_static, where=arg)(tree=self, **kwargs)

                    # derived methods

                    def add(op_self, set_value):
                        return op_self.apply(lambda x: x + set_value)

                    def multiply(op_self, set_value):
                        return op_self.apply(lambda x: x * set_value)

                    def divide(op_self, set_value):
                        return op_self.apply(lambda x: x / set_value)

                    def power(op_self, set_value):
                        return op_self.apply(lambda x: x**set_value)

                    def min(op_self, set_value):
                        return op_self.apply(lambda x: jnp.minimum(x, set_value))

                    def max(op_self, set_value):
                        return op_self.apply(lambda x: jnp.maximum(x, set_value))

                    def reduce_sum(op_self):
                        return op_self.reduce(lambda x, y: x + jnp.sum(y))

                    def reduce_product(op_self):
                        return op_self.reduce(
                            lambda x, y: x * jnp.prod(y), initializer=1
                        )

                    def reduce_max(op_self):
                        return op_self.reduce(
                            lambda x, y: jnp.maximum(x, jnp.max(y)),
                            initializer=-jnp.inf,
                        )

                    def reduce_min(op_self):
                        return op_self.reduce(
                            lambda x, y: jnp.minimum(x, jnp.min(y)),
                            initializer=+jnp.inf,
                        )

                return opIndexer()

            @__getitem__.register(str)
            def _(mask_self, arg):
                class opIndexer:
                    def get(op_self):
                        return getattr(self, arg)

                    def set(op_self, set_value):
                        getattr(self, arg)  # check if attribute already defined
                        new_self = tree_copy(self)
                        object.__setattr__(new_self, arg, set_value)
                        return new_self

                    def apply(op_self, func, **kwargs):
                        return self.at[arg].set(func(self.at[arg].get()))

                    def __call__(op_self, *args, **kwargs):
                        new_self = tree_copy(self)
                        method = getattr(new_self, arg)
                        object.__setattr__(new_self, "__immutable_treeclass__", False)
                        value = method(*args, **kwargs)
                        object.__setattr__(new_self, "__immutable_treeclass__", True)
                        return value, new_self

                    def freeze(op_self):
                        return self.at[arg].set(
                            _freeze_nodes(tree_copy(getattr(self, arg)))
                        )

                    def unfreeze(op_self):
                        return self.at[arg].set(
                            _freeze_nodes(tree_copy(getattr(self, arg)))
                        )

                return opIndexer()

            @__getitem__.register(type(Ellipsis))
            def _(mask_self, arg):
                """Ellipsis as an alias for all elements"""

                class opIndexer:
                    freeze = lambda _: _freeze_nodes(tree_copy(self))
                    unfreeze = lambda _: _unfreeze_nodes(tree_copy(self))
                    get = self.at[self == self].get
                    set = self.at[self == self].set
                    apply = self.at[self == self].apply
                    reduce = self.at[self == self].reduce
                    static = self.at[self == self].static
                    add = self.at[self == self].add
                    multiply = self.at[self == self].multiply
                    divide = self.at[self == self].divide
                    power = self.at[self == self].power
                    min = self.at[self == self].min
                    max = self.at[self == self].max
                    reduce_sum = self.at[self == self].reduce_sum
                    reduce_max = self.at[self == self].reduce_max

                return opIndexer()

        return indexer()
