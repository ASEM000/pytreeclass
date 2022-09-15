from __future__ import annotations

import functools as ft
from collections.abc import Iterable
from dataclasses import Field, field
from types import FunctionType
from typing import Any, Callable

import jax.numpy as jnp
import jax.tree_util as jtu

from pytreeclass._src.tree_util import (
    _pytree_map,
    _tree_immutate,
    _tree_mutate,
    is_treeclass,
    _tree_fields
)

PyTree = Any



def _mutable(func):
    """decorator that allow mutable behvior
    for class methods/ function with treeclass as first argument

    Example:
        class ... :

        >>> @_mutable
        ... def mutable_method(self):
        ...    return self.value + 1
    """
    assert isinstance(
        func, FunctionType
    ), f"mutable can only be applied to methods. Found{type(func)}"

    @ft.wraps(func)
    def mutable_method(self, *args, **kwargs):
        self = _tree_mutate(tree=self)
        output = func(self, *args, **kwargs)
        self = _tree_immutate(tree=self)
        return output

    return mutable_method


class cached_method:
    def __init__(self, func):
        self.name = func.__name__
        self.func = func

    def __get__(self, instance, owner):
        output = self.func(instance)
        cached_func = ft.wraps(self.func)(lambda *args, **kwargs: output)
        object.__setattr__(instance, self.name, cached_func)
        return cached_func


def _is_nondiff(item):
    """check if tree is non-differentiable"""

    def _is_nondiff_item(node):
        """check if node is non-differentiable"""
        # non-differentiable types
        if isinstance(node, (int, bool, str)):
            return True

        # non-differentiable array
        elif isinstance(node, jnp.ndarray) and not jnp.issubdtype(
            node.dtype, jnp.inexact
        ):
            return True

        # non-differentiable type
        elif isinstance(node, Callable) and not is_treeclass(node):
            return True

        return False

    if isinstance(item, Iterable):
        # if an iterable has at least one non-differentiable item
        # then the whole iterable is non-differentiable
        return jtu.tree_all(jtu.tree_map(_is_nondiff_item, item))
    return _is_nondiff_item(item)


def filter_nondiff(tree, where: PyTree | None = None):
    """filter non-differentiable fields from a treeclass instance

    Note:
        Mark fields as non-differentiable with adding metadata `dict(static=True,nondiff=True)`
        the way it is done is by adding a field `__undeclared_fields__` to the treeclass instance
        that contains the non-differentiable fields.
        This is done to avoid mutating the original treeclass class or copying .

        during the `tree_flatten`, tree_fields are the combination of
        {__dataclass_fields__, __undeclared_fields__} this means that a field defined
        in `__undeclared_fields__` with the same name as in __dataclass_fields__
        will override its properties, this is useful if you want to change the metadata
        of a field but don't want to change the original field definition.

    Example:
        @pytc.treeclass
        class Linear:
            weight: jnp.ndarray
            bias: jnp.ndarray
            other: jnp.ndarray = (1,2,3,4)
            a: int = 1
            b: float = 1.0
            c: int = 1
            d: float = 2.0
            act : Callable = jax.nn.tanh

            def __init__(self,in_dim,out_dim):
                self.weight = jnp.ones((in_dim,out_dim))
                self.bias =  jnp.ones((1,out_dim))

            def __call__(self,x):
                return self.act(self.b+x)

        @jax.value_and_grad
        def loss_func(model):
            return jnp.mean((model(1.)-0.5)**2)

        @jax.jit
        def update(model):
            value,grad = loss_func(model)
            return value,model-1e-3*grad

        def train(model,epochs=10_000):
            model = filter_nondiff(model)
            for _ in range(epochs):
                value,model = update(model)
            return model

        >>> model = Linear(1,1)
        >>> model = train(model)
        >>> print(model)
        # Linear(
        #   weight=[[1.]],
        #   bias=[[1.]],
        #   *other=(1,2,3,4),
        #   *a=1,
        #   b=-0.36423424,
        #   *c=1,
        #   d=2.0,
        #   *act=tanh(x)
        # )


    ** for non None where, fields as non-differentiable by using `where` mask.

    Example:
        @pytc.treeclass
        class L0:
            a: int = 1
            b: int = 2
            c: int = 3

        @pytc.treeclass
        class L1:
            a: int = 1
            b: int = 2
            c: int = 3
            d: L0 = L0()

        @pytc.treeclass
        class L2:
            a: int = 10
            b: int = 20
            c: int = 30
            d: L1 = L1()

        >>> t = L2()

        >>> print(t.tree_diagram())
        # L2
        #     ├── a=10
        #     ├── b=20
        #     ├── c=30
        #     └── d=L1
        #         ├── a=1
        #         ├── b=2
        #         ├── c=3
        #         └── d=L0
        #             ├── a=1
        #             ├── b=2
        #             └── c=3

        # Let's mark `a` and `b`  in `L0` as non-differentiable
        # we can do this by using `where` mask

        >>> t = t.at["d"].at["d"].set(filter_nondiff(t.d.d, where= L0(a=True,b=True, c=False)))

        >>> print(t.tree_diagram())
        # L2
        #     ├── a=10
        #     ├── b=20
        #     ├── c=30
        #     └── d=L1
        #         ├── a=1
        #         ├── b=2
        #         ├── c=3
        #         └── d=L0
        #             ├*─ a=1
        #             ├*─ b=2
        #             └── c=3
        # note `*` indicates that the field is non-differentiable

    """

    # we use _pytree_map to add {nondiff:True} to a non-differentiable field metadata
    # this operation is done in-place and changes the tree structure
    # thus its not bundled with `.at[..]` as it will break composability

    # this dispatch filter the non-differentiable fields from the tree based
    # on the node value.
    # in essence any field with non-differentiable value is filtered out.

    def true_func(tree, field_item, node_item):
        new_field = _field(
            name=field_item.name,
            type=field_item.type,
            metadata={"static": True, "nondiff": True},
            repr=field_item.repr,
        )

        return {
            **tree.__undeclared_fields__,
            **{field_item.name: new_field},
        }

    return _pytree_map(
        tree,
        # condition is a lambda function that returns True if the field is non-differentiable
        cond=where or (lambda _, __, node_item: _is_nondiff(node_item)),
        # Extends the field metadata to add {nondiff:True}
        true_func=true_func,
        # keep the field as is if its differentiable
        false_func=lambda tree, __, ___: tree.__undeclared_fields__,
        attr_func=lambda _, __, ___: "__undeclared_fields__",
        # do not recurse if the field is `static`
        is_leaf=lambda _, field_item, __: field_item.metadata.get("static", False),
    )


def unfilter_nondiff(tree):
    """remove fields added by `filter_nondiff"""

    def true_func(tree, field_item, node_item):
        return {
            field_name: field_value
            for field_name, field_value in tree.__undeclared_fields__.items()
            if not field_value.metadata.get("nondiff", False)
        }

    return _pytree_map(
        tree,
        cond=lambda _, __, ___: True,
        true_func=true_func,
        false_func=lambda _, __, ___: {},
        attr_func=lambda _, __, ___: "__undeclared_fields__",
        is_leaf=lambda _, __, ___: False,
    )


def _field(name: str, type: type, metadata: dict[str, Any], repr: bool) -> Field:
    """field factory with option to add name, and type"""
    field_item = field(metadata=metadata, repr=repr)
    object.__setattr__(field_item, "name", name)
    object.__setattr__(field_item, "type", type)
    return field_item
