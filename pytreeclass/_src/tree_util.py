from __future__ import annotations

from collections.abc import Iterable
from dataclasses import Field, field
from types import FunctionType
from typing import Any, Callable

import jax.numpy as jnp
import jax.tree_util as jtu

import pytreeclass._src as src

PyTree = Any


def nondiff_field(**kwargs):
    """ignore from pytree computations"""
    return field(**{**kwargs, **{"metadata": {"static": True, "nondiff": True}}})


def frozen_field(**kwargs):
    """ignore from pytree computations"""
    return field(**{**kwargs, **{"metadata": {"static": True, "frozen": True}}})


def is_frozen_field(field_item: Field) -> bool:
    """check if field is frozen"""
    return field_item.metadata.get("frozen", False)


def is_nondiff_field(field_item: Field) -> bool:
    """check if field is strictly static"""
    return field_item.metadata.get("nondiff", False)


def is_static_field(field_item: Field) -> bool:
    """check if field is static"""
    return field_item.metadata.get("static", False)


def is_treeclass_frozen(tree):
    """assert if a treeclass is frozen"""
    if is_treeclass(tree):
        return all(is_frozen_field(f) for f in _tree_fields(tree).values())
    else:
        return False


def is_treeclass_nondiff(tree):
    """assert if a treeclass is static"""
    if is_treeclass(tree):
        return all(is_nondiff_field(f) for f in _tree_fields(tree).values())
    else:
        return False


def is_treeclass(tree):
    """check if a class is treeclass"""
    return hasattr(tree, "__immutable_pytree__")


def is_treeclass_leaf_bool(node):
    """assert if treeclass leaf is boolean (for boolen indexing)"""
    if isinstance(node, jnp.ndarray):
        return node.dtype == "bool"
    else:
        return isinstance(node, bool)


def is_treeclass_leaf(tree):
    """assert if a node is treeclass leaf"""
    if is_treeclass(tree):

        return is_treeclass(tree) and not any(
            [is_treeclass(tree.__dict__[fi.name]) for fi in _tree_fields(tree).values()]
        )
    else:
        return False


def is_treeclass_non_leaf(tree):
    return is_treeclass(tree) and not is_treeclass_leaf(tree)


def is_treeclass_equal(lhs, rhs):
    """Assert if two treeclasses are equal"""
    lhs_leaves, lhs_treedef = jtu.tree_flatten(lhs)
    rhs_leaves, rhs_treedef = jtu.tree_flatten(rhs)

    def is_node_equal(lhs_node, rhs_node):
        if isinstance(lhs_node, jnp.ndarray) and isinstance(rhs_node, jnp.ndarray):
            return jnp.array_equal(lhs_node, rhs_node)
        else:
            return lhs_node == rhs_node

    return (lhs_treedef == rhs_treedef) and all(
        [is_node_equal(lhs_leaves[i], rhs_leaves[i]) for i in range(len(lhs_leaves))]
    )


def tree_copy(tree):
    return jtu.tree_unflatten(*jtu.tree_flatten(tree)[::-1])


def _tree_mutate(tree):
    """Enable mutable behavior for a treeclass instance"""
    if is_treeclass(tree):
        object.__setattr__(tree, "__immutable_pytree__", False)
        for field_item in _tree_fields(tree).values():
            if hasattr(tree, field_item.name):
                _tree_mutate(getattr(tree, field_item.name))
    return tree


class _fieldDict(dict):
    """A dict used for `__pytree_structure__` attribute of a treeclass instance"""

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def _tree_structure(tree) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return dynamic and static fields of the pytree instance"""
    # this property scans the class fields and returns a tuple of two dicts (dynamic, static)
    # that mark the tree leaves seen by JAX computations and the static(tree structure) that are
    # not seen by JAX computations. the scanning is done if the instance is not frozen.
    # otherwise the cached values are returned.
    dynamic = _fieldDict()

    # undeclared fields are the fields that are not defined in the dataclass fields
    static = _fieldDict(__undeclared_fields__=tree.__undeclared_fields__)

    for field_item in _tree_fields(tree).values():
        if field_item.metadata.get("static", False):
            static[field_item.name] = getattr(tree, field_item.name)
        else:
            dynamic[field_item.name] = getattr(tree, field_item.name)
    return (dynamic, static)


def _tree_immutate(tree):
    """Enable immutable behavior for a treeclass instance"""
    if is_treeclass(tree):
        object.__setattr__(tree, "__immutable_pytree__", True)
        for field_item in _tree_fields(tree).values():
            if hasattr(tree, field_item.name):
                _tree_immutate(getattr(tree, field_item.name))
    return tree


def _tree_fields(tree):
    """Return a dictionary of all fields in the dataclass"""
    # in case of explicit treebase with no `param` then
    # its preferable not to create a new dict and just point to `__dataclass_fields__`
    # ** another feature of using an instance variable to store extra fields is that:
    # we can shadow the fields in the dataclass by creating a similarly named field in
    # the `undeclared_fields` instance variable, this avoids mutating the class fields.
    # For example in {**a,**b},  b keys will override a keys if they exist in both dicts.
    # this feature is used in functions that can set the `static` metadata
    # to specific instance fields (e.g. `filter_non_diff`)

    return (
        tree.__dataclass_fields__
        if len(tree.__undeclared_fields__) == 0
        else {**tree.__dataclass_fields__, **tree.__undeclared_fields__}
    )


# filtering nondifferentiable fields


def _is_nondiff(item: Any) -> bool:
    """check if tree is non-differentiable"""

    def _is_nondiff_item(node: Any):
        """check if node is non-differentiable"""
        # differentiable types
        if isinstance(node, (float, complex, src.tree_base._treeBase)):
            return False

        # differentiable array
        elif isinstance(node, jnp.ndarray) and jnp.issubdtype(node.dtype, jnp.inexact):
            return False

        return True

    if isinstance(item, Iterable):
        # if an iterable has at least one non-differentiable item
        # then the whole iterable is non-differentiable
        return any([_is_nondiff_item(item) for item in jtu.tree_leaves(item)])
    return _is_nondiff_item(item)


def filter_nondiff(
    tree: PyTree, where: Callable[[Field, Any], bool] | PyTree = _is_nondiff
):

    """filter non-differentiable fields from a treeclass instance

    Note:
        Mark fields as non-differentiable with adding metadata `dict(static=True,nondiff=True)`
        the way it is done is by adding a field in `__undeclared_fields__` to the treeclass instance
        that contains the non-differentiable fields.
        This is done to avoid mutating the original treeclass class .
        during the `tree_flatten`, tree_fields are the combination of
        {__dataclass_fields__, __undeclared_fields__} this means that a field defined
        in `__undeclared_fields__` with the same name as in __dataclass_fields__
        will override its properties, this is useful if you want to change the metadata
        of a field but don't want to change the original field definition defined in the class.

    Example:
        # here we try to optimize a differentiable value `b` that
        # is wrapped within a non-differentiable function `jax.nn.tanh`
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

    ** for non-`None` where, mark fields as non-differentiable by using `where` mask.

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

    def _nondiff_callable_map(tree: PyTree, where):
        for field_item in _tree_fields(tree).values():
            node_item = getattr(tree, field_item.name)

            if is_treeclass(node_item):
                _nondiff_callable_map(tree=node_item, where=where)

            elif where(node_item):
                new_field = nondiff_field(repr=field_item.repr)
                object.__setattr__(new_field, "name", field_item.name)
                object.__setattr__(new_field, "type", field_item.type)
                new_fields = {**tree.__undeclared_fields__, **{field_item.name: new_field}}  # fmt: skip
                object.__setattr__(tree, "__undeclared_fields__", new_fields)

        return tree

    def _nondiff_mask_map(tree: PyTree, where):
        for (lhs_field_item, rhs_field_item) in zip(
            _tree_fields(tree).values(), _tree_fields(where).values()
        ):
            lhs_node_item = getattr(tree, lhs_field_item.name)
            rhs_node_item = getattr(where, rhs_field_item.name)

            if is_treeclass(lhs_node_item):
                _nondiff_mask_map(tree=lhs_node_item, where=rhs_node_item)

            elif rhs_node_item:
                new_field = nondiff_field(repr=lhs_field_item.repr)
                object.__setattr__(new_field, "name", lhs_field_item.name)
                object.__setattr__(new_field, "type", lhs_field_item.type)
                new_fields = {**tree.__undeclared_fields__, **{lhs_field_item.name: new_field}}  # fmt: skip
                object.__setattr__(tree, "__undeclared_fields__", new_fields)

        return tree

    if isinstance(where, FunctionType):
        return _nondiff_callable_map(tree_copy(tree), where)
    elif isinstance(where, type(tree)):
        return _nondiff_mask_map(tree_copy(tree), where)
    else:
        raise TypeError(f"`where` must be a Callable or a {type(tree)}")


def unfilter_nondiff(tree):
    """remove fields added by `filter_nondiff"""

    def _recurse(tree):
        for field_item in _tree_fields(tree).values():
            node_item = getattr(tree, field_item.name)
            if is_treeclass(node_item):
                _recurse(tree=node_item)
            elif is_nondiff_field(field_item):
                del tree.__undeclared_fields__[field_item.name]
        return tree

    return _recurse(tree_copy(tree))


# filtering freezing


def tree_freeze(
    tree: PyTree, where: Callable[[Field, Any], bool] | PyTree = lambda _: True
):
    def _callable_map(tree: PyTree, where):
        for field_item in _tree_fields(tree).values():
            node_item = getattr(tree, field_item.name)

            if is_treeclass(node_item):
                _callable_map(tree=node_item, where=where)

            elif where(node_item):
                new_field = frozen_field(repr=field_item.repr)
                object.__setattr__(new_field, "name", field_item.name)
                object.__setattr__(new_field, "type", field_item.type)
                new_fields = {**tree.__undeclared_fields__, **{field_item.name: new_field}}  # fmt: skip
                object.__setattr__(tree, "__undeclared_fields__", new_fields)

        return tree

    def _mask_map(tree: PyTree, where):
        for (lhs_field_item, rhs_field_item) in zip(
            _tree_fields(tree).values(), _tree_fields(where).values()
        ):
            lhs_node_item = getattr(tree, lhs_field_item.name)
            rhs_node_item = getattr(where, rhs_field_item.name)

            if is_treeclass(lhs_node_item):
                _mask_map(tree=lhs_node_item, where=rhs_node_item)

            elif rhs_node_item:
                new_field = frozen_field(repr=lhs_field_item.repr)
                object.__setattr__(new_field, "name", lhs_field_item.name)
                object.__setattr__(new_field, "type", lhs_field_item.type)
                new_fields = {**tree.__undeclared_fields__, **{lhs_field_item.name: new_field}}  # fmt: skip
                object.__setattr__(tree, "__undeclared_fields__", new_fields)

        return tree

    if isinstance(where, FunctionType):
        return _callable_map(tree_copy(tree), where)
    elif isinstance(where, type(tree)):
        return _mask_map(tree_copy(tree), where)
    else:
        raise TypeError(f"`where` must be a Callable or a {type(tree)}")


def tree_unfreeze(tree):
    """remove fields added by `tree_freeze"""

    def _recurse(tree):
        for field_item in _tree_fields(tree).values():
            node_item = getattr(tree, field_item.name)
            if is_treeclass(node_item):
                _recurse(tree=node_item)
            elif is_frozen_field(field_item):
                del tree.__undeclared_fields__[field_item.name]
        return tree

    return _recurse(tree_copy(tree))
