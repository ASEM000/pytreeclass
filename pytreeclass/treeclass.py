from __future__ import annotations

# from dataclasses import dataclass, field
import dataclasses
import inspect
from types import MappingProxyType
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from pytreeclass._src.tree_base import _treeBase
from pytreeclass._src.tree_indexer import _treeIndexer
from pytreeclass._src.tree_op import _treeOp
from pytreeclass._src.tree_pretty import _treePretty
from pytreeclass._src.tree_util import _mutable


class ImmutableInstanceError(Exception):
    pass


def field(
    *, nondiff: bool = False, frozen: bool = False, **kwargs
) -> dataclasses.Field:
    """Similar to dataclasses.field but with additional arguments
    Args:
        nondiff: if True, the field will not be differentiated
        frozen: if True, the field will be frozen
        **kwargs: additional arguments to pass to dataclasses.field
    """
    metadata = kwargs.pop("metadata", {})
    metadata["nondiff"] = nondiff
    metadata["frozen"] = frozen
    metadata["static"] = nondiff or frozen
    return dataclasses.field(metadata=metadata, **kwargs)


def is_frozen_field(field_item: dataclasses.Field) -> bool:
    """check if field is frozen"""
    return isinstance(field_item, dataclasses.Field) and field_item.metadata.get(
        "frozen", False
    )


def is_nondiff_field(field_item: dataclasses.Field) -> bool:
    """check if field is strictly static"""
    return isinstance(field_item, dataclasses.Field) and field_item.metadata.get(
        "nondiff", False
    )


def fields(tree):
    if len(tree.__undeclared_fields__) == 0:
        return tree.__dataclass_fields__.values()
    return {**tree.__dataclass_fields__, **tree.__undeclared_fields__}.values()


def treeclass(*args, **kwargs):
    def immutable_setter(tree, key: str, value: Any) -> None:

        if tree.__immutable_pytree__:
            msg = f"Cannot set {key}={value!r}. Use `.at['{key}'].set({value!r})` instead."
            raise ImmutableInstanceError(msg)

        object.__setattr__(tree, key, value)

        if (isinstance(value, _treeBase)) and (
            key not in [f.name for f in fields(tree)]
        ):
            # create field
            field_value = field()

            object.__setattr__(field_value, "name", key)
            object.__setattr__(field_value, "type", type(value))

            # register it to class
            new_fields = {**tree.__undeclared_fields__, **{key: field_value}}  # fmt: skip
            object.__setattr__(tree, "__undeclared_fields__", MappingProxyType(new_fields))  # fmt: skip

    def class_wrapper(cls):

        if "__setattr__" in vars(cls):
            raise AttributeError("`treeclass` cannot be applied to class with `__setattr__` method.")  # fmt: skip

        dCls = dataclasses.dataclass(
            init="__init__" not in vars(cls),
            repr=False,  # repr is handled by _treePretty
            eq=False,  # eq is handled by _treeOpBase
            unsafe_hash=False,  # hash is handled by _treeOpBase
            order=False,  # order is handled by _treeOpBase
            frozen=False,  # frozen is `immutable_setter`
        )(cls)

        bases = (dCls, _treeIndexer, _treeOp, _treePretty, _treeBase)

        attrs_keys = ("__setattr__", "__immutable_pytree__", "__undeclared_fields__")
        attrs_vals = (immutable_setter, False, MappingProxyType({}))
        new_cls = type(cls.__name__, bases, dict(zip(attrs_keys, attrs_vals)))  # fmt: skip

        # temporarily mutate the tree instance to execute the __init__ method
        # without raising `__immutable_treeclass__` error
        # then restore the tree original immutable behavior after the function is called
        # _mutable can be applied to any class method that is decorated with @treeclass
        # to temporarily make the class mutable
        # however, it is not recommended to use it outside of __init__ method
        new_cls.__init__ = _mutable(new_cls.__init__)

        return jax.tree_util.register_pytree_node_class(new_cls)

    if len(args) == 1 and inspect.isclass(args[0]):
        # no args are passed to the decorator (i.e. @treeclass)
        return class_wrapper(args[0])

    raise TypeError(f"`treeclass` input must be of `class` type. Found {(*args, *kwargs)}.")  # fmt: skip


def is_treeclass(tree):
    """check if a class is treeclass"""
    return hasattr(tree, "__immutable_pytree__")


def is_treeclass_frozen(tree):
    """assert if a treeclass is frozen"""
    if is_treeclass(tree):
        field_items = fields(tree)
        if len(field_items) > 0:
            return all(is_frozen_field(f) for f in field_items)
    return False


def is_treeclass_nondiff(tree):
    """assert if a treeclass is static"""
    if is_treeclass(tree):
        field_items = fields(tree)
        if len(field_items) > 0:
            return all(is_nondiff_field(f) for f in field_items)
    return False


def is_treeclass_leaf_bool(node):
    """assert if treeclass leaf is boolean (for boolen indexing)"""
    if isinstance(node, jnp.ndarray):
        return node.dtype == "bool"
    return isinstance(node, bool)


def is_treeclass_leaf(tree):
    """assert if a node is treeclass leaf"""
    if is_treeclass(tree):

        return is_treeclass(tree) and not any(
            [is_treeclass(getattr(tree, fi.name)) for fi in fields(tree)]
        )
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
        return lhs_node == rhs_node

    return (lhs_treedef == rhs_treedef) and all(
        [is_node_equal(lhs_leaves[i], rhs_leaves[i]) for i in range(len(lhs_leaves))]
    )
