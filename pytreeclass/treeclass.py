from __future__ import annotations

# from dataclasses import dataclass, field
import dataclasses
import inspect
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu

from pytreeclass._src.tree_base import _treeBase
from pytreeclass._src.tree_indexer import _treeIndexer
from pytreeclass._src.tree_op import _treeOp
from pytreeclass._src.tree_pretty import _treePretty
from pytreeclass._src.tree_util import _mutable

# field related ------------------------------------------------------------------------------------------------------ #


def field(
    *, nondiff: bool = False, frozen: bool = False, **kwargs
) -> dataclasses.Field:
    """Similar to dataclasses.field but with additional arguments
    Args:
        nondiff: if True, the field will not be differentiated
        frozen: if True, the field will be frozen
        name: name of the field. Will be inferred from the variable name if its assigned to a class attribute.
        type: type of the field. Will be inferred from the variable type if its assigned to a class attribute.
        **kwargs: additional arguments to pass to dataclasses.field
    """
    if frozen and nondiff:
        raise ValueError("Cannot be both frozen and nondiff")

    metadata = kwargs.pop("metadata", {})
    if nondiff is True:
        metadata["nondiff"] = metadata["static"] = True
    elif frozen is True:
        metadata["frozen"] = metadata["static"] = True

    return dataclasses.field(metadata=metadata, **kwargs)


def fields(tree):
    if len(tree.__treeclass_fields__) == 0:
        return tree.__dataclass_fields__.values()
    return tuple({**tree.__dataclass_fields__, **tree.__treeclass_fields__}.values())


def is_field_frozen(field_item: dataclasses.Field) -> bool:
    """check if field is frozen"""
    return isinstance(field_item, dataclasses.Field) and field_item.metadata.get(
        "frozen", False
    )


def is_field_nondiff(field_item: dataclasses.Field) -> bool:
    """check if field is strictly static"""
    return isinstance(field_item, dataclasses.Field) and field_item.metadata.get(
        "nondiff", False
    )


# frozen methods ----------------------------------------------------------------------------------------------------- #
class ImmutableInstanceError(Exception):
    pass


def _immutable_setter(tree, key: str, value: Any) -> None:

    if is_treeclass_immutable(tree):
        msg = f"Cannot set {key}={value!r}. Use `.at['{key}'].set({value!r})` instead."
        raise ImmutableInstanceError(msg)

    object.__setattr__(tree, key, value)

    if (isinstance(value, _treeBase)) and (key not in [f.name for f in fields(tree)]):
        # create field
        field_item = field()

        object.__setattr__(field_item, "name", key)
        object.__setattr__(field_item, "type", type(value))

        # register it to class
        new_fields = {**tree.__treeclass_fields__, **{key: field_item}}
        object.__setattr__(tree, "__treeclass_fields__", new_fields)


def _immutable_delattr(tree, key: str) -> None:
    if is_treeclass_immutable(tree):
        raise ImmutableInstanceError(f"Cannot delete {key}.")
    object.__delattr__(tree, key)


# treeclass related ---------------------------------------------------------------------------------------------------#


def _check_and_return_cls(cls):
    # check if the input is a class
    if not inspect.isclass(cls):
        msg = f"Input must be of `class` type. Found {cls}."
        raise TypeError(msg)

    # check if the class does not have setattr
    if "__setattr__" in vars(cls):
        msg = f"Cannot overwrite attribute __setattr__ on class {cls.__name__}"
        raise TypeError(msg)

    # check if the class does not have delattr
    if "__delattr__" in vars(cls):
        msg = f"Cannot overwrite attribute __delattr__ on class {cls.__name__}"
        raise TypeError(msg)

    return cls


def treeclass(cls):
    """Decorator to make a class a treeclass"""

    dcls = dataclasses.dataclass(
        init="__init__" not in vars(cls),  # if __init__ is defined, do not overwrite it
        repr=False,  # repr is handled by _treePretty
        eq=False,  # eq is handled by _treeOp
        order=False,  # order is handled by _treeOp
        unsafe_hash=False,  # unsafe_hash is handled by _treeOp
        frozen=False,  # frozen is handled by _immutable_setter/_immutable_delattr
    )(_check_and_return_cls(cls))

    attrs = dict(
        __setattr__=_immutable_setter,  # disable direct attribute setting unless __immutable_treeclass__ is False
        __delattr__=_immutable_delattr,  # disable direct attribute deletion unless __immutable_treeclass__ is False
        __treeclass_fields__=dict(),  # fields that are not in dataclass
        __immutable_treeclass__=True,  # flag to disable direct attribute setting/deletion
    )

    bases = (dcls, _treeBase, _treeIndexer, _treeOp, _treePretty)
    new_cls = type(cls.__name__, bases, attrs)

    # temporarily make the class mutable during class creation
    new_cls.__init__ = _mutable(new_cls.__init__)

    return jax.tree_util.register_pytree_node_class(new_cls)


def is_treeclass_immutable(tree):
    """assert if a treeclass is immutable"""
    return is_treeclass(tree) and tree.__immutable_treeclass__


def is_treeclass(tree):
    """check if a class is treeclass"""
    return hasattr(tree, "__immutable_treeclass__")


def is_treeclass_frozen(tree):
    """assert if a treeclass is frozen"""
    if is_treeclass(tree):
        field_items = fields(tree)
        if len(field_items) > 0:
            return all(is_field_frozen(f) for f in field_items)
    return False


def is_treeclass_nondiff(tree):
    """assert if a treeclass is static"""
    if is_treeclass(tree):
        field_items = fields(tree)
        if len(field_items) > 0:
            return all(is_field_nondiff(f) for f in field_items)
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
