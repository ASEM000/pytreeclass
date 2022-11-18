from __future__ import annotations

import dataclasses

import jax.numpy as jnp
import numpy as np


def field(*, nondiff: bool = False, **k) -> dataclasses.Field:
    """Similar to dataclasses.field but with additional arguments
    Args:
        nondiff: if True, the field will not be differentiated or modified by any filtering operations
        name: name of the field. Will be inferred from the variable name if its assigned to a class attribute.
        type: type of the field. Will be inferred from the variable type if its assigned to a class attribute.
        **k: additional arguments to pass to dataclasses.field
    """
    metadata = k.pop("metadata", {})
    if nondiff is True:
        metadata["static"] = "nondiff"

    return dataclasses.field(metadata=metadata, **k)


def field_copy(field_item):
    """Copy a dataclass field item."""
    new_field = dataclasses.field(
        default=field_item.default,
        default_factory=field_item.default_factory,
        init=field_item.init,
        repr=field_item.repr,
        hash=field_item.hash,
        compare=field_item.compare,
        metadata=field_item.metadata,
    )

    object.__setattr__(new_field, "name", field_item.name)
    object.__setattr__(new_field, "type", field_item.type)
    object.__setattr__(new_field, "_field_type", field_item._field_type)

    return new_field


def is_field_nondiff(field_item: dataclasses.Field) -> bool:
    """check if field is strictly static"""
    return (
        isinstance(field_item, dataclasses.Field)
        and field_item.metadata.get("static", False) == "nondiff"
    )


def is_field_frozen(field_item: dataclasses.Field) -> bool:
    """check if field is strictly static"""
    return (
        isinstance(field_item, dataclasses.Field)
        and field_item.metadata.get("static", False) == "frozen"
    )


def is_dataclass_fields_nondiff(tree):
    """assert if a dataclass is static"""
    if dataclasses.is_dataclass(tree):
        field_items = dataclasses.fields(tree)
        if len(field_items) > 0:
            return all(is_field_nondiff(f) for f in field_items)
    return False


def is_dataclass_fields_frozen(tree):
    """assert if a dataclass is static"""
    if dataclasses.is_dataclass(tree):
        field_items = dataclasses.fields(tree)
        if len(field_items) > 0:
            return all(is_field_frozen(f) for f in field_items)
    return False


def is_dataclass_leaf_bool(node):
    """assert if dataclass leaf is boolean (for boolen indexing)"""
    if isinstance(node, (jnp.ndarray, np.ndarray)):
        return node.dtype == "bool"
    return isinstance(node, bool)


def is_dataclass_leaf(tree):
    """assert if a node is dataclass leaf"""
    if dataclasses.is_dataclass(tree):

        return dataclasses.is_dataclass(tree) and not any(
            [
                dataclasses.is_dataclass(getattr(tree, fi.name))
                for fi in dataclasses.fields(tree)
            ]
        )
    return False


def is_dataclass_non_leaf(tree):
    return dataclasses.is_dataclass(tree) and not is_dataclass_leaf(tree)
