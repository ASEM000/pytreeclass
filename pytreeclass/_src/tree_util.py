from __future__ import annotations

from collections.abc import Iterable
from dataclasses import Field, field
from types import FunctionType, MappingProxyType
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
    return isinstance(field_item, Field) and field_item.metadata.get("frozen", False)


def is_nondiff_field(field_item: Field) -> bool:
    """check if field is strictly static"""
    return isinstance(field_item, Field) and field_item.metadata.get("nondiff", False)


def is_static_field(field_item: Field) -> bool:
    """check if field is static"""
    return isinstance(field_item, Field) and field_item.metadata.get("static", False)


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


def _tree_immutate(tree):
    """Enable immutable behavior for a treeclass instance"""
    if is_treeclass(tree):
        object.__setattr__(tree, "__immutable_pytree__", True)
        for field_item in _tree_fields(tree).values():
            if hasattr(tree, field_item.name):
                _tree_immutate(getattr(tree, field_item.name))
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


def _append_field(
    tree: PyTree,
    where: Callable[[Field, Any], bool] | PyTree,
    replacing_field: Field = field,
) -> PyTree:
    """append a dataclass field to a treeclass `__undeclared_fields__`

    Args:
        tree (PyTree): tree to append field to
        where (Callable[[Field, Any], bool] | PyTree, optional): where to append field. Defaults to _is_nondiff.
        replacing_field (Field, optional): type of field. Defaults to field.

    Note:
        This is the base mechanism for controlling the static/dynamic behavior of a treeclass instance.

        during the `tree_flatten`, tree_fields are the combination of
        {__dataclass_fields__, __undeclared_fields__} this means that a field defined
        in `__undeclared_fields__` with the same name as in __dataclass_fields__
        will override its properties, this is useful if you want to change the metadata
        of a field but don't want to change the original field definition defined in the class.
    """

    def _callable_map(tree: PyTree, where: Callable[[Field, Any], bool]) -> PyTree:
        # filter based on a conditional callable
        for field_item in _tree_fields(tree).values():
            node_item = getattr(tree, field_item.name)

            if is_treeclass(node_item):
                _callable_map(tree=node_item, where=where)

            elif where(node_item):
                new_field = replacing_field(repr=field_item.repr)
                object.__setattr__(new_field, "name", field_item.name)
                object.__setattr__(new_field, "type", field_item.type)
                new_fields = {**tree.__undeclared_fields__, **{field_item.name: new_field}}  # fmt: skip
                object.__setattr__(tree, "__undeclared_fields__", MappingProxyType(new_fields))  # fmt: skip

        return tree

    def _mask_map(tree: PyTree, where: PyTree) -> PyTree:
        # filter based on a mask of the same type as `tree`
        for (lhs_field_item, rhs_field_item) in zip(
            _tree_fields(tree).values(), _tree_fields(where).values()
        ):
            lhs_node_item = getattr(tree, lhs_field_item.name)
            rhs_node_item = getattr(where, rhs_field_item.name)

            if is_treeclass(lhs_node_item):
                _mask_map(tree=lhs_node_item, where=rhs_node_item)

            elif jnp.all(rhs_node_item):
                new_field = replacing_field(repr=lhs_field_item.repr)
                object.__setattr__(new_field, "name", lhs_field_item.name)
                object.__setattr__(new_field, "type", lhs_field_item.type)
                new_fields = {**tree.__undeclared_fields__, **{lhs_field_item.name: new_field}}  # fmt: skip
                object.__setattr__(tree, "__undeclared_fields__", MappingProxyType(new_fields))  # fmt: skip

        return tree

    if isinstance(where, FunctionType):
        return _callable_map(tree_copy(tree), where)
    elif isinstance(where, type(tree)):
        return _mask_map(tree_copy(tree), where)
    else:
        raise TypeError(f"`where` must be a Callable or a {type(tree)}")


def _unappend_field(tree: PyTree, cond: Callable[[Field], bool]) -> PyTree:
    """remove a dataclass field from `__undeclared_fields__` added if some condition is met"""

    def _recurse(tree):
        for field_item in _tree_fields(tree).values():
            node_item = getattr(tree, field_item.name)
            if is_treeclass(node_item):
                _recurse(tree=node_item)
            elif cond(field_item):
                new_fields = dict(tree.__undeclared_fields__)
                new_fields.pop(field_item.name)
                object.__setattr__(
                    tree, "__undeclared_fields__", MappingProxyType(new_fields)
                )
        return tree

    return _recurse(tree_copy(tree))


def filter_nondiff(
    tree: PyTree, where: PyTree | Callable[[Field, Any], bool] = _is_nondiff
) -> PyTree:
    return _append_field(tree=tree, where=where, replacing_field=nondiff_field)


def unfilter_nondiff(tree):
    return _unappend_field(tree, is_nondiff_field)


def tree_freeze(
    tree: PyTree, where: PyTree | Callable[[Field, Any], bool] = lambda _: True
) -> PyTree:
    return _append_field(tree=tree, where=where, replacing_field=frozen_field)


def tree_unfreeze(tree):
    return _unappend_field(tree, is_frozen_field)
