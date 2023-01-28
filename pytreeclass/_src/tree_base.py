from __future__ import annotations

import dataclasses as dc
import functools as ft
from typing import Any

import jax.tree_util as jtu
import numpy as np

from pytreeclass._src.tree_freeze import (
    _FrozenWrapper,
    _NonDiffField,
    _set_dataclass_frozen,
    _UnfrozenWrapper,
)
from pytreeclass._src.tree_indexer import _TreeAtIndexer
from pytreeclass._src.tree_operator import _TreeOperator
from pytreeclass.tree_viz.tree_pprint import tree_repr, tree_str

PyTree = Any

# The name of an attribute on the class where we store the Field
_FIELDS = "__dataclass_fields__"

# The name of an attribute on the class that stores the parameters to
# @dataclass.
_PARAMS = "__dataclass_params__"
_PARAMS_SLOTS = dc._DataclassParams.__slots__


class _DataclassParams(dc._DataclassParams):
    # dataclass params frozen is used to mark the tree as immutable
    # in treeclass we set it as an instance variable to mark individual trees as immutable
    # this means it is part of the instance treedef/metadata.
    # in order to make same instance of _DataclassParams equal to each other, we need to override __eq__
    def __eq__(self, rhs):
        if not isinstance(rhs, _DataclassParams):
            return False
        return f"{self}!" == f"{rhs}!"


def field(
    *,
    nondiff: bool = False,
    default: Any = dc.MISSING,
    default_factory: Any = dc.MISSING,
    init: bool = True,
    repr: bool = True,
    hash: bool = None,
    compare: bool = True,
    metadata: Any = None,
    kw_only: Any = dc.MISSING,
) -> dc.Field:
    """Create a field for a dataclass

    Args:
        nondiff: if True, the field will be non-differentiable (i.e. will not be updated by optimizers)
        default: default value of the field
        default_factory: default factory of the field
        init: if True, the field will be initialized
        repr: if True, the field will be included in the repr
        hash: if True, the field will be included in the hash
        compare: if True, the field will be included in the comparison
        metadata: metadata of the field
        kw_only: if True, the field will be keyword only
    """
    params = dict(
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        hash=hash,
        compare=compare,
        metadata=metadata,
    )

    if "kw_only" in dir(dc.Field):
        params["kw_only"] = kw_only

    if default is not dc.MISSING and default_factory is not dc.MISSING:
        raise ValueError("cannot specify both default and default_factory")

    return _NonDiffField(**params) if nondiff else dc.Field(**params)


def _setattr(tree: PyTree, key: str, value: Any) -> None:
    """Set the attribute of the tree if the tree is not frozen"""
    if getattr(tree, _PARAMS).frozen is True:
        msg = f"Cannot set {key}={value!r}. Use `.at['{key}'].set({value!r})` instead."
        raise dc.FrozenInstanceError(msg)

    object.__setattr__(tree, key, value)

    if dc.is_dataclass(value) and (key not in [f.name for f in dc.fields(tree)]):
        # register the field to the tree to mark it as a leaf
        field = dc.field()
        field.name = key
        field.type = type(value)
        field._field_type = dc._FIELD

        # register it to dataclass fields
        getattr(tree, _FIELDS)[key] = field


def _delattr(tree, key: str) -> None:
    """Delete the attribute of the  if tree is not frozen"""
    # Delete if __dataclass_params__.frozen is False otherwise raise dc.FrozenInstanceError"""
    if getattr(tree, _PARAMS).frozen is True:
        raise dc.FrozenInstanceError(f"Cannot delete {key}.")
    object.__delattr__(tree, key)


def _new_wrapper(new_func, params, fields):
    @ft.wraps(new_func)
    def new(cls, *a, **k) -> PyTree:
        # set the params and fields as instance variables
        tree = object.__new__(cls)
        tree.__dict__[_PARAMS] = params
        tree.__dict__[_FIELDS] = fields
        return tree

    return new


def _init_wrapper(init_func):
    @ft.wraps(init_func)
    def init(self, *a, **k) -> None:
        self = _set_dataclass_frozen(self, frozen=False)
        output = init_func(self, *a, **k)
        self = _set_dataclass_frozen(self, frozen=True)
        return output

    return init


def _copy(tree: PyTree) -> PyTree:
    """Return a copy of the tree"""
    return jtu.tree_unflatten(*jtu.tree_flatten(tree)[::-1])


def _flatten(tree) -> tuple[Any, tuple[str, dict[str, Any]]]:
    """Flatten rule for `jax.tree_flatten`"""
    # in essence anything not declared as a dataclass fields will be considered static
    static, dynamic = dict(tree.__dict__), dict()
    # avoid mutating the original dict by making a copy of dataclass fields
    static[_FIELDS] = dict(static[_FIELDS])

    for field in static[_FIELDS].values():
        if isinstance(field, _NonDiffField):
            # permenantly non-differentiable fields case
            continue

        if not isinstance(field, _FrozenWrapper):
            # non-wrapper differentiable fields case
            dynamic[field.name] = static.pop(field.name)
            continue

        # handle wrappers
        # expose the `_FrozenWrapper` to `tree_unflatten by moving the wrapper
        static[_FIELDS][field.name] = field.__wrapped__
        if isinstance(static[field.name], _UnfrozenWrapper):
            # `_UnfrozenWrapper` value and `_FrozenWrapper` field case
            # this is the case encountered when unfreezing a tree
            dynamic[field.name] = static.pop(field.name).__wrapped__
        else:
            # frozen value and frozen field -> move wrapper to value
            # _FrozenWrapper cause an error if the value inside is jnp.array
            dynamic[field.name] = _FrozenWrapper(static.pop(field.name))

    return dynamic.values(), (dynamic.keys(), static)


def _unflatten(cls, treedef, leaves):
    """Unflatten rule for `jax.tree_unflatten`"""
    tree = object.__new__(cls)  # do not call cls constructor
    static = treedef[1]
    dynamic = dict(zip(treedef[0], leaves))

    # handle wrappers
    # a wrapper is used in `tree_freeze`/`tree_unfreeze` to mark a leaf as frozen/unfrozen
    for field in static[_FIELDS].values():
        if field.name in dynamic:
            # dynamic fields case
            if isinstance(dynamic[field.name], _FrozenWrapper):
                # this is the case where `tree_freeze` is applied to a leaf
                # thus we move the frozen wrapper from leaf to the corresponding field
                dynamic[field.name] = dynamic[field.name].__wrapped__
                static[_FIELDS][field.name] = _FrozenWrapper(field)

            elif isinstance(dynamic[field.name], _UnfrozenWrapper):
                # this is the case where `tree_unfreeze` is applied to a leaf (redunant)
                dynamic[field.name] = dynamic[field.name].__wrapped__
            continue

        # static fields case
        if isinstance(static[field.name], _FrozenWrapper):
            # `tree_freeze` is applied to a static leaf node by .at[] methods (redunant)
            static[field.name] = static[field.name].__wrapped__
        elif isinstance(static[field.name], _UnfrozenWrapper):
            # `tree_unfreeze` is applied to a static leaf node by .at[] methods
            static[field.name] = static[field.name].__wrapped__
            static[_FIELDS][field.name] = field.__wrapped__

    tree.__dict__.update(static)
    tree.__dict__.update(dynamic)
    return tree


def treeclass(cls):
    """Decorator to convert a class to a `treeclass`

    Example:
        >>> @treeclass
        ... class Tree:
        ...     x: float
        ...     y: float
        ...     z: float
        >>> tree = Tree(1, 2, 3)
        >>> tree
        Tree(x=1, y=2, z=3)

    Args:
        cls: class to be converted to a `treeclass`

    Returns:
        `treeclass` of the input class

    Raises:
        TypeError: if the input class is not a `class`
    """
    dcls = dc.dataclass(init=("__init__" not in vars(cls)), repr=False, eq=False)(cls)
    params = getattr(dcls, _PARAMS)
    params = _DataclassParams(**{k: getattr(params, k) for k in _PARAMS_SLOTS})
    fields = dcls.__dict__.get(_FIELDS)

    attrs = dict(
        __new__=_new_wrapper(dcls.__new__, params, fields),
        __init__=_init_wrapper(dcls.__init__),  # make it mutable during initialization
        __setattr__=_setattr,  # disable direct attribute setting unless __immutable_treeclass__ is False
        __delattr__=_delattr,  # disable direct attribute deletion unless __immutable_treeclass__ is False
        __repr__=tree_repr,  # pretty print the tree representation
        __str__=tree_str,  # pretty print the tree
        __copy__=_copy,  # copy the tree
        tree_flatten=_flatten,  # jax.tree_util.tree_flatten rule
        tree_unflatten=classmethod(_unflatten),  # jax.tree_util.tree_unflatten rule
    )

    dcls = type(cls.__name__, (dcls, _TreeAtIndexer, _TreeOperator), attrs)

    return jtu.register_pytree_node_class(dcls)


def is_treeclass(cls_or_instance):
    """Check if the input is a `treeclass`"""
    if not dc.is_dataclass(cls_or_instance):
        # treeclass must be a dataclass
        return False

    if isinstance(cls_or_instance, type):
        # check if the input is a class
        # then check if the class is a subclass of `_TreeAtIndexer` and `_TreeOperator`
        return issubclass(cls_or_instance, _TreeAtIndexer) and issubclass(
            cls_or_instance, _TreeOperator
        )

    # finally check if the input is an instance of `_TreeAtIndexer` and `_TreeOperator`
    return isinstance(cls_or_instance, _TreeAtIndexer) and isinstance(
        cls_or_instance, _TreeOperator
    )


def is_treeclass_equal(lhs, rhs):
    """Assert if two treeclasses are equal"""
    lhs_leaves, lhs_treedef = jtu.tree_flatten(lhs)
    rhs_leaves, rhs_treedef = jtu.tree_flatten(rhs)

    def is_equal(lhs_node, rhs_node):
        if hasattr(lhs_node, "dtype") and hasattr(rhs_node, "shape"):
            if hasattr(lhs_node, "dtype") and hasattr(rhs_node, "shape"):
                return np.all(lhs_node == rhs_node)
            return False
        return lhs_node == rhs_node

    if not (lhs_treedef == rhs_treedef):
        return False

    for (lhs, rhs) in zip(lhs_leaves, rhs_leaves):
        if not is_equal(lhs, rhs):
            return False
    return True
