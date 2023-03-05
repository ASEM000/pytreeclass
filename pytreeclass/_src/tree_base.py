from __future__ import annotations

import copy
import functools as ft
import math
import operator as op
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from pytreeclass._src.tree_decorator import (
    _FIELD_MAP,
    _FROZEN,
    _NOT_SET,
    _POST_INIT,
    _WRAPPED,
    Field,
    _apply_callbacks,
    _generate_field_map,
    _generate_init,
)
from pytreeclass._src.tree_freeze import _tree_hash, _tree_map_unwrap
from pytreeclass._src.tree_indexer import _tree_copy, _tree_indexer, bcmap
from pytreeclass._src.tree_pprint import tree_repr, tree_str

PyTree = Any


def _flatten(tree) -> tuple[Any, tuple[str, dict[str, Any]]]:
    """Flatten rule for `jax.tree_flatten`"""
    # in essence anything not declared in dataclass fields will be considered static
    static, dynamic = dict(vars(tree)), dict()
    for key in getattr(tree, _FIELD_MAP):
        dynamic[key] = static.pop(key)

    return dynamic.values(), (dynamic.keys(), static)


def _unflatten(klass, treedef, leaves):
    """Unflatten rule for `jax.tree_unflatten`"""
    # call the wrapped `__new__` method (non-field initializer)
    tree = getattr(klass.__new__, _WRAPPED)(klass)
    vars(tree).update(treedef[1])
    vars(tree).update(zip(treedef[0], leaves))
    return tree


@ft.lru_cache(maxsize=None)
def _register_treeclass(klass):
    # register a treeclass only once by using `lru_cache`
    # there are two cases where a class is registered more than once:
    # first, when a class is decorated with `treeclass` more than once (e.g. `treeclass(treeclass(Class))`)
    # second when a class is decorated with `treeclass` and has a parent class that is also decorated with `treeclass`
    # in that case `__init_subclass__` registers the class before the decorator registers it.
    # this can be also be done using metaclass that registers the class on initialization
    # but we are trying to stay away from deep magic.
    jtu.register_pytree_node(klass, _flatten, ft.partial(_unflatten, klass))
    return klass


def _getattr_wrapper(getattr_func):
    @ft.wraps(getattr_func)
    def getattr_method(tree, key: str) -> Any:
        # this current approach replaces the older metdata based approach
        # that is used in `dataclasses`-based libraries like `flax.struct.dataclass` and v0.1 of `treeclass`.
        # the metadata approach is defined at class variable and can not be changed at runtime while the current
        # approach is more flexible because it can be changed at runtime using `tree_map` or by using `at`
        # moreover, metadata-based approach falls short when handling nested data structures values.
        # for example if a field value is a tuple of (1, 2, 3), then metadata-based approach will only be able
        # to freeze the whole tuple, but not its elements.
        # with the current approach, we can use `tree_map`/ or direct application to freeze certain tuple elements
        # and leave the rest of the tuple as is.
        # another pro of the current approach is that the field metadata is not checked during flattening/unflattening
        # so in essence, it's more efficient than the metadata-based approach during applying `jax` transformations
        # that flatten/unflatten the tree.
        # Example: when fetching `tree.a` it will be unwrapped
        # >>> @pytc.treeclass
        # ... class Tree:
        # ...    a:int = pytc.freeze(1)
        # >>> tree = Tree()
        # >>> tree
        # Tree(a=#1)  # frozen value is displayed in the repr with a prefix `#`
        # >>> tree.a
        # 1  # the value is unwrapped when accessed directly
        value = getattr_func(tree, key)

        if key in getattr_func(tree, "__dict__"):
            # unwrap non-`TreeClass`` instance variables
            # so the getattr will always return unwrapped values.
            return _tree_map_unwrap(value)
        # return the value as is if it is not an instance variable
        return value

    return getattr_method


def _setattr(tree: PyTree, key: str, value: Any) -> None:
    """Set the attribute of the tree if the tree is not frozen"""
    if getattr(tree, _FROZEN):
        msg = f"Cannot set {key}={value!r}. Use `.at['{key}'].set({value!r})` instead."
        raise AttributeError(msg)

    vars(tree)[key] = value

    if hasattr(value, _FIELD_MAP) and (key not in getattr(tree, _FIELD_MAP)):
        field = Field(name=key, type=type(value))  # type: ignore
        # register it to field map, to avoid re-registering it in field_map
        getattr(tree, _FIELD_MAP)[key] = field


def _delattr(tree, key: str) -> None:
    """Delete the attribute of the  if tree is not frozen"""
    if getattr(tree, _FROZEN):
        raise AttributeError(f"Cannot delete {key}.")
    del vars(tree)[key]


def _new_wrapper(new_func):
    @ft.wraps(new_func)
    def new_method(klass, *_, **__) -> PyTree:
        self = new_func(klass)
        for field in getattr(klass, _FIELD_MAP).values():
            if field.default is not _NOT_SET:
                vars(self)[field.name] = field.default
            elif field.default_factory is not None:
                vars(self)[field.name] = field.default_factory()
        return self

    # wrap the original `new_func`, to use it later in `tree_unflatten`
    # to avoid repeating iterating over fields and setting default values
    setattr(new_method, _WRAPPED, new_func)
    return new_method


def _init_sub_wrapper(init_subclass_func):
    @classmethod
    @ft.wraps(init_subclass_func)
    def _init_subclass(klass) -> None:
        # Non-decorated subclasses uses the base `treeclass` leaves only
        # this behavior is aligned with `dataclasses` not registering non-decorated
        # subclasses dataclass fields. for example:
        # >>> @treeclass
        # ... class A:
        # ...   a:int=1
        # >>> class B(A):
        # ...    b:int=2
        # >>> tree = B()
        # >>> jax.tree_leaves(tree)
        # [1]
        # however if we decorate `B` with `treeclass` then the fields of `B` will be registered as leaves
        # >>> @treeclass
        # ... class B(A):
        # ...    b:int=2
        # >>> tree = B()
        # >>> jax.tree_leaves(tree)
        # [1, 2]
        # this behavior is different from `flax.struct.dataclass`
        # as it does not register non-decorated subclasses field that inherited from decorated subclasses.
        init_subclass_func()
        return _register_treeclass(klass)

    return _init_subclass


def _init_wrapper(init_func):
    @ft.wraps(init_func)
    def init_method(self, *a, **k) -> None:
        vars(self)[_FROZEN] = False
        output = init_func(self, *a, **k)

        # call callbacks on fields that are initialized
        vars(self)[_FROZEN] = False
        _apply_callbacks(self, init=True)

        # in case __post_init__ is defined then call it
        # after the tree is initialized
        # here, we assume that __post_init__ is a method
        if hasattr(self, _POST_INIT):
            # in case we found post_init in super class
            # then defreeze it first and call it
            # this behavior is differet to `dataclasses` with `frozen=True`
            # but similar if `frozen=False`
            # vars(self)[_FROZEN] = False
            # the following code will raise FrozenInstanceError in `dataclasses`
            # but it will work in `treeclass`,
            # i.e. `treeclass` defreezes the tree after `__post_init__`
            # >>> @dc.dataclass(frozen=True)
            # ... class Test:
            # ...    a:int = 1
            # ...    def __post_init__(self):
            # ...        self.b = 1
            output = getattr(self, _POST_INIT)()
            # call validation on fields that are not initialized
            vars(self)[_FROZEN] = False
            _apply_callbacks(self, init=False)

        # handle uninitialized fields
        for field in getattr(self, _FIELD_MAP).values():
            if field.name not in vars(self):
                # at this point, all fields should be initialized
                # in principle, this error will be caught when invoking `repr`/`str`
                # like in `dataclasses` but we raise it here for better error message.
                raise AttributeError(f"field=`{field.name}` is not initialized.")

        # delete the shadowing `__dict__` attribute to
        # restore the frozen behavior
        del vars(self)[_FROZEN]
        return output

    return init_method


def _validate_class(klass):
    if not isinstance(klass, type):
        raise TypeError(f"Expected `class` but got `{type(klass)}`.")

    for key, method in zip(("__delattr__", "__setattr__"), (_delattr, _setattr)):
        if key in vars(klass) and vars(klass)[key] is not method:
            # raise error if the current setattr/delattr is not immutable
            raise AttributeError(f"Cannot define `{key}` in {klass.__name__}.")

    return klass


def _treeclass_transform(klass):

    # add custom `dataclass` field_map and frozen attributes to the class
    setattr(klass, _FIELD_MAP, _generate_field_map(klass))
    setattr(klass, _FROZEN, True)
    setattr(klass, "__match_args__", tuple(getattr(klass, _FIELD_MAP).keys()))

    if "__init__" not in vars(klass):
        # generate the init method in case it is not defined by the user
        setattr(klass, "__init__", _generate_init(klass))

    # class initialization wrapper
    if "__getattr__" in vars(klass):
        setattr(klass, "__getattr__", _getattr_wrapper(klass.__getattr__))

    setattr(klass, "__new__", _new_wrapper(klass.__new__))
    setattr(klass, "__init__", _init_wrapper(klass.__init__))
    setattr(klass, "__init_subclass__", _init_sub_wrapper(klass.__init_subclass__))
    setattr(klass, "__getattribute__", _getattr_wrapper(klass.__getattribute__))

    # immutable attributes similar to `dataclasses`
    setattr(klass, "__setattr__", _setattr)
    setattr(klass, "__delattr__", _delattr)

    return klass


def _process_optional_methods(klass):
    # optional attributes
    attrs = dict()

    # pretty printing
    attrs["__repr__"] = tree_repr
    attrs["__str__"] = tree_str

    # indexing and masking
    attrs["at"] = property(_tree_indexer)

    # hashing and copying
    attrs["__copy__"] = _tree_copy
    attrs["__hash__"] = _tree_hash

    # math operations
    attrs["__abs__"] = bcmap(op.abs)
    attrs["__add__"] = bcmap(op.add)
    attrs["__and__"] = bcmap(op.and_)
    attrs["__ceil__"] = bcmap(math.ceil)
    attrs["__divmod__"] = bcmap(divmod)
    attrs["__eq__"] = bcmap(op.eq)
    attrs["__floor__"] = bcmap(math.floor)
    attrs["__floordiv__"] = bcmap(op.floordiv)
    attrs["__ge__"] = bcmap(op.ge)
    attrs["__gt__"] = bcmap(op.gt)
    attrs["__int__"] = bcmap(int)
    attrs["__invert__"] = bcmap(op.invert)
    attrs["__le__"] = bcmap(op.le)
    attrs["__lshift__"] = bcmap(op.lshift)
    attrs["__lt__"] = bcmap(op.lt)
    attrs["__matmul__"] = bcmap(op.matmul)
    attrs["__mod__"] = bcmap(op.mod)
    attrs["__mul__"] = bcmap(op.mul)
    attrs["__ne__"] = bcmap(op.ne)
    attrs["__neg__"] = bcmap(op.neg)
    attrs["__or__"] = bcmap(op.or_)
    attrs["__pos__"] = bcmap(op.pos)
    attrs["__pow__"] = bcmap(op.pow)
    attrs["__radd__"] = bcmap(op.add)
    attrs["__rand__"] = bcmap(op.and_)
    attrs["__rdivmod__"] = bcmap(ft.wraps(divmod)(lambda x, y: divmod(y, x)))
    attrs["__rfloordiv__"] = bcmap(ft.wraps(op.floordiv)(lambda x, y: y // x))
    attrs["__rlshift__"] = bcmap(ft.wraps(op.lshift)(lambda x, y: (y << x)))
    attrs["__rmatmul__"] = bcmap(ft.wraps(op.matmul)(lambda x, y: y @ x))
    attrs["__rmod__"] = bcmap(ft.wraps(op.mod)(lambda x, y: y % x))
    attrs["__rmul__"] = bcmap(op.mul)
    attrs["__ror__"] = bcmap(op.or_)
    attrs["__round__"] = bcmap(round)
    attrs["__rpow__"] = bcmap(ft.wraps(op.pow)(lambda x, y: op.pow(y, x)))
    attrs["__rrshift__"] = bcmap(ft.wraps(op.rshift)(lambda x, y: (y >> x)))
    attrs["__rshift__"] = bcmap(op.rshift)
    attrs["__rsub__"] = bcmap(ft.wraps(op.sub)(lambda x, y: (y - x)))
    attrs["__rtruediv__"] = bcmap(ft.wraps(op.truediv)(lambda x, y: (y / x)))
    attrs["__rxor__"] = bcmap(op.xor)
    attrs["__sub__"] = bcmap(op.sub)
    attrs["__truediv__"] = bcmap(op.truediv)
    attrs["__trunc__"] = bcmap(math.trunc)
    attrs["__xor__"] = bcmap(op.xor)

    for key in attrs:
        if key not in vars(klass):
            # do not override any user defined methods
            # this behavior similar is to `dataclasses.dataclass`
            setattr(klass, key, attrs[key])
    return klass


def treeclass(klass):
    """Decorator to convert a class to a `treeclass`

    Example:
        >>> import jax
        >>> import jax.numpy as jnp
        >>> import pytreeclass as pytc

        >>> @pytc.treeclass
        ... class Linear :
        ...     weight : jnp.ndarray
        ...     bias   : jnp.ndarray

        >>> def __init__(self,key,in_dim,out_dim):
        ...    self.weight = jax.random.normal(key,shape=(in_dim, out_dim)) * jnp.sqrt(2/in_dim)
        ...    self.bias = jnp.ones((1,out_dim))

        >>> def __call__(self,x):
        ...    return x @ self.weight + self.bias

    Raises:
        TypeError: if the input class is not a `class`
    """
    # check if the input is a valid class
    # in essence, it should be a type with immutable setters and deleters
    klass = _validate_class(klass)

    # add the immutable setters and deleters
    # and generate the `__init__` method if not present
    # generate fields from type annotations
    klass = _treeclass_transform(copy.deepcopy(klass))

    # add the optional methods to the class
    # optional methods are math operations, indexing and masking,
    # hashing and copying, and pretty printing
    klass = _process_optional_methods(klass)

    # add the class to the `JAX` registry if not registered
    return _register_treeclass(klass)


def is_tree_equal(lhs: Any, rhs: Any) -> bool:
    """Assert if two pytrees are equal"""
    lhs_leaves, lhs_treedef = jtu.tree_flatten(lhs)
    rhs_leaves, rhs_treedef = jtu.tree_flatten(rhs)

    if not (lhs_treedef == rhs_treedef):
        return False

    for (lhs, rhs) in zip(lhs_leaves, rhs_leaves):
        if isinstance(lhs, (jnp.ndarray, np.ndarray)):
            if isinstance(rhs, (jnp.ndarray, np.ndarray)):
                if not np.array_equal(lhs, rhs):
                    return False
            else:
                return False
        else:
            return lhs == rhs
    return True
