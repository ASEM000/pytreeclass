from __future__ import annotations

import dataclasses as dc
import functools as ft
import math
import operator as op
from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax._src.tree_util import _registry

from pytreeclass._src.tree_decorator import (
    _FIELD_MAP,
    _FROZEN,
    _NOT_SET,
    _POST_INIT,
    Field,
    _apply_callbacks,
    _process_class,
)
from pytreeclass._src.tree_freeze import _tree_hash, unfreeze
from pytreeclass._src.tree_indexer import _tree_copy, _tree_indexer, bcmap
from pytreeclass._src.tree_pprint import tree_repr, tree_str

PyTree = Any


def _register_treeclass(klass):
    if klass not in _registry:
        jtu.register_pytree_node(klass, _flatten, ft.partial(_unflatten, klass))
    return klass


def _setattr(tree: PyTree, key: str, value: Any) -> None:
    """Set the attribute of the tree if the tree is not frozen"""
    if getattr(tree, _FROZEN):
        msg = f"Cannot set {key}={value!r}. Use `.at['{key}'].set({value!r})` instead."
        raise dc.FrozenInstanceError(msg)

    tree.__dict__[key] = value

    if hasattr(value, _FIELD_MAP) and (key not in getattr(tree, _FIELD_MAP)):
        field = Field(name=key, type=type(value))  # type: ignore
        # register it to dataclass fields
        getattr(tree, _FIELD_MAP)[key] = field


def _delattr(tree, key: str) -> None:
    """Delete the attribute of the  if tree is not frozen"""
    if getattr(tree, _FROZEN):
        raise dc.FrozenInstanceError(f"Cannot delete {key}.")
    del tree.__dict__[key]


def _new_wrapper(new_func):
    @ft.wraps(new_func)
    def new_method(klass, *_, **__) -> PyTree:
        self = new_func(klass)
        for field in getattr(klass, _FIELD_MAP).values():
            if field.default is not _NOT_SET:
                self.__dict__[field.name] = field.default
            elif field.default_factory is not None:
                self.__dict__[field.name] = field.default_factory()
        return self

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
        self.__dict__[_FROZEN] = False
        output = init_func(self, *a, **k)

        # call callbacks on fields that are initialized
        self.__dict__[_FROZEN] = False
        _apply_callbacks(self, init=True)

        # in case __post_init__ is defined then call it
        # after the tree is initialized
        # here, we assume that __post_init__ is a method
        if hasattr(self, _POST_INIT):
            # in case we found post_init in super class
            # then defreeze it first and call it
            # this behavior is differet to `dataclasses` with `frozen=True`
            # but similar if `frozen=False`
            # self.__dict__[_FROZEN] = False
            output = getattr(self, _POST_INIT)()
            # call validation on fields that are not initialized
            self.__dict__[_FROZEN] = False
            _apply_callbacks(self, init=False)

        # handle freezing values and uninitialized fields
        for field in getattr(self, _FIELD_MAP).values():
            if field.name not in vars(self):
                # at this point, all fields should be initialized
                # in principle, this error will be caught when invoking `repr`/`str`
                # like in `dataclasses` but we raise it here for better error message.
                raise AttributeError(f"field=`{field.name}` is not initialized.")

        # delete the shadowing `__dict__` attribute to
        # restore the frozen behavior
        del self.__dict__[_FROZEN]
        return output

    return init_method


def _getattr(tree, key: str) -> Any:
    # avoid non-scalar error, raised by `jax` transformation
    # if a frozen value is returned.
    return unfreeze(object.__getattribute__(tree, key))


def _flatten(tree) -> tuple[Any, tuple[str, dict[str, Any]]]:
    """Flatten rule for `jax.tree_flatten`"""
    # in essence anything not declared in dataclass fields will be considered static
    static, dynamic = dict(vars(tree)), dict()
    for key in getattr(tree, _FIELD_MAP):
        dynamic[key] = static.pop(key)

    return dynamic.values(), (dynamic.keys(), static)


def _unflatten(klass, treedef, leaves):
    """Unflatten rule for `jax.tree_unflatten`"""
    tree = klass.__new__(klass)  # do not call klass constructor
    tree.__dict__.update(treedef[1])
    tree.__dict__.update(zip(treedef[0], leaves))
    return tree


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

    if not isinstance(klass, type):
        # class decorator must be applied to a class
        raise TypeError(f"Expected `class` but got `{type(klass)}`.")

    for method_name in (
        "__setattr__",
        "__delattr__",
        "__getattribute__",
        "__getattr__",
    ):
        if method_name in vars(klass):
            # explicitly ensure that the class does not define these methods
            # even though it is being overriden by the decorator
            raise AttributeError(f"Cannot define `{method_name}` in {klass.__name__}.")

    # add `_FIELD_MAP`, `_FROZEN` and generate `__init__`
    # method from fields if not defined.
    klass = _process_class(klass)

    # class initialization
    klass.__new__ = _new_wrapper(klass.__new__)
    klass.__init__ = _init_wrapper(klass.__init__)
    klass.__init_subclass__ = _init_sub_wrapper(klass.__init_subclass__)

    # immutable attributes
    klass.__getattribute__ = _getattr
    klass.__setattr__ = _setattr
    klass.__delattr__ = _delattr

    # pretty printing
    klass.__repr__ = tree_repr
    klass.__str__ = tree_str

    # indexing and masking
    klass.at = property(_tree_indexer)

    # hasing and copying
    klass.__copy__ = _tree_copy
    klass.__hash__ = _tree_hash

    # math operations
    klass.__abs__ = bcmap(op.abs)
    klass.__add__ = bcmap(op.add)
    klass.__and__ = bcmap(op.and_)
    klass.__ceil__ = bcmap(math.ceil)
    klass.__divmod__ = bcmap(divmod)
    klass.__eq__ = bcmap(op.eq)
    klass.__floor__ = bcmap(math.floor)
    klass.__floordiv__ = bcmap(op.floordiv)
    klass.__ge__ = bcmap(op.ge)
    klass.__gt__ = bcmap(op.gt)
    klass.__inv__ = bcmap(op.inv)
    klass.__invert__ = bcmap(op.invert)
    klass.__le__ = bcmap(op.le)
    klass.__lshift__ = bcmap(op.lshift)
    klass.__lt__ = bcmap(op.lt)
    klass.__matmul__ = bcmap(op.matmul)
    klass.__mod__ = bcmap(op.mod)
    klass.__mul__ = bcmap(op.mul)
    klass.__ne__ = bcmap(op.ne)
    klass.__neg__ = bcmap(op.neg)
    klass.__or__ = bcmap(op.or_)
    klass.__pos__ = bcmap(op.pos)
    klass.__pow__ = bcmap(op.pow)
    klass.__radd__ = bcmap(op.add)
    klass.__rand__ = bcmap(op.and_)
    klass.__rdivmod__ = bcmap(divmod)
    klass.__rfloordiv__ = bcmap(op.floordiv)
    klass.__rlshift__ = bcmap(op.lshift)
    klass.__rmatmul__ = bcmap(op.matmul)
    klass.__rmod__ = bcmap(op.mod)
    klass.__rmul__ = bcmap(op.mul)
    klass.__ror__ = bcmap(op.or_)
    klass.__round__ = bcmap(round)
    klass.__rpow__ = bcmap(op.pow)
    klass.__rrshift__ = bcmap(op.rshift)
    klass.__rshift__ = bcmap(op.rshift)
    klass.__rsub__ = bcmap(op.sub)
    klass.__rtruediv__ = bcmap(op.truediv)
    klass.__rxor__ = bcmap(op.xor)
    klass.__sub__ = bcmap(op.sub)
    klass.__truediv__ = bcmap(op.truediv)
    klass.__xor__ = bcmap(op.xor)

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
