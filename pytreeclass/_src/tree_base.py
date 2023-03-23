from __future__ import annotations

import functools as ft
import operator as op
from math import ceil, floor, trunc
from typing import Any, Callable, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from typing_extensions import dataclass_transform

from pytreeclass._src.tree_decorator import (
    _FROZEN,
    _VARS,
    _delattr,
    _field_registry,
    _generate_field_map,
    _generate_init,
    _init_wrapper,
    _setattr,
    field,
)
from pytreeclass._src.tree_freeze import _tree_hash, _tree_unwrap
from pytreeclass._src.tree_indexer import tree_indexer
from pytreeclass._src.tree_pprint import tree_repr, tree_str
from pytreeclass._src.tree_trace import _trace_registry, _TraceRegistryEntry

PyTree = Any
T = TypeVar("T")


def _tree_unflatten(klass: type, treedef: Any, leaves: list[Any]):
    """Unflatten rule for `treeclass` to use with `jax.tree_unflatten`."""
    tree = getattr(object, "__new__")(klass)
    # update through vars, to avoid calling the `setattr` method
    # that will check for callbacks.
    # calling `setattr` will trigger any defined callbacks by the user
    # on each unflattening which is not efficient.
    # however it might be useful to constantly check if the updated value is
    # satisfying the constraints defined by the user in the callbacks.
    getattr(tree, _VARS).update(treedef[1])
    getattr(tree, _VARS).update(zip(treedef[0], leaves))
    return tree


def _tree_flatten(
    tree: PyTree,
) -> tuple[list[Any], tuple[tuple[str], dict[str, Any]]]:
    """Flatten rule for `treeclass` to use with `jax.tree_flatten`."""
    static, dynamic = dict(getattr(tree, _VARS)), dict()
    for key in _field_registry[type(tree)]:
        dynamic[key] = static.pop(key)
    return list(dynamic.values()), (tuple(dynamic.keys()), static)


def _tree_trace(
    tree: PyTree,
) -> list[tuple[Any, Any, tuple[int, int], Any]]:
    """Trace flatten rule to be used with the `tree_trace` module."""
    leaves, (keys, _) = _tree_flatten(tree)
    names = (f"{key}" for key in keys)
    types = map(type, leaves)
    indices = range(len(leaves))
    fields = (_field_registry[type(tree)][key] for key in keys)
    metadatas = (dict(repr=F.repr, id=id(getattr(tree, F.name))) for F in fields)
    return [*zip(names, types, indices, metadatas)]


@ft.lru_cache(maxsize=1)
def _register_treeclass(klass: type[T]) -> type[T]:
    # register a treeclass only once by using `lru_cache`
    # there are two cases where a class is registered more than once:
    # first, when a class is decorated with `treeclass` more than once (e.g. `treeclass(treeclass(Class))`)
    # second when a class is decorated with `treeclass` and has a parent class that is also decorated with `treeclass`
    # in that case `__init_subclass__` registers the class before the decorator registers it.
    # this can be also be done using metaclass that registers the class on initialization
    # but we are trying to stay away from deep magic.
    jtu.register_pytree_node(klass, _tree_flatten, ft.partial(_tree_unflatten, klass))  # type: ignore
    # register the trace flatten rule without the validation to avoid
    # the unnecessary overhead of the first call validation.
    _trace_registry[klass] = _TraceRegistryEntry(_tree_trace)
    # generate field map for the class and register it in a weakref registry
    _field_registry[klass] = _generate_field_map(klass)
    return klass


def _getattr_wrapper(getattr_func):
    @ft.wraps(getattr_func)
    def wrapper(self, key: str) -> Any:
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
        value = getattr_func(self, key)
        # unwrap non-`treeclass` wrapped instance variables
        # so the getattr will always return unwrapped values.
        # this renders the wrapped instance variables transparent to the user
        return _tree_unwrap(value) if key in getattr_func(self, _VARS) else value

    return wrapper


def _init_sub_wrapper(init_subclass_func: Callable) -> Callable:
    @classmethod  # type: ignore
    @ft.wraps(init_subclass_func)
    def wrapper(klass: type, *a, **k) -> None:
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
        init_subclass_func(*a, **k)
        return _register_treeclass(klass)

    return wrapper


def _validate_class(klass: type) -> type:
    if not isinstance(klass, type):
        raise TypeError(f"Expected `class` but got `{type(klass)}`.")

    for key, method in zip(("__delattr__", "__setattr__"), (_delattr, _setattr)):
        if key in getattr(klass, _VARS) and getattr(klass, _VARS)[key] is not method:
            # raise error if the current setattr/delattr is not immutable
            raise AttributeError(f"Cannot define `{key}` in {klass.__name__}.")

    return klass


def is_tree_equal(*trees: Any) -> bool | jax.Array:
    """Return `True` or Array(True) if all pytrees are equal.

    Note:
        trees are compared using their leaves and treedefs.
        For `array` leaves `np.array_equal` is used, for other leaves
        method `__eq__` is used.
    """
    tree, *rest = trees
    leaves0, treedef0 = jtu.tree_flatten(tree)

    for tree in rest:
        leaves, treedef = jtu.tree_flatten(tree)
        if treedef != treedef0:
            # non matching treedefs
            return False

        for lhs, rhs in zip(leaves0, leaves):
            if hasattr(lhs, "shape") and hasattr(lhs, "dtype"):
                # lhs leaf is an array
                if hasattr(rhs, "shape") and hasattr(rhs, "dtype"):
                    # rhs leaf is an array
                    if jnp.array_equal(lhs, rhs) == False:
                        # lhs array leaf is not equal to rhs array leaf
                        return False
                else:
                    # lhs array leaf is an array but rhs
                    # leaf is not an array
                    return False
            else:
                if lhs != rhs:
                    # non-array lhs leaf is not equal to the
                    # non-array rhs leaf
                    return False
    return True


def _treeclass_transform(klass: type[T]) -> type[T]:
    # add custom `dataclass`-like fields map
    # flag for the immutable behavior used throughout the code
    setattr(klass, _FROZEN, True)

    if "__init__" not in getattr(klass, _VARS):
        # generate the init method in case it is not defined by the user
        setattr(klass, "__init__", _generate_init(klass))

    # wrappers to enable the field initialization,
    # callback functionality and transparent wrapper behavior
    for key, wrapper in (
        ("__init__", _init_wrapper),
        ("__init_subclass__", _init_sub_wrapper),
        ("__getattribute__", _getattr_wrapper),
    ):
        setattr(klass, key, wrapper(getattr(klass, key)))

    # basic required methods
    for key, method in (
        ("__setattr__", _setattr),
        ("__delattr__", _delattr),
        ("__match_args__", tuple(_field_registry[klass].keys())),
    ):
        setattr(klass, key, method)

    # basic optional methods
    for key, method in (
        ("__repr__", tree_repr),
        ("__str__", tree_str),
        ("__copy__", _tree_copy),
        ("__hash__", _tree_hash),
        ("__eq__", is_tree_equal),
        ("at", property(tree_indexer)),
    ):
        if key not in getattr(klass, _VARS):
            # keep the original method if it is defined by the user
            # this behavior similar is to `dataclasses.dataclass`
            setattr(klass, key, method)

    return klass


def _tree_copy(tree: PyTree) -> PyTree:
    """Return a copy of the tree."""
    return jtu.tree_unflatten(*jtu.tree_flatten(tree)[::-1])  # type: ignore


def _unary_op(func):
    def wrapper(self):
        return jtu.tree_map(func, self)

    return ft.wraps(func)(wrapper)


def _binary_op(func):
    def wrapper(lhs, rhs=None):
        if isinstance(rhs, type(lhs)):
            return jtu.tree_map(func, lhs, rhs)
        return jtu.tree_map(lambda x: func(x, rhs), lhs)

    return ft.wraps(func)(wrapper)


def _swop(func):
    # swaping the arguments of a two-arg function
    return ft.wraps(func)(lambda lhs, rhs: func(rhs, lhs))


def _leafwise_transform(klass: type[T]) -> type[T]:
    # add leafwise transform methods to the class
    # that enable the user to apply a function to
    # all the leaves of the tree
    for key, method in (
        ("__abs__", _unary_op(abs)),
        ("__add__", _binary_op(op.add)),
        ("__and__", _binary_op(op.and_)),
        ("__ceil__", _unary_op(ceil)),
        ("__divmod__", _binary_op(divmod)),
        ("__eq__", _binary_op(op.eq)),
        ("__floor__", _unary_op(floor)),
        ("__floordiv__", _binary_op(op.floordiv)),
        ("__ge__", _binary_op(op.ge)),
        ("__gt__", _binary_op(op.gt)),
        ("__invert__", _unary_op(op.invert)),
        ("__le__", _binary_op(op.le)),
        ("__lshift__", _binary_op(op.lshift)),
        ("__lt__", _binary_op(op.lt)),
        ("__matmul__", _binary_op(op.matmul)),
        ("__mod__", _binary_op(op.mod)),
        ("__mul__", _binary_op(op.mul)),
        ("__ne__", _binary_op(op.ne)),
        ("__neg__", _unary_op(op.neg)),
        ("__or__", _binary_op(op.or_)),
        ("__pos__", _unary_op(op.pos)),
        ("__pow__", _binary_op(op.pow)),
        ("__radd__", _binary_op(_swop(op.add))),
        ("__rand__", _binary_op(_swop(op.and_))),
        ("__rdivmod__", _binary_op(_swop(divmod))),
        ("__rfloordiv__", _binary_op(_swop(op.floordiv))),
        ("__rlshift__", _binary_op(_swop(op.lshift))),
        ("__rmatmul__", _binary_op(_swop(op.matmul))),
        ("__rmod__", _binary_op(_swop(op.mod))),
        ("__rmul__", _binary_op(_swop(op.mul))),
        ("__ror__", _binary_op(_swop(op.or_))),
        ("__round__", _binary_op(round)),
        ("__rpow__", _binary_op(_swop(op.pow))),
        ("__rrshift__", _binary_op(_swop(op.rshift))),
        ("__rshift__", _binary_op(op.rshift)),
        ("__rsub__", _binary_op(_swop(op.sub))),
        ("__rtruediv__", _binary_op(_swop(op.truediv))),
        ("__rxor__", _binary_op(_swop(op.xor))),
        ("__sub__", _binary_op(op.sub)),
        ("__truediv__", _binary_op(op.truediv)),
        ("__trunc__", _unary_op(trunc)),
        ("__xor__", _binary_op(op.xor)),
    ):
        if key not in getattr(klass, _VARS):
            # do not override any user defined methods
            # this behavior similar is to `dataclasses.dataclass`
            setattr(klass, key, method)
    return klass


@dataclass_transform(field_specifiers=(field,))
def treeclass(klass: type[T], *, leafwise: bool = False) -> type[T]:
    """Convert a class to a JAX compatible tree structure.

    Args:
        klass: class to be converted to a `treeclass`
        leafwise: Wether to generate leafwise math operations methods. Defaults to `False`.

    Example:
        >>> import functools as ft
        >>> import jax
        >>> import pytreeclass as pytc

        >>> # Tree leaves are defined by type hinted fields at the class level
        >>> @pytc.treeclass
        ... class Tree:
        ...     a:int = 1
        ...     b:float = 2.0
        >>> tree = Tree()
        >>> jax.tree_util.tree_leaves(tree)
        [1, 2.0]

        >>> # Leaf-wise math operations are supported by setting `leafwise=True`
        >>> @ft.partial(pytc.treeclass, leafwise=True)
        ... class Tree:
        ...     a:int = 1
        ...     b:float = 2.0
        >>> tree = Tree()
        >>> tree + 1
        Tree(a=2, b=3.0)

        >>> # Advanced indexing is supported using `at` property
        >>> @pytc.treeclass
        ... class Tree:
        ...     a:int = 1
        ...     b:float = 2.0
        >>> tree = Tree()
        >>> tree.at[0].get()
        Tree(a=1, b=None)
        >>> tree.at["a"].get()
        Tree(a=1, b=None)

    Note:
        Indexing is supported for {`list`, `tuple`, `dict`, `defaultdict`, `OrderedDict`, `namedtuple`}
        and `treeclass` wrapped classes.

        Extending indexing to other types is possible by registering the type with
        `pytreeclass.register_pytree_node_trace`

    Note:
        `leafwise`=True adds the following methods to the class:
        .. code-block:: python
            '__add__', '__and__', '__ceil__', '__divmod__', '__eq__', '__floor__', '__floordiv__',
            '__ge__', '__gt__', '__invert__', '__le__', '__lshift__', '__lt__',
            '__matmul__', '__mod__', '__mul__', '__ne__', '__neg__', '__or__', '__pos__',
            '__pow__', '__radd__', '__rand__', '__rdivmod__', '__reduce__', '__rfloordiv__',
            '__rlshift__', '__rmatmul__', '__rmod__', '__rmul__', '__ror__', '__round__', '__rpow__',
            '__rrshift__', '__rshift__', '__rsub__', '__rtruediv__', '__rxor__', '__sub__',
            '__truediv__', '__trunc__', '__xor__',

    Raises:
        TypeError: if the input is not a class.
    """
    # check if the input is a valid class
    # in essence, it should be a type with immutable setters and deleters
    klass = _validate_class(klass)

    # add the class to the `JAX`, `trace`, and `field_map` registries
    klass = _register_treeclass(klass)

    # add math operations methods if leafwise
    # do not override any user defined methods
    klass = _leafwise_transform(klass) if leafwise else klass

    # add `repr`,'str', 'at', 'copy', 'hash', 'copy'
    # add the immutable setters and deleters
    # generate the `__init__` method if not present using type hints.
    klass = _treeclass_transform(klass)

    return klass
