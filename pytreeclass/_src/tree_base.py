from __future__ import annotations

import functools as ft
from typing import Any, Callable, TypeVar

import jax.tree_util as jtu
from typing_extensions import dataclass_transform

from pytreeclass._src.tree_decorator import (
    _delattr,
    _field_registry,
    _generate_field_map,
    _generate_init,
    _init_wrapper,
    _setattr,
    field,
    fields,
    ovars,
    register_pytree_field_map,
)
from pytreeclass._src.tree_freeze import ImmutableWrapper, tree_hash
from pytreeclass._src.tree_indexer import (
    _leafwise_transform,
    is_tree_equal,
    tree_copy,
    tree_indexer,
)
from pytreeclass._src.tree_pprint import tree_repr, tree_str
from pytreeclass._src.tree_trace import register_pytree_node_trace

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
    ovars(tree).update(treedef[1])
    ovars(tree).update(zip(treedef[0], leaves))
    return tree


def _tree_flatten(tree: PyTree):
    """Flatten rule for `treeclass` to use with `jax.tree_flatten`."""
    static, dynamic = dict(ovars(tree)), dict()
    for key in _field_registry[type(tree)]:
        dynamic[key] = static.pop(key)
    return list(dynamic.values()), (tuple(dynamic.keys()), static)


def _tree_trace(tree: PyTree) -> list[tuple[Any, Any, Any, Any]]:
    """Trace flatten rule to be used with the `tree_trace` module."""
    leaves, (keys, _) = _tree_flatten(tree)
    names = (f"{key}" for key in keys)
    types = map(type, leaves)
    indices = range(len(leaves))
    metadatas = (dict(repr=F.repr, id=id(getattr(tree, F.name))) for F in fields(tree))  # type: ignore
    return [*zip(names, types, indices, metadatas)]


def _register_treeclass(klass: type[T]) -> type[T]:
    if klass not in _field_registry:
        # there are two cases where a class is registered more than once:
        # first, when a class is decorated with `treeclass` more than once (e.g. `treeclass(treeclass(Class))`)
        # second when a class is decorated with `treeclass` and has a parent class that is decorated with `treeclass`
        # in that case `__init_subclass__` registers the class before the decorator registers it.
        # this can be also be done using metaclass that registers the class on initialization
        # but we are trying to stay away from deep magic.
        # register the trace flatten rule
        register_pytree_node_trace(klass, _tree_trace)
        # register the generated field map
        register_pytree_field_map(klass, _generate_field_map(klass))
        # register the flatten/unflatten rules with jax
        jtu.register_pytree_node(klass, _tree_flatten, ft.partial(_tree_unflatten, klass))  # type: ignore
    return klass


def _getattribute_wrapper(getattribute_method):
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
    def tree_unwrap(value: Any) -> Any:
        # enables the transparent wrapper behavior iniside `treeclass` wrapped classes
        def is_leaf(x: Any) -> bool:
            return isinstance(x, ImmutableWrapper) or type(x) in _field_registry

        def unwrap(value: Any) -> Any:
            return value.unwrap() if isinstance(value, ImmutableWrapper) else value

        return jtu.tree_map(unwrap, value, is_leaf=is_leaf)

    @ft.wraps(getattribute_method)
    def wrapper(self, key: str) -> Any:
        value = getattribute_method(self, key)
        return tree_unwrap(value) if key in ovars(self) else value

    return wrapper


def _init_subclass_wrapper(init_subclass_method: Callable) -> Callable:
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
    @classmethod  # type: ignore
    @ft.wraps(init_subclass_method)
    def wrapper(klass: type, *a, **k) -> None:
        init_subclass_method(*a, **k)
        _register_treeclass(klass)

    return wrapper


def _treeclass_transform(klass: type[T]) -> type[T]:
    # the method is called after registering the class with `_register_treeclass`
    # cached to prevent wrapping the same class multiple times

    for key, method in (("__setattr__", _setattr), ("__delattr__", _delattr)):
        # basic required methods
        if key in vars(klass):
            if vars(klass)[key] is method:
                return klass  # already transformed
            # the user defined a method that conflicts with the required method
            msg = f"Unable to transform the class `{klass.__name__}` with {key} method defined."
            raise TypeError(msg)
        setattr(klass, key, method)

    if "__init__" not in vars(klass):
        # generate the init method in case it is not defined by the user
        setattr(klass, "__init__", _generate_init(klass))

    for key, wrapper in (
        ("__init__", _init_wrapper),
        ("__init_subclass__", _init_subclass_wrapper),
        ("__getattribute__", _getattribute_wrapper),
    ):
        # wrappers to enable the field initialization,
        # callback functionality and transparent wrapper behavior
        setattr(klass, key, wrapper(getattr(klass, key)))

    # basic optional methods
    for key, method in (
        ("__repr__", tree_repr),
        ("__str__", tree_str),
        ("__copy__", tree_copy),
        ("__hash__", tree_hash),
        ("__eq__", is_tree_equal),
        ("at", property(tree_indexer)),
    ):
        if key not in vars(klass):
            # keep the original method if it is defined by the user
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
    klass = _register_treeclass(klass)
    # add math operations methods if leafwise
    # do not override any user defined methods
    klass = _leafwise_transform(klass) if leafwise else klass
    # add `repr`,'str', 'at', 'copy', 'hash', 'copy'
    # add the immutable setters and deleters
    # generate the `__init__` method if not present using type hints.
    klass = _treeclass_transform(klass)

    return klass
