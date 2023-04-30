from __future__ import annotations

import abc
import functools as ft
import sys
from collections.abc import MutableMapping, MutableSequence
from types import FunctionType, MappingProxyType
from typing import Any, Callable, Hashable, NamedTuple, Sequence, TypeVar

import jax
import jax.tree_util as jtu
from typing_extensions import dataclass_transform

from pytreeclass._src.tree_freeze import tree_hash
from pytreeclass._src.tree_indexer import (
    AtIndexer,
    _leafwise_transform,
    _mutable_context,
    _mutable_instance_registry,
    is_tree_equal,
    tree_copy,
)
from pytreeclass._src.tree_pprint import tree_repr, tree_str
from pytreeclass._src.tree_trace import NamedSequenceKey


class NOT_SET:
    __repr__ = lambda _: "?"


T = TypeVar("T", bound=Hashable)
PyTree = Any

_NOT_SET = NOT_SET()
_MUTABLE_TYPES = (MutableSequence, MutableMapping, set)


"""Define a class that convert a class to a JAX compatible tree structure"""


class Field(NamedTuple):
    name: str | None = None
    type: type | None = None
    default: Any = _NOT_SET
    factory: Any = None
    init: bool = True
    repr: bool = True
    kw_only: bool = False
    pos_only: bool = False
    metadata: MappingProxyType[str, Any] | None = None
    callbacks: Sequence[Any] = ()
    alias: str | None = None

    def __eq__(self, other: Any) -> bool:
        return hash(self) == hash(other)

    def __hash__(self) -> int:
        return tree_hash(
            self.name,
            self.default,
            self.factory,
            self.init,
            self.kw_only,
            self.pos_only,
            self.alias,
        )


def field(
    *,
    default: Any = _NOT_SET,
    factory: Callable | None = None,
    init: bool = True,
    repr: bool = True,
    kw_only: bool = False,
    pos_only: bool = False,
    metadata: dict[str, Any] | None = None,  # type: ignore
    callbacks: Sequence[Any] = (),
    alias: str | None = None,
) -> Field:
    """
    Args:
        default: The default value of the field. Mutually exclusive with `factory`.
        factory: A 0-argument function called to initialize field value. Mutually exclusive with `default`.
        init: Whether the field is included in the object's __init__ function.
        repr: Whether the field is included in the object's __repr__ function.
        kw_only: Whether the field is keyword-only. Mutually exclusive with `pos_only`.
        pos_only: Whether the field is positional-only. Mutually exclusive with `kw_only`.
        metadata: A mapping of user-defined data for the field.
        callbacks: A sequence of functions to called on `setattr` during initialization to modify the field value.
        alias: An a alias for the field name in the constructor.

    Example:
        >>> import pytreeclass as pytc
        >>> def instance_cb_factory(klass):
        ...    def wrapper(x):
        ...        assert isinstance(x, klass)
        ...        return x
        ...    return wrapper


        >>> def positive_check_callback(x):
        ...    assert x > 0
        ...    return x

        >>> class Employee(pytc.TreeClass):
        ...    # assert employee `name` is str
        ...    name: str = pytc.field(callbacks=[instance_cb_factory(str)])
        ...    # use callback compostion to assert employee `age` is int and positive
        ...    age: int = pytc.field(callbacks=[instance_cb_factory(int), positive_check_callback])
        ...    # use `id` in the constructor for `_id` attribute
        ...    # this is useful for private attributes that are not supposed to be accessed directly
        ...    # and hide it from the repr
        ...    _id: int = pytc.field(alias="id", repr=False)

        >>> tree = Employee(name="Asem", age=10, id=1)
        >>> print(tree)  # _id is not shown
        Employee(name=Asem, age=10)
        >>> assert tree._id == 1  # this is the private attribute
    """
    if not isinstance(alias, (str, type(None))):
        msg = "`alias` must be a string or None, "
        msg += f"got type=`{type(alias).__name__}` instead."
        raise TypeError(msg)

    if default is not _NOT_SET and factory is not None:
        msg = "`default` and `factory` are mutually exclusive arguments."
        msg += f"got {default=} and {factory=}"
        raise ValueError(msg)

    if kw_only is True and pos_only is True:
        msg = "`kw_only` and `pos_only` are mutually exclusive arguments."
        msg += f"got {kw_only=} and {pos_only=}"
        raise ValueError(msg)

    if isinstance(metadata, dict):
        metadata = MappingProxyType(metadata)  # type: ignore
    elif metadata is not None:
        raise TypeError("`metadata` must be a Mapping or None")

    if not isinstance(callbacks, Sequence):
        msg = f"`callbacks` must be a Sequence of functions, got {type(callbacks)}"
        raise TypeError(msg)

    for index, callback in enumerate(callbacks):
        if not isinstance(callback, Callable):  # type: ignore
            msg = "`callbacks` must be a Sequence of zero argument functions, "
            msg += f"got `{type(callbacks).__name__}` at index={index}"
            raise TypeError(msg)

    return Field(
        name=None,
        type=None,
        default=default,
        factory=factory,
        init=init,
        repr=repr,
        kw_only=kw_only,
        pos_only=pos_only,
        metadata=metadata,  # type: ignore
        callbacks=callbacks,
        alias=alias,
    )


@ft.lru_cache
def build_field_map(klass: type) -> MappingProxyType[str, Field]:
    field_map = dict()  # type: dict[str, Field]

    if klass is object:
        return MappingProxyType(field_map)

    for base in reversed(klass.__mro__[1:]):
        field_map.update(build_field_map(base))

    # TODO: use inspect to get annotations, once we are on minimum python version >3.9
    if "__annotations__" not in vars(klass):
        return MappingProxyType(field_map)

    for name in (annotation_map := vars(klass)["__annotations__"]):
        value = vars(klass).get(name, _NOT_SET)
        type = annotation_map[name]

        if name == "self":
            # while `dataclasses` allows `self` as a field name, its confusing
            # and not recommended. so raise an error
            raise ValueError("Field name cannot be `self`.")

        if isinstance(value, Field):
            # case: `x: Any = field(default=1)`
            if isinstance(value.default, _MUTABLE_TYPES):
                # example case: `x: Any = field(default=[1, 2, 3])`
                msg = f"Mutable default value of field `{name}` is not allowed, use "
                msg += f"`factory=lambda: {value.default}` instead."
                raise TypeError(msg)

            field_map[name] = value._replace(name=name, type=type)

        elif isinstance(value, _MUTABLE_TYPES):
            # https://github.com/ericvsmith/dataclasses/issues/3
            # example case: `x: Any = [1, 2, 3]`
            msg = f"Mutable value= {(value)} is not allowed"
            msg += f" for field `{name}` in class `{klass.__name__}`.\n"
            msg += f" use `field(... ,factory=lambda:{value})` instead"
            raise TypeError(msg)

        elif value is _NOT_SET:
            # case: `x: Any`
            field_map[name] = Field(name=name, type=type)

        else:
            # case: `x: int = 1`
            field_map[name] = Field(name=name, type=type, default=value)

    return MappingProxyType(field_map)


def fields(tree: Any) -> Sequence[Field]:
    """Returns a tuple of `Field` objects for the given instance or class."""
    klass = tree if isinstance(tree, type) else type(tree)
    return tuple(build_field_map(klass).values())


@ft.lru_cache
def _generate_init_code(fields: Sequence[Field]) -> str:
    head = body = ""

    for field in fields:
        name = field.name  # name in body
        alias = field.alias or name  # name in constructor

        if field.kw_only and "*" not in head and field.init:
            head += "*, "

        if field.default is not _NOT_SET:
            vref = f"field_map['{name}'].default"
            head += f"{alias}={vref}, " if field.init else ""
            body += f"\t\tself.{name}=" + (f"{alias}\n " if field.init else f"{vref}\n")
        elif field.factory is not None:
            vref = f"field_map['{name}'].factory()"
            head += f"{alias}={vref}, " if field.init else ""
            body += f"\t\tself.{name}=" + (f"{alias}\n" if field.init else f"{vref}\n")
        else:
            head += f"{alias}, " if field.init else ""
            body += f"\t\tself.{name}={alias}\n " if field.init else ""

        if field.pos_only and field.init:
            head = head.replace("/,", "") + "/, "

    body += "\t\tpass"  # add pass to avoid syntax error if all fieds are ignored
    body = "\tdef __init__(self, " + head[:-2] + "):\n" + body
    body = f"def closure(field_map):\n{body}\n\treturn __init__"
    return body.expandtabs(4)


def _generate_init_method(klass: type) -> FunctionType:
    field_map = build_field_map(klass)
    init_code = _generate_init_code(tuple(field_map.values()))
    exec(init_code, vars(sys.modules[klass.__module__]), local_namespace := dict())  # type: ignore
    method = local_namespace["closure"](field_map)
    method.__qualname__ = f"{klass.__qualname__}.__init__"
    return method


def tree_setattr(tree: PyTree, key: str, value: Any) -> None:
    if id(tree) not in _mutable_instance_registry:
        msg = f"Cannot set attribute `{key}` = {value!r} on immutable instance of "
        msg += f"`{type(tree).__name__}`.\nUse `.at[`{key}`].set({value!r})` instead."
        raise AttributeError(msg)

    if key in (field_map := build_field_map(type(tree))):
        for callback in field_map[key].callbacks:
            try:
                # callback is a function that takes the value of the field
                # and returns a modified value
                value = callback(value)
            except Exception as e:
                msg = f"Error for field=`{key}`:\n{e}"
                raise type(e)(msg)

    vars(tree)[key] = value  # type: ignore


def tree_delattr(tree, key: str) -> None:
    # delete the attribute under `_mutable_context` context
    # otherwise raise an error
    if id(tree) not in _mutable_instance_registry:
        msg = f"Cannot delete attribute `{key}` on immutable instance of "
        msg += f"`{type(tree).__name__}`.\n"
        raise AttributeError(msg)

    del vars(tree)[key]


def _register_treeclass(klass: type[T]) -> type[T]:
    # handle all registration logic for `treeclass`

    def tree_unflatten(keys: tuple[str, ...], leaves: tuple[Any, ...]) -> T:
        # unflatten rule for `treeclass` to use with `jax.tree_unflatten`
        tree = getattr(object, "__new__")(klass)
        vars(tree).update(zip(keys, leaves))
        return tree

    def tree_flatten(tree: T) -> tuple[tuple[Any, ...], tuple[str, ...]]:
        # flatten rule for `treeclass` to use with `jax.tree_flatten`
        dynamic = vars(tree)
        return tuple(dynamic.values()), tuple(dynamic.keys())

    def tree_flatten_with_keys(tree: T):
        # flatten rule for `treeclass` to use with `jax.tree_util.tree_flatten_with_path`
        dynamic = dict(vars(tree))
        for idx, key in enumerate(vars(tree)):
            entry = NamedSequenceKey(idx, key)
            dynamic[key] = (entry, dynamic[key])
        return tuple(dynamic.values()), tuple(dynamic.keys())

    jtu.register_pytree_with_keys(
        nodetype=klass,
        flatten_func=tree_flatten,
        flatten_with_keys=tree_flatten_with_keys,
        unflatten_func=tree_unflatten,
    )
    return klass


class TreeClassMeta(abc.ABCMeta):
    def __call__(klass: type[T], *a, **k) -> T:
        self = getattr(klass, "__new__")(klass, *a, **k)

        with _mutable_context(self):
            getattr(klass, "__init__")(self, *a, **k)

            if post_init_func := getattr(klass, "__post_init__", None):
                # to simplify the logic, we call the post init method
                # even if the init method is not code-generated.
                post_init_func(self)

        # handle non-initialized fields
        if len(keys := set(build_field_map(klass)) - set(vars(self))) > 0:
            msg = f"Uninitialized fields: ({', '.join(keys)}) in the "
            msg += f"instance of `{type(self).__name__}`"
            raise AttributeError(msg)
        return self


@dataclass_transform(field_specifiers=(field, Field), frozen_default=True)
class TreeClass(metaclass=TreeClassMeta):
    """Convert a class to a JAX compatible tree structure.

    Example:
        >>> import jax
        >>> import pytreeclass as pytc

        >>> # Tree leaves are instance attributes
        >>> class Tree(pytc.TreeClass):
        ...     a:int = 1
        ...     b:float = 2.0
        >>> tree = Tree()
        >>> jax.tree_util.tree_leaves(tree)
        [1, 2.0]

        >>> # Leaf-wise math operations are supported by setting `leafwise=True`
        >>> class Tree(pytc.TreeClass, leafwise=True):
        ...     a:int = 1
        ...     b:float = 2.0
        >>> tree = Tree()
        >>> tree + 1
        Tree(a=2, b=3.0)

        >>> # Advanced indexing is supported using `at` property
        >>> class Tree(pytc.TreeClass):
        ...     a:int = 1
        ...     b:float = 2.0
        >>> tree = Tree()
        >>> tree.at["a"].get()
        Tree(a=1, b=None)
        >>> tree.at[0].get()
        Tree(a=1, b=None)

    Note:
        ``leafwise=True`` adds the following methods to the class

        ==================      ============
        Method                  Operator
        ==================      ============
        ``__add__``              ``+``
        ``__and__``              ``&``
        ``__ceil__``             ``math.ceil``
        ``__divmod__``           ``divmod``
        ``__eq__``               ``==``
        ``__floor__``            ``math.floor``
        ``__floordiv__``         ``//``
        ``__ge__``               ``>=``
        ``__gt__``               ``>``
        ``__invert__``           ``~``
        ``__le__``               ``<=``
        ``__lshift__``           ``<<``
        ``__lt__``               ``<``
        ``__matmul__``           ``@``
        ``__mod__``              ``%``
        ``__mul__``              ``*``
        ``__ne__``               ``!=``
        ``__neg__``              ``-``
        ``__or__``               ``|``
        ``__pos__``              ``+``
        ``__pow__``              ``**``
        ``__round__``            ``round``
        ``__sub__``              ``-``
        ``__truediv__``          ``/``
        ``__trunc__``            ``math.trunc``
        ``__xor__``              ``^``
        ==================      ============

    """

    def __init_subclass__(klass: type[T], *a, leafwise: bool = False, **k) -> None:
        super().__init_subclass__(*a, **k)

        if "__setattr__" in vars(klass) or "__delattr__" in vars(klass):
            # the user defined a method that conflicts with the reserved method
            msg = f"Unable to transform the class `{klass.__name__}` "
            msg += "with resereved methods: `__setattr__` or `__delattr__` defined."
            raise TypeError(msg)

        if "__init__" not in vars(klass):
            # generate the init method if not defined similar to dataclass
            setattr(klass, "__init__", _generate_init_method(klass))

        klass = _register_treeclass(klass)
        klass = _leafwise_transform(klass) if leafwise else klass

    @property
    def at(self) -> AtIndexer:
        return AtIndexer(self, where=())

    def __repr__(self) -> str:
        return tree_repr(self)

    def __str__(self) -> str:
        return tree_str(self)

    def __copy__(self) -> T:
        return tree_copy(self)

    def __hash__(self) -> int:
        return tree_hash(self)

    def __eq__(self, other: Any) -> bool | jax.Array:
        return is_tree_equal(self, other)

    def __setattr__(self, key: str, value: Any) -> None:
        return tree_setattr(self, key, value)

    def __delattr__(self, key: str) -> None:
        return tree_delattr(self, key)
