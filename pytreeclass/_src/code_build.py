# Copyright 2023 pytreeclass authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Constructor code generation from type annotations."""

from __future__ import annotations

import functools as ft
import sys
from collections import defaultdict
from collections.abc import Callable, MutableMapping, MutableSequence, MutableSet
from types import MappingProxyType
from typing import Any, Literal, Sequence, TypeVar, get_args

import jax.tree_util as jtu
from typing_extensions import dataclass_transform

T = TypeVar("T")
PyTree = Any
EllipsisType = type(Ellipsis)
ArgKindType = Literal["POS_ONLY", "POS_OR_KW", "VAR_POS", "KW_ONLY", "VAR_KW"]
ArgKind = get_args(ArgKindType)


class Null:
    __slots__ = ()

    def __repr__(self) -> str:
        return "NULL"


NULL = Null()


def slots(klass) -> tuple[str, ...]:
    return getattr(klass, "__slots__", ())


class Field:
    """Field placeholder for `autoinit`."""

    __slots__ = [
        "name",
        "type",
        "default",
        "init",
        "repr",
        "kind",
        "metadata",
        "on_setattr",
        "on_getattr",
        "alias",
    ]

    def __init__(
        self,
        *,
        name: str | None = None,
        type: type | None = None,
        default: Any = NULL,
        init: bool = True,
        repr: bool = True,
        kind: ArgKind = "POS_OR_KW",
        metadata: dict[str, Any] | None = None,
        on_setattr: Sequence[Callable[[Any], Any]] = (),
        on_getattr: Sequence[Callable[[Any], Any]] = (),
        alias: str | None = None,
    ):
        """Initialize the field attributes.

        Args:
            name: The field name.
            type: The field type.
            default: The default value of the field.
            init: Whether the field is included in the object's ``__init__`` function.
            repr: Whether the field is included in the object's ``__repr__`` function.
            kind: Argument kind, one of:

                - ``POS_ONLY``: positional only argument (e.g. ``x`` in ``def f(x, /):``)
                - ``VAR_POS``: variable positional argument (e.g. ``*x`` in ``def f(*x):``)
                - ``POS_OR_KW``: positional or keyword argument (e.g. ``x`` in ``def f(x):``)
                - ``KW_ONLY``: keyword only argument (e.g. ``x`` in ``def f(*, x):``)
                - ``VAR_KW``: variable keyword argument (e.g. ``**x`` in ``def f(**x):``)

            metadata: A mapping of user-defined data for the field.
            on_setattr: A sequence of functions called on ``__setattr__``.
            on_getattr: A sequence of functions called on ``__getattr__``.
            alias: An a alias for the field name in the constructor. e.g ``name=x``,
                ``alias=y`` will allow ``obj = Class(y=1)`` to be equivalent to
                ``obj = Class(x=1)``.
        """
        self.name = name
        self.type = type
        self.default = default
        self.init = init
        self.repr = repr
        self.kind = kind
        self.metadata = metadata
        self.on_setattr = on_setattr
        self.on_getattr = on_getattr
        self.alias = alias

    def replace(self, **kwargs) -> Field:
        """Replace the field attributes."""
        return type(self)(**{k: kwargs.get(k, getattr(self, k)) for k in slots(Field)})

    def pipe(self, funcs: Sequence[Callable[[Any], Any]], value: Any):
        """Apply a sequence of functions on the field value."""
        for func in funcs:
            try:
                value = func(value)
            except Exception as e:
                cname = getattr(func, "__name__", func)
                raise type(e)(f"On applying {cname} for field=`{self.name}`:\n{e}")
        return value

    def __repr__(self) -> str:
        """Return the string representation of the field."""
        attrs = [f"{k}={getattr(self, k)!r}" for k in slots(Field)]
        return f"{type(self).__name__}({', '.join(attrs)})"

    def __set_name__(self, _, name: str) -> None:
        """Set the field name."""
        self.name = name

    def __get__(self: T, instance, _) -> T | Any:
        """Return the field value."""
        if instance is None:
            return self
        return self.pipe(self.on_getattr, vars(instance)[self.name])

    def __set__(self: T, instance, value) -> None:
        """Set the field value."""
        vars(instance)[self.name] = self.pipe(self.on_setattr, value)

    def __delete__(self: T, instance) -> None:
        """Delete the field value."""
        del vars(instance)[self.name]


jtu.register_pytree_node(
    nodetype=Field,
    flatten_func=lambda field: ((), {k: getattr(field, k) for k in slots(Field)}),
    unflatten_func=lambda data, _: Field(**data),
)


def field(
    *,
    default: Any = NULL,
    init: bool = True,
    repr: bool = True,
    kind: ArgKindType = "POS_OR_KW",
    metadata: dict[str, Any] | None = None,  # type: ignore
    on_setattr: Sequence[Any] = (),
    on_getattr: Sequence[Any] = (),
    alias: str | None = None,
) -> Field:
    """Field placeholder for type hinted attributes.

    Args:
        default: The default value of the field.
        init: Whether the field is included in the object's ``__init__`` function.
        repr: Whether the field is included in the object's ``__repr__`` function.
        kind: Argument kind, one of:

            - ``POS_ONLY``: positional only argument (e.g. ``x`` in ``def f(x, /):``)
            - ``VAR_POS``: variable positional argument (e.g. ``*x`` in ``def f(*x):``)
            - ``POS_OR_KW``: positional or keyword argument (e.g. ``x`` in ``def f(x):``)
            - ``KW_ONLY``: keyword only argument (e.g. ``x`` in ``def f(*, x):``)
            - ``VAR_KW``: variable keyword argument (e.g. ``**x`` in ``def f(**x):``)

        metadata: A mapping of user-defined data for the field.
        on_setattr: A sequence of functions to called on ``__setattr__``.
        on_getattr: A sequence of functions to called on ``__getattr__``.
        alias: An a alias for the field name in the constructor. e.g ``name=x``,
            ``alias=y`` will allow ``obj = Class(y=1)`` to be equivalent to
            ``obj = Class(x=1)``.

    Example:
        Type and range validation using :attr:`on_setattr`:

        >>> import pytreeclass as pytc
        >>> @pytc.autoinit
        ... class IsInstance(pytc.TreeClass):
        ...    klass: type
        ...    def __call__(self, x):
        ...        assert isinstance(x, self.klass)
        ...        return x
        <BLANKLINE>
        >>> @pytc.autoinit
        ... class Range(pytc.TreeClass):
        ...    start: int|float = float("-inf")
        ...    stop: int|float = float("inf")
        ...    def __call__(self, x):
        ...        assert self.start <= x <= self.stop
        ...        return x
        <BLANKLINE>
        >>> @pytc.autoinit
        ... class Employee(pytc.TreeClass):
        ...    # assert employee ``name`` is str
        ...    name: str = pytc.field(on_setattr=[IsInstance(str)])
        ...    # use callback compostion to assert employee ``age`` is int and positive
        ...    age: int = pytc.field(on_setattr=[IsInstance(int), Range(1)])
        >>> employee = Employee(name="Asem", age=10)
        >>> print(employee)
        Employee(name=Asem, age=10)

    Example:
        Private attribute using :attr:`alias`:

        >>> import pytreeclass as pytc
        >>> @pytc.autoinit
        ... class Employee(pytc.TreeClass):
        ...     # `alias` is the name used in the constructor
        ...    _name: str = pytc.field(alias="name")
        >>> employee = Employee(name="Asem")  # use `name` in the constructor
        >>> print(employee)  # `_name` is the private attribute name
        Employee(_name=Asem)

    Example:
        Buffer creation using :attr:`on_getattr`:

        >>> import pytreeclass as pytc
        >>> import jax.numpy as jnp
        >>> @pytc.autoinit
        ... class Tree(pytc.TreeClass):
        ...     buffer: jax.Array = pytc.field(on_getattr=[jax.lax.stop_gradient])
        >>> tree = Tree(buffer=jnp.array((1.0, 2.0)))
        >>> def sum_buffer(tree):
        ...     return tree.buffer.sum()
        >>> print(jax.grad(sum_buffer)(tree))  # no gradient on `buffer`
        Tree(buffer=[0. 0.])

    Example:
        Parameterization using :attr:`on_getattr`:

        >>> import pytreeclass as pytc
        >>> import jax.numpy as jnp
        >>> def symmetric(array: jax.Array) -> jax.Array:
        ...    triangle = jnp.triu(array)  # upper triangle
        ...    return triangle + triangle.transpose(-1, -2)
        >>> @pytc.autoinit
        ... class Tree(pytc.TreeClass):
        ...    symmetric_matrix: jax.Array = pytc.field(on_getattr=[symmetric])
        >>> tree = Tree(symmetric_matrix=jnp.arange(9).reshape(3, 3))
        >>> print(tree.symmetric_matrix)
        [[ 0  1  2]
         [ 1  8  5]
         [ 2  5 16]]

    Note:
        - :func:`field` is commonly used to annotate the class attributes to be
          used by the :func:`autoinit` decorator to generate the ``__init__``
          method similar to ``dataclasses.dataclass``.

        - :func:`field` can be used without the :func:`autoinit` as a descriptor
          to apply functions on the field values during initialization using
          the ``on_setattr`` / ``on_getattr`` argument.

            >>> import pytreeclass as pytc
            >>> def print_and_return(x):
            ...    print(f"Setting {x}")
            ...    return x
            >>> class Tree:
            ...    # `a` must be defined as a class attribute for the descriptor to work
            ...    a: int = pytc.field(on_setattr=[print_and_return])
            ...    def __init__(self, a):
            ...        self.a = a
            >>> tree = Tree(1)
            Setting 1
    """
    if not isinstance(alias, (str, type(None))):
        raise TypeError(f"Non-string {alias=} argument provided to `field`")
    if not isinstance(metadata, (dict, type(None))):
        raise TypeError(f"Non-dict {metadata=} argument provided to `field`")
    if kind not in ArgKind:
        raise ValueError(f"{kind=} not in {ArgKind}")
    if not isinstance(on_setattr, Sequence):
        raise TypeError(f"Non-sequence {on_setattr=} argument provided to `field`")
    if not isinstance(on_getattr, Sequence):
        raise TypeError(f"Non-sequence {on_getattr=} argument provided to `field`")
    for func in on_setattr:
        if not isinstance(func, Callable):  # type: ignore
            raise TypeError(f"Non-callable {func=} provided to `field`")
    for func in on_getattr:
        if not isinstance(func, Callable):
            raise TypeError(f"Non-callable {func=} provided to `field`")

    return Field(
        default=default,
        init=init,
        repr=repr,
        kind=kind,
        metadata=metadata,  # type: ignore
        on_setattr=on_setattr,
        on_getattr=on_getattr,
        alias=alias,
    )


@ft.lru_cache(maxsize=128)
def build_field_map(klass: type) -> MappingProxyType[str, Field]:
    field_map: dict[str, Field] = dict()

    if klass is object:
        return MappingProxyType(field_map)

    for base in reversed(klass.__mro__[1:]):
        field_map.update(build_field_map(base))

    if (hint_map := vars(klass).get("__annotations__", NULL)) is NULL:
        return MappingProxyType(field_map)

    if "self" in hint_map:
        raise ValueError("`Field` name cannot be `self`.")

    for key, hint in hint_map.items():
        # get the current base key
        value = vars(klass).get(key, NULL)

        if not isinstance(value, Field):
            # non-`Field` annotation is ignored
            # non-autoinit base class type hints are ignored
            continue

        if isinstance(value.default, (MutableSequence, MutableMapping, MutableSet)):
            # https://github.com/google/jax/issues/14295
            # example case: `x: Any = field(default=[1, 2, 3])`
            raise TypeError(f"Mutable {value.default=} is not allowed.")

        # case: `x: Any = field(default=1)`
        field_map[key] = value.replace(name=key, type=hint)

    return MappingProxyType(field_map)


def fields(x: Any) -> tuple[Field, ...]:
    """Returns a tuple of ``Field`` objects for the given instance or class.

    ``Field`` objects are generated from the class type hints and contains
    the information about the field information.if the user uses
    the ``pytreeclass.field`` to annotate.

    Note:
        - If the class is not annotated, an empty tuple is returned.
        - The ``Field`` generation is cached for class and its bases.
    """
    return tuple(build_field_map(x if isinstance(x, type) else type(x)).values())


def convert_hints_to_fields(klass: type[T]) -> type[T]:
    # convert klass hints to `Field` objects for the current decorated class
    if (hint_map := vars(klass).get("__annotations__", NULL)) is NULL:
        return klass

    for key, hint in hint_map.items():
        if not isinstance(value := vars(klass).get(key, NULL), Field):
            setattr(klass, key, Field(default=value, type=hint, name=key))
    return klass


def build_init_method(klass: type[T]) -> type[T]:
    # generate a code object for the __init__ method and compile it
    # for the given class and return the function object
    body: list[str] = []
    hints: dict[str, str | type | None] = dict()
    head = ["self"]
    heads: dict[str, list[str]] = defaultdict(list)
    has_post = "__post_init__" in vars(klass)

    seen = set()

    for field in (field_map := build_field_map(klass)).values():
        default = "" if field.default is NULL else f"=field_map['{field.name}'].default"

        if field.init:
            if field.kind in ("VAR_POS", "VAR_KW"):
                # disallow multiple `VAR_POS` and `VAR_KW`
                if field.kind in seen:
                    raise TypeError(f"Duplicate {field.kind=} for {field.name=}")
                seen.add(field.kind)

            alias = field.alias or field.name
            hints[field.name] = field.type
            body += [f"self.{field.name}={alias}"]
            heads[field.kind] += [f"{alias}{default}"]
        else:
            body += [f"self.{field.name}{default}"]

    hints["return"] = None
    # add the post init call if the class has it, otherwise add a pass
    # in case all fields are not initialized in the __init__ method
    body += ["self.__post_init__()"] if has_post else ["pass"]

    # organize the arguments order:
    # (POS_ONLY, POS_OR_KW, VAR_POS, KW_ONLY, VAR_KW)
    head += (heads["POS_ONLY"] + ["/"]) if heads["POS_ONLY"] else []
    head += heads["POS_OR_KW"]
    head += ["*" + "".join(heads["VAR_POS"])] if heads["VAR_POS"] else []
    # case for ...(*a, b) and ...(a, *, b)
    head += ["*"] if (heads["KW_ONLY"] and not heads["VAR_POS"]) else []
    head += heads["KW_ONLY"]
    head += ["**" + "".join(heads["VAR_KW"])] if heads["VAR_KW"] else []

    # generate the code for the __init__ method
    code = "def closure(field_map):\n"
    code += f"\tdef __init__({','.join(head)}):"
    code += f"\n\t\t{';'.join(body)}"
    code += f"\n\t__init__.__qualname__ = '{klass.__qualname__}.__init__'"
    code += "\n\treturn __init__"

    exec(code, vars(sys.modules[klass.__module__]), namespace := dict())
    setattr(init := namespace["closure"](field_map), "__annotations__", hints)
    setattr(klass, "__init__", init)
    return klass


@dataclass_transform(field_specifiers=(Field, field))
def autoinit(klass: type[T]) -> type[T]:
    """A class decorator that generates the ``__init__`` method from type hints.

    Similar to ``dataclasses.dataclass``, this decorator generates the ``__init__``
    method for the given class from the type hints or the :func:`field` objects
    set to the class attributes.

    Compared to ``dataclasses.dataclass``, ``autoinit`` with :func:`field` objects
    can be used to apply functions on the field values during initialization,
    and/or support multiple argument kinds.

    Example:
        >>> import pytreeclass as pytc
        >>> @pytc.autoinit
        ... class Tree:
        ...     x: int
        ...     y: int
        >>> tree = Tree(1, 2)
        >>> tree.x, tree.y
        (1, 2)

    Example:
        >>> # define fields with different argument kinds
        >>> import pytreeclass as pytc
        >>> @pytc.autoinit
        ... class Tree:
        ...     kw_only_field: int = pytc.field(default=1, kind="KW_ONLY")
        ...     pos_only_field: int = pytc.field(default=2, kind="POS_ONLY")

    Example:
        >>> # define a converter to apply ``abs`` on the field value
        >>> @pytc.autoinit
        ... class Tree:
        ...     a:int = pytc.field(on_setattr=[abs])
        >>> Tree(a=-1).a
        1

    .. warning::
        - The ``autoinit`` decorator will is no-op if the class already has a
          user-defined ``__init__`` method.

    Note:
        - In case of inheritance, the ``__init__`` method is generated from the
          the type hints of the current class and any base classes that
          are decorated with ``autoinit``.

        >>> import pytreeclass as pytc
        >>> import inspect
        >>> @pytc.autoinit
        ... class Base:
        ...     x: int
        >>> @pytc.autoinit
        ... class Derived(Base):
        ...     y: int
        >>> obj = Derived(x=1, y=2)
        >>> inspect.signature(obj.__init__)
        <Signature (x: int, y: int) -> None>

        - Base classes that are not decorated with ``autoinit`` are ignored during
          synthesis of the ``__init__`` method.

        >>> import pytreeclass as pytc
        >>> import inspect
        >>> class Base:
        ...     x: int
        >>> @pytc.autoinit
        ... class Derived(Base):
        ...     y: int
        >>> obj = Derived(y=2)
        >>> inspect.signature(obj.__init__)
        <Signature (y: int) -> None>

    Note:
        Use ``autoinit`` instead of ``dataclasses.dataclass`` if you want to
        use ``jax.Array`` as a field default value. As ``dataclasses.dataclass``
        will incorrectly raise an error starting from python 3.11 complaining
        that ``jax.Array`` is not immutable.
    """
    return (
        klass
        # if the class already has a user-defined __init__ method
        # then return the class as is without any modification
        if "__init__" in vars(klass)
        # first convert the current class hints to fields
        # then build the __init__ method from the fields of the current class
        # and any base classes that are decorated with `autoinit`
        else build_init_method(convert_hints_to_fields(klass))
    )
