from __future__ import annotations

import functools as ft
import sys
from collections.abc import Callable, MutableMapping, MutableSequence
from types import FunctionType, MappingProxyType
from typing import Any, NamedTuple, Sequence

"""Constructor code generation from type annotations."""

PyTree = Any
EllipsisType = type(Ellipsis)
_NOT_SET = type("NOT_SET", (), {"__repr__": lambda _: "NOT_SET"})()
_MUTABLE_TYPES = (MutableSequence, MutableMapping, set)
# https://github.com/google/jax/issues/14295


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
        factory: A 0-argument function called to initialize field value.
            Mutually exclusive with `default`.
        init: Whether the field is included in the object's __init__ function.
        repr: Whether the field is included in the object's __repr__ function.
        kw_only: Whether the field is keyword-only. Mutually exclusive
            with `pos_only`, effectively placing `/` in the constructor.
        pos_only: Whether the field is positional-only. Mutually exclusive
            with `kw_only`, effectively placing `*` in the constructor.
        metadata: A mapping of user-defined data for the field.
        callbacks: A sequence of functions to called on `setattr` during
            initialization to modify the field value.
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
        ...    # this is useful for private attributes that are not supposed
        ...    # to be accessed directly and hide it from the repr
        ...    _id: int = pytc.field(alias="id", repr=False)

        >>> tree = Employee(name="Asem", age=10, id=1)
        >>> print(tree)  # _id is not shown
        Employee(name=Asem, age=10)
        >>> assert tree._id == 1  # this is the private attribute
    """
    if not isinstance(alias, (str, type(None))):
        raise TypeError(
            "`alias` must be a string describing the alias name of the field"
            "in the constructor or `None` if no alias is provided, "
            f"got type=`{type(alias).__name__}` instead."
        )

    if default is not _NOT_SET and factory is not None:
        raise ValueError(
            "`default` and `factory` are mutually exclusive arguments."
            "Use `default` if the value is immutable or a zero-argument "
            "function returning a mutable value in `factory` otherwise.\n"
            f"got {default=} and {factory=}"
        )

    if kw_only is True and pos_only is True:
        raise ValueError(
            "`kw_only` and `pos_only` are mutually exclusive arguments."
            "Use `kw_only` if the field is a keyword-only argument in "
            "the constructor and `pos_only` if the field is positional only, "
            f"got {kw_only=} and {pos_only=}"
        )

    if isinstance(metadata, dict):
        metadata = MappingProxyType(metadata)  # type: ignore
    elif metadata is not None:
        raise TypeError(
            "`metadata` must be a dictionary describing the metadata of "
            "the field or `None` if no metadata is provided, "
            f"got type=`{type(metadata).__name__}` instead."
        )

    if not isinstance(callbacks, Sequence):
        raise TypeError(
            f"`callbacks` must be a Sequence of one-argument function(s) "
            "operating on the field value, and returning a modified value, "
            f"got type `{type(callbacks).__name__}` instead."
        )

    for index, callback in enumerate(callbacks):
        if not isinstance(callback, Callable):  # type: ignore
            raise TypeError(
                f"`callback` must be a one-argument function "
                "operating on the field value, and returning a modified value, "
                f"got type `{type(callback).__name__}` at {index=} instead."
            )

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


@ft.lru_cache(maxsize=128)
def _build_field_map(klass: type) -> MappingProxyType[str, Field]:
    field_map = dict()  # type: dict[str, Field]

    if klass is object:
        return MappingProxyType(field_map)

    for base in reversed(klass.__mro__[1:]):
        field_map.update(_build_field_map(base))

    # TODO: use inspect to get annotations, once min python version >3.9
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
            if isinstance(value.default, _MUTABLE_TYPES):
                # example case: `x: Any = field(default=[1, 2, 3])`
                raise TypeError(
                    f"Mutable default value= {value.default} is not allowed"
                    f" for field `{name}` in class `{klass.__name__}`.\n"
                    f" use `field(... ,factory=lambda:{value.default})` instead"
                )
            # case: `x: Any = field(default=1)`
            field_map[name] = value._replace(name=name, type=type)
        else:
            if isinstance(value, _MUTABLE_TYPES):
                # https://github.com/ericvsmith/dataclasses/issues/3
                # example case: `x: Any = [1, 2, 3]`
                raise TypeError(
                    f"Mutable value= {(value)} is not allowed"
                    f" for field `{name}` in class `{klass.__name__}`.\n"
                    f" use `field(... ,factory=lambda:{value})` instead"
                )
            # case: `x: int = 1` or `x: Any`
            field_map[name] = Field(name=name, type=type, default=value)
    return MappingProxyType(field_map)


def fields(value: Any) -> Sequence[Field]:
    """Returns a tuple of `Field` objects for the given instance or class."""
    if isinstance(value, type):
        return tuple(_build_field_map(value).values())
    return tuple(_build_field_map(type(value)).values())


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
            body += f"\t\tself.{name}="
            body += f"{alias}\n " if field.init else f"{vref}\n"
        elif field.factory is not None:
            vref = f"field_map['{name}'].factory()"
            head += f"{alias}={vref}, " if field.init else ""
            body += f"\t\tself.{name}="
            body += f"{alias}\n" if field.init else f"{vref}\n"
        else:
            head += f"{alias}, " if field.init else ""
            body += f"\t\tself.{name}={alias}\n " if field.init else ""

        if field.pos_only and field.init:
            head = head.replace("/,", "") + "/, "

    body += "\t\tpass"  # add pass if all fields `init=False`
    body = "\tdef __init__(self, " + head[:-2] + "):\n" + body
    body = f"def closure(field_map):\n{body}\n\treturn __init__"
    return body.expandtabs(4)


def _generate_init_method(klass: type) -> FunctionType:
    # get the field map for the class  MappingProxyType[str, Field]
    field_map = _build_field_map(klass)
    # genrate init code string
    init_code = _generate_init_code(tuple(field_map.values()))
    # execute the code in a new namespace using the class module namespace
    global_namespace = vars(sys.modules[klass.__module__])
    local_namespace = dict()
    exec(init_code, global_namespace, local_namespace)  # type: ignore
    # call the closure to get the init method with the field map as closure
    # to avoid hardcoding the field map in the generated code
    method = local_namespace["closure"](field_map)
    # set the qualname to the class name
    setattr(method, "__qualname__", f"{klass.__qualname__}.__init__")
    return method
