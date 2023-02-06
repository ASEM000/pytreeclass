# similar to dataclass decorator for init code generation
# the motivation for writing this is to avoid the need to use dataclasses
# especially after this update https://github.com/google/jax/issues/14295
# in essence, after this upadte jax arrays are considred mutable by the field logic

from __future__ import annotations

import dataclasses as dc
import functools as ft
import inspect
import sys
from types import FunctionType
from typing import Any, Callable, NamedTuple

_MISSING = type("MISSING", (), {"__repr__": lambda _: "?"})()
_FROZEN = "__FROZEN__"
# required to get recognized by dataclasses when using `dataclasses.is_dataclass`
_FIELD_MAP = "__dataclass_fields__"


class ImmutableTreeError(Exception):
    """Raised when the tree is immutable"""

    pass


class Field(NamedTuple):
    init: bool = True
    default: Any = _MISSING
    name: str = None
    type: type = Any
    repr: bool = True
    kwonly: bool = False
    default_factory: Any = _MISSING
    # make it get recognized as a dataclass field
    _field_type = dc._FIELD


class NonDiffField(Field):
    pass


def field(
    *,
    default: Any = _MISSING,
    init: bool = True,
    nondiff: bool = False,
    repr: bool = True,
    kwonly: bool = False,
    default_factory: Any = _MISSING,
):
    if default is not _MISSING and default_factory is not _MISSING:
        raise ValueError(" cannot specify both default and default_factory")

    return (NonDiffField if nondiff else Field)(
        default=default,
        name=None,
        type=None,
        repr=repr,
        kwonly=kwonly,
        init=init,
        default_factory=default_factory,
    )


@ft.lru_cache(maxsize=None)
def _generate_field_map(cls) -> dict[str, Field]:
    # get all the fields of the class and its base classes

    # get the fields of the class and its base classes
    field_map = dict()

    for base in reversed(cls.__mro__):
        if hasattr(base, _FIELD_MAP):
            field_map.update(getattr(base, _FIELD_MAP))

    # transform the annotated attributes of the class into Fields
    # while assigning the default values of the Fields to the annotated attributes
    annotations = inspect.get_annotations(cls)

    for name in annotations:
        value = getattr(cls, name, _MISSING)
        type = annotations[name]

        if isinstance(value, Field):
            # the annotated attribute is a Field
            # assign the name and type to the Field from the annotation
            field_map[name] = value._replace(name=name, type=type)

        elif value is _MISSING:
            # nothing is assigned to the annotated attribute
            # then we create a Field and assign it to the class
            field_map[name] = Field(name=name, type=type)

        else:
            # the annotated attribute has a non-field default value
            # check for mutable types and raise an error if found
            if isinstance(value, (list, dict, set)):
                msg = f"mutable type {(value)} is not allowed as a value"
                msg += f" for field `{name}` in class `{cls.__name__}`.\n"
                msg += f" use `field(default_factory=lambda:{value})` instead"
                raise TypeError(msg)

            field_map[name] = Field(name=name, type=type, default=value)

    return field_map


def _patch_init_method(cls):
    # this methods generates and injects the __init__ method into the class namespace

    local_namespace = dict()
    global_namespace = sys.modules[cls.__module__].__dict__

    # patch class with field map
    # designating the field map as class variable is important in case
    field_map = _generate_field_map(cls)
    setattr(cls, _FIELD_MAP, field_map)

    if "__init__" in cls.__dict__:
        # do not generate the init method if the class already has one
        return cls

    # generate the init method code string
    head = body = ""

    for key in field_map:
        field = field_map[key]
        mark0 = f"FIELD_MAP['{key}'].default"
        mark1 = f"self.{key}"

        head += "*, " if (field.kwonly and "*" not in head and field.init) else ""

        if field.default is not _MISSING:
            head += f"{key}={mark0}, " if field.init else ""
            body += f"{mark1}=" + (f"{key}; " if field.init else f"{mark0};")
        elif field.default_factory is not _MISSING:
            head += f"{key}={mark0}_factory(), " if field.init else ""
            body += f"{mark1}=" + (f"{key};" if field.init else f"{mark0}_factory();")
        else:
            head += f"{key}, "
            body += f"{mark1}={key}; "

    body = " def __init__(self, " + head[:-2] + "):" + body
    body = f"def closure(FIELD_MAP):\n{body}\n return __init__"
    exec(body, global_namespace, local_namespace)
    method = local_namespace["closure"](field_map)

    # inject the method into the class namespace
    method = FunctionType(
        code=method.__code__,
        globals=global_namespace,
        name=method.__name__,
        argdefs=method.__defaults__,
        closure=method.__closure__,
    )

    setattr(cls, method.__name__, method)
    return cls
