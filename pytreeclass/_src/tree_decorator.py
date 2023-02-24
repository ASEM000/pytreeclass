# similar to dataclass decorator for init code generation
# the motivation for writing this is to avoid the need to use dataclasses
# especially after this update https://github.com/google/jax/issues/14295
# in essence, after this upadte jax arrays are considred mutable by the field logic

from __future__ import annotations

import dataclasses as dc
import functools as ft
import sys
from types import FunctionType, MappingProxyType
from typing import Any, Callable, NamedTuple, Sequence

_NOT_SET = type("NOT_SET", (), {"__repr__": lambda _: "?"})()
_FROZEN = "__FROZEN__"
_FIELD_MAP = "__FIELD_MAP__"  # to make it work with `dataclasses.is_dataclass`
_POST_INIT = "__post_init__"
_MUTABLE_TYPES = (list, dict, set)


class Field(NamedTuple):
    # Immutable version of dataclasses.Field
    # with the addition `callbacks` attributes
    name: str | None = None
    type: type | None = None
    default: Any = _NOT_SET
    default_factory: Any = None
    init: bool = True
    repr: bool = True
    kw_only: bool = False
    metadata: MappingProxyType | None = None
    callbacks: Sequence[Callable] | None = None


def field(
    *,
    default: Any | _NOT_SET = _NOT_SET,
    default_factory: Any | None = None,
    init: bool = True,
    repr: bool = True,
    kw_only: bool = False,
    metadata: dict | None = None,
    callbacks: Sequence[Callable] | None = None,
):
    """
    default: The default value of the field.
    default_factory: A 0-argument function called to initialize a field's value.
    init: Whether the field is included in the object's __init__ function.
    repr: Whether the field is included in the object's __repr__ function.
    kw_only: Whether the field is keyword-only.
    metadata: A mapping of user-defined data for the field.
    callbacks: A sequence of functions to call after initialization to modify the field value.
    """

    if default is not _NOT_SET and default_factory is not None:
        # this is the similar behavior to `dataclasses`
        raise ValueError("Cannot specify both `default` and `default_factory`")

    # check metadata
    if isinstance(metadata, dict):
        metadata = MappingProxyType(metadata)
    elif metadata is not None:
        raise TypeError("`metadata` must be a dict")

    # check if `callbacks` is a Sequence of functions
    if isinstance(callbacks, Sequence):
        for index, callback in enumerate(callbacks):
            if not isinstance(callback, Callable):
                msg = f"`callbacks` must be a Sequence of functions, got {type(callbacks)}"
                msg += f" at index={index}"
                raise TypeError(msg)
    elif callbacks is not None:
        msg = f"`callbacks` must be a Sequence of functions, got {type(callbacks)}"
        raise TypeError(msg)

    # set name and type post initialization
    return Field(
        name=None,
        type=None,
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        kw_only=kw_only,
        metadata=metadata,
        callbacks=callbacks,
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
    # TODO: use inspect to get annotations, once we are on minimum python version >3.9
    annotations = cls.__dict__.get("__annotations__", dict())

    for name in annotations:
        # get the value associated with the type hint
        # in essence will skip any non type-hinted attributes
        value = getattr(cls, name, _NOT_SET)
        # at this point we stick to the type hint provided by the user
        # inconsistency between the type hint and the value will be handled later
        type = annotations[name]

        if isinstance(value, Field):
            # the annotated attribute is a `Field``
            # assign the name and type to the Field from the annotation
            field_map[name] = value._replace(name=name, type=type)

        elif value is _NOT_SET:
            # nothing is assigned to the annotated attribute
            # then we create a Field and assign it to the class
            field_map[name] = Field(name=name, type=type)

        else:
            # the annotated attribute has a non-field default value
            # check for mutable types and raise an error if found
            if isinstance(value, _MUTABLE_TYPES):
                msg = f"mutable value= {(value)} is not allowed as a value"
                msg += f" for field `{name}` in class `{cls.__name__}`.\n"
                msg += f" use `field(default_factory=lambda:{value})` instead"
                raise TypeError(msg)

            # otherwise, we create a Field and assign default value to the class
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
    setattr(cls, _FROZEN, True)

    if "__init__" in cls.__dict__:
        # do not generate the init method if the class already has one
        return cls

    # generate the init method code string
    # in here, we generate the function head and body and add `default`/`default_factory`
    head = body = ""

    for key in field_map:
        field = field_map[key]
        mark0 = f"FIELD_MAP['{key}'].default"
        mark1 = f"self.{key}"

        # add keyword marker in we have a `kw_only` field
        head += "*, " if (field.kw_only and "*" not in head and field.init) else ""

        if field.default is not _NOT_SET:
            # we add the default into the function head (def f(.. x= default_value))
            # if the the field require initialization. if not, then omit it from head
            head += f"{key}={mark0}, " if field.init else ""
            # we then add self.x = x for the body function if field is initialized
            # otherwise, define the default value inside the body ( self.x = default_value)
            body += f"{mark1}=" + (f"{key}; " if field.init else f"{mark0};")
        elif field.default_factory is not None:
            # same story for functions as above
            head += f"{key}={mark0}_factory(), " if field.init else ""
            body += f"{mark1}=" + (f"{key};" if field.init else f"{mark0}_factory();")
        else:
            # no defaults are added
            head += f"{key}, " if field.init else ""
            body += f"{mark1}={key}; " if field.init else ""

    # in case no field is initialized, we add a pass statement to the body
    # to avoid syntax error in the generated code
    body += "pass"
    # add the body to the head
    body = " def __init__(self, " + head[:-2] + "):" + body
    # use closure to be able to reference default values of all types
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


def _is_dataclass_like(node):
    # maybe include other dataclass-like objects here?
    return dc.is_dataclass(node) or hasattr(node, _FIELD_MAP)


def _dataclass_like_fields(node):
    """Get the fields of a dataclass-like object."""
    # maybe include other dataclass-like objects here?
    if not _is_dataclass_like(node):
        raise TypeError(f"Cannot get fields from {node!r}.")
    if dc.is_dataclass(node):
        return dc.fields(node)
    return getattr(node, _FIELD_MAP).values()
