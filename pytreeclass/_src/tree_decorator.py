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
from typing import Any, Callable, NamedTuple, Sequence

_NOT_SET = object()
_FROZEN = "__frozen__"
_FIELD_MAP = "__field_map__"
_POST_INIT = "__post_init__"
_MUTABLE_TYPES = (list, dict, set)
_WRAPPED = "__wrapped__"
_VARS = "__dict__"


def is_treeclass(cls: type) -> bool:
    return hasattr(cls, _FROZEN) and hasattr(cls, _FIELD_MAP)


@ft.lru_cache
def _is_one_arg_func(func: Callable) -> bool:
    return len(inspect.signature(func).parameters) == 1


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
    pos_only: bool = False
    metadata: dict[str, Any] | None = None
    callbacks: Sequence[Callable] | None = None


def field(
    *,
    default: Any = _NOT_SET,
    default_factory: Callable | None = None,
    init: bool = True,
    repr: bool = True,
    kw_only: bool = False,
    pos_only: bool = False,
    metadata: dict[str, Any] | None = None,  # type: ignore
    callbacks: Sequence[Callable] | None = None,
) -> Field:
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

    if kw_only is True and pos_only is True:
        raise ValueError("Cannot specify both `kw_only=True` and `pos_only=True`")

    if not isinstance(metadata, (dict, type(None))):
        raise TypeError("`metadata` must be a dict")

    # check if `callbacks` is a Sequence of functions
    if isinstance(callbacks, Sequence):
        for index, callback in enumerate(callbacks):
            if not isinstance(callback, Callable):  # type: ignore
                msg = f"`callbacks` must be a Sequence of functions, got {type(callbacks)}"
                msg += f" at index={index}"
                raise TypeError(msg)
            if not _is_one_arg_func(callback):
                msg = "`callbacks` must be a Sequence of functions with 1 argument, that takes the value as argument"
                msg += f"got {type(callbacks)} at index={index}"
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
        pos_only=pos_only,
        metadata=metadata,
        callbacks=callbacks,
    )


@ft.lru_cache(maxsize=None)
def _generate_field_map(klass: type) -> dict[str, Field]:
    # get all the fields of the class and its base classes
    # get the fields of the class and its base classes
    FIELD_MAP = dict()

    for base in reversed(klass.__mro__):
        # get the fields of the base class in the MRO
        # in reverse order to ensure the correct order of the fields
        # are preserved, i.e. the fields of the base class are added first
        # and the fields of the derived class are added last so that
        # in case of name collision, the derived class fields are preserved
        if hasattr(base, _FIELD_MAP):
            FIELD_MAP.update(getattr(base, _FIELD_MAP))

    # transform the annotated attributes of the class into Fields
    # while assigning the default values of the Fields to the annotated attributes
    # TODO: use inspect to get annotations, once we are on minimum python version >3.9
    annotations = getattr(klass, _VARS).get("__annotations__", dict())

    for name in annotations:
        # get the value associated with the type hint
        # in essence will skip any non type-hinted attributes
        value = getattr(klass, name, _NOT_SET)
        # at this point we stick to the type hint provided by the user
        # inconsistency between the type hint and the value will be handled later
        type = annotations[name]

        if isinstance(value, Field):
            # the annotated attribute is a `Field``
            # example case: `x: Any = field(default=1)`
            # assign the name and type to the Field from the annotation
            FIELD_MAP[name] = value._replace(name=name, type=type)

        elif value is _NOT_SET:
            # nothing is assigned to the annotated attribute
            # example case: `x: Any`
            # then we create a Field and assign it to the class
            FIELD_MAP[name] = Field(name=name, type=type)

        else:
            # the annotated attribute has a non-field default value
            # check for mutable types and raise an error if found
            if isinstance(value, _MUTABLE_TYPES):
                # example case: `x: Any = [1, 2, 3]`
                # this is the prime motivation for writing this decorator
                # as from python 3.11, jax arrays `dataclasses` will raise an error if
                # `JAX` arrays are used as default values.
                # the `dataclasses` logic is flawed by using `__hash__` existence
                # as a proxy for immutability, which is not the case for `JAX` arrays
                # which are immutable but do not have a `__hash__` method
                msg = f"mutable value= {(value)} is not allowed as a value"
                msg += f" for field `{name}` in class `{klass.__name__}`.\n"
                msg += f" use `field(default_factory=lambda:{value})` instead"
                raise TypeError(msg)

            # example case: `x: int = 1`
            # otherwise, we create a Field and assign default value to the class
            FIELD_MAP[name] = Field(name=name, type=type, default=value)

    return FIELD_MAP


@ft.lru_cache(maxsize=None)
def _generate_init_code(fields: Sequence[Field]):
    # generate the init method code string
    # in here, we generate the function head and body and add `default`/`default_factory`
    # for example, if we have a class with fields `x` and `y`
    # then generated code will something like  `def __init__(self, x, y): self.x = x; self.y = y`
    head = body = ""

    for field in fields:
        key = field.name
        mark0 = f"FIELD_MAP['{key}'].default"
        mark1 = f"self.{key}"

        if field.kw_only and "*" not in head and field.init:
            # if the field is keyword only, and we have not added the `*` yet
            head += "*, "

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

        if field.pos_only and field.init:
            # if the field is positional only, we add a "/" marker after it
            if "/" in head:
                head = head.replace("/,", "")

            head += "/, "

    # in case no field is initialized, we add a pass statement to the body
    # to avoid syntax error in the generated code
    body += "pass"
    # add the body to the head
    body = " def __init__(self, " + head[:-2] + "):" + body
    # use closure to be able to reference default values of all types
    body = f"def closure(FIELD_MAP):\n{body}\n return __init__"
    return body


def _generate_init(klass: type) -> FunctionType:
    # generate the field map for the class
    FIELD_MAP = _generate_field_map(klass)
    # generate init method
    local_namespace = dict()  # type: ignore
    global_namespace = getattr(sys.modules[klass.__module__], _VARS)

    # generate the init method code string
    # in here, we generate the function head and body and add `default`/`default_factory`
    exec(_generate_init_code(FIELD_MAP.values()), global_namespace, local_namespace)
    method = local_namespace["closure"](FIELD_MAP)

    # inject the method into the class namespace
    return FunctionType(
        code=method.__code__,
        globals=global_namespace,
        name=method.__name__,
        argdefs=method.__defaults__,
        closure=method.__closure__,
    )


def _is_dataclass_like(node: Any) -> bool:
    # maybe include other dataclass-like objects here? (e.g. attrs)
    return dc.is_dataclass(node) or is_treeclass(node)


def fields(node: Any) -> Sequence[Field]:
    if not is_treeclass(node):
        raise TypeError(f"Cannot get fields from {node!r}.")
    field_map = getattr(node, _FIELD_MAP, {})
    return tuple(field_map[k] for k in field_map if isinstance(field_map[k], Field))


def _dataclass_like_fields(node):
    """Get the fields of a dataclass-like object."""
    # maybe include other dataclass-like objects here? (e.g. attrs)
    if not _is_dataclass_like(node):
        raise TypeError(f"Cannot get fields from {node!r}.")
    if dc.is_dataclass(node):
        return dc.fields(node)
    return fields(node)
