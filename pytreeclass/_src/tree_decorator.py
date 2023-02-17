# similar to dataclass decorator for init code generation
# the motivation for writing this is to avoid the need to use dataclasses
# especially after this update https://github.com/google/jax/issues/14295
# in essence, after this upadte jax arrays are considred mutable by the field logic

from __future__ import annotations

import dataclasses as dc
import functools as ft
import sys
from types import FunctionType, MappingProxyType
from typing import Any, Callable, NamedTuple

import numpy as np

from pytreeclass._src.tree_freeze import freeze

_MISSING = type("MISSING", (), {"__repr__": lambda _: "?"})()  # type: ignore
_FROZEN = "__datalcass_frozen__"
_FIELD_MAP = "__dataclass_fields__"  # to make it work with `dataclasses.is_dataclass`
_POST_INIT = "__post_init__"


class Field(NamedTuple):
    name: str | _MISSING = _MISSING
    type: type | _MISSING = _MISSING
    default: Any = _MISSING
    default_factory: Any = _MISSING
    init: bool = True
    repr: bool = True
    kw_only: bool = False
    metadata: MappingProxyType | _MISSING = _MISSING
    frozen: bool = False
    validator: tuple[FunctionType, ...] | _MISSING = _MISSING
    # make it get recognized as a dataclass field
    # to make it work with `dataclasses.fields`
    _field_type = dc._FIELD


def field(
    *,
    default: Any | _MISSING = _MISSING,
    default_factory: Any | _MISSING = _MISSING,
    init: bool = True,
    repr: bool = True,
    kw_only: bool = False,
    metadata: dict | _MISSING = _MISSING,
    frozen: bool = False,
    validator: FunctionType | tuple[FunctionType, ...] | _MISSING = _MISSING,
):
    """
    default: The default value of the field.
    default_factory: A 0-argument function called to initialize a field's value.
    init: Whether the field is included in the object's __init__ function.
    repr: Whether the field is included in the object's __repr__ function.
    kw_only: Whether the field is keyword-only.
    frozen: Whether the field is frozen, if frozen its excluded from `jax` transformations.
    metadata: A mapping of user-defined data for the field.
    validator: A function or tuple of functions to call after initialization to validate the field value.
    """

    if default is not _MISSING and default_factory is not _MISSING:
        # this is the similar behavior to `dataclasses`
        raise ValueError("Cannot specify both `default` and `default_factory`")

    # check metadata
    if isinstance(metadata, dict):
        metadata = MappingProxyType(metadata)
    elif metadata is not _MISSING:
        raise TypeError("`metadata` must be a dict")

    # check if validator is a tuple of functions or a single function
    msg = "`validator` must be a function or a tuple of type `FunctionType`"
    if isinstance(validator, FunctionType):
        validator = (validator,)
    elif isinstance(validator, tuple):
        for _validator in validator:
            if not isinstance(_validator, FunctionType):
                raise TypeError(msg + f", got {_validator}")
    elif validator is not _MISSING:
        raise TypeError(msg + f", got {validator}")

    # set name and type post initialization

    return Field(
        name=_MISSING,
        type=_MISSING,
        default=default,
        default_factory=default_factory,
        init=init,
        repr=repr,
        kw_only=kw_only,
        metadata=metadata,
        frozen=frozen,
        validator=validator,
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
        value = getattr(cls, name, _MISSING)
        type = annotations[name]

        if isinstance(value, Field):
            # the annotated attribute is a Field
            # assign the name and type to the Field from the annotation
            field = value._replace(name=name, type=type)
            # decide to wrap the field in a frozen wrapper
            field_map[name] = freeze(field) if field.frozen else field

        elif value is _MISSING:
            # nothing is assigned to the annotated attribute
            # then we create a Field and assign it to the class
            field_map[name] = Field(name=name, type=type)

        else:
            # the annotated attribute has a non-field default value
            # check for mutable types and raise an error if found
            if isinstance(value, (list, dict, set)):
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

    if "__init__" in cls.__dict__:
        # do not generate the init method if the class already has one
        return cls

    # generate the init method code string
    # in here, we generate the function head and body and add `default`/`default_factory`
    # here, for `validator` will be handled in the `__post_init__` method in `treeclass`
    # and for `frozen` is handled in `_generate_field_map`
    head = body = ""

    for key in field_map:
        field = field_map[key]
        mark0 = f"FIELD_MAP['{key}'].default"
        mark1 = f"self.{key}"

        # add keyword marker in we have a `kw_only` field
        head += "*, " if (field.kw_only and "*" not in head and field.init) else ""

        if field.default is not _MISSING:
            # we add the default into the function head (def f(.. x= default_value))
            # if the the field require initialization. if not, then omit it from head
            head += f"{key}={mark0}, " if field.init else ""
            # we then add self.x = x for the body function if field is initialized
            # otherwise, define the default value inside the body ( self.x = default_value)
            body += f"{mark1}=" + (f"{key}; " if field.init else f"{mark0};")
        elif field.default_factory is not _MISSING:
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


def shape_validator(shape: tuple[int | None | Ellipsis]) -> Callable:
    """A shape validator for numpy arrays

    Args:
        shape : A tuple of ints and Nones, where None represents a dimension that can be of any size

    Returns:
        validator : A function that takes a numpy array and checks if the shape is valid

    Example:
        >>> x = jnp.ones([1,2,3])
        >>> shape_validator((1,2,3))(x)  # no error
        >>> shape_validator((1,2,None))(x)  # no error
        >>> shape_validator((1,2,4))(x)  # ValueError raised because the last dimension is not 4
        >>> shape_validator((1,2,3,4))(x)  # ValueError raised because the shape length is not 3
        >>> shape_validator((1,2,3.))(x)  # TypeError raised because the shape is not a tuple of ints, Nones, or ...
    """

    ellipsis_index = None

    # first lets check the input shape is valid
    for index, dim in enumerate(shape):
        if not isinstance(dim, (int, type(None), type(Ellipsis))):
            msg = f"Expected a tuple of int, None or Ellipsis, got {type(dim)} at index={index}"
            raise TypeError(msg)

        if isinstance(dim, type(Ellipsis)):
            if ellipsis_index is not None:
                raise ValueError(f"Only one Ellipsis is allowed, got {shape}")
            ellipsis_index = index

    def validator(x: Any) -> None:
        if not hasattr(x, "shape"):
            raise TypeError(f"Expected an object with a shape attribute, got {type(x)}")

        new_shape = shape

        if ellipsis_index is not None:
            # replace ellipsis with Nones
            new_shape = list(shape)
            nones = [None] * (len(x.shape) - len(shape) + 1)
            new_shape[ellipsis_index : ellipsis_index + 1] = nones

        if len(new_shape) != len(x.shape):
            msg = f"Shape length mismatch, expected a shape definition of length={len(shape)}, got {len(x.shape)}"
            raise ValueError(msg)

        for index, dim in enumerate(new_shape):
            if dim is not None and dim != x.shape[index]:
                msg = f"Shape mismatch, expected value at dimension={index} to have shape={dim}, "
                msg += f"got shape={x.shape[index]}"
                raise ValueError(msg)

    return validator


def type_validator(types: type | tuple[type, ...]) -> Callable:
    """Returns a validator that checks if the value is an instance of the given types."""
    # convert to tuple if not already
    def validator(x):
        if not isinstance(x, types):
            msg = f"Expected type in ({types}), for value=({x}), but got {type(x)}"
            raise TypeError(msg)

    return validator


def range_validator(min=-float("inf"), max=float("inf")) -> Callable:
    """Returns a validator that checks if the value is in the given range."""

    def validator(x):
        # using numpy to handle jax arrays
        x = np.asarray(x)
        if not np.all(x >= min):
            msg = f"Value is not in range: min is not greater than or equal to {min}, got {x}"
            raise ValueError(msg)
        if not np.all(x <= max):
            msg = f"Value is not in range: max is not less than or equal to {max}, got {x}"
            raise ValueError(msg)

    return validator


def enum_validator(args: Any | tuple[Any, ...]) -> Callable:
    """Returns a validator that checks if the value is in the given set."""
    args = (args,) if not isinstance(args, tuple) else args

    def validator(x):
        if x not in args:
            raise ValueError(f"Expected value in ({args}), for value=({x})")

    return validator


def normalization_validator(atol: float = 1e-2) -> Callable:
    """Returns a validator that checks if the data mean and standard deviation are close to zero and one respectively.

    Args:
        atol: The absolute tolerance for the mean and standard deviation checks.

    Returns:
        validator: A function that takes a numpy array and checks if the data is normalized.

    Example:
        >>> x = jnp.ones([1,2,3])
        >>> normalization_validator()(x)  # no error

        >>> x = jnp.ones([1,2,3]) * 2
        >>> normalization_validator()(x)  # ValueError raised because the mean is not close to zero
    """

    def validator(x) -> None:
        x_mean, x_std = np.mean(x, axis=0), np.std(x, axis=0)
        if not np.allclose(x_mean, 0.0, atol=atol):
            raise ValueError("Data is not normalized: mean is not close to zero")
        if not np.allclose(x_std, 1.0, atol=atol):
            msg = f"Data is not normalized: standard deviation is not close to one, got {x_std}"
            raise ValueError(msg)

    return validator


def invert_validator(func: Callable) -> Callable:
    """Returns a validator that checks if the value is not valid."""

    def validator(x):
        try:
            func(x)
        except Exception:
            # if the validator raises an exception,invert the result
            # meaning the value is valid
            pass
        else:
            # no error was raised, so the value is not valid
            # no we want to raise the original error
            raise Exception(f"Expected value to not be valid, for value=({x})")

    return validator
