from __future__ import annotations

import dataclasses as dc
import functools as ft
import inspect
import sys
from types import FunctionType, MappingProxyType
from typing import Any, Callable, NamedTuple, Sequence, TypeVar
from weakref import WeakKeyDictionary

_NOT_SET = type("NOT_SET", (), {"__repr__": lambda self: "?"})()
_FROZEN = "__frozen__"
_POST_INIT = "__post_init__"
_MUTABLE_TYPES = (list, dict, set)
_WRAPPED = "__wrapped__"
_ANNOTATIONS = "__annotations__"

T = TypeVar("T")

PyTree = Any

"""Define custom fozen `dataclasses.dataclass`-like decorator"""
# similar to dataclass decorator for init code generation
# the motivation for writing this is to avoid the need to use dataclasses
# especially after this update https://github.com/google/jax/issues/14295
# in essence, after this upadte jax arrays are considred mutable by the field logic


# A registry to store the fields of the `treeclass` wrapped classes. fields are a similar concept to
# `dataclasses.Field` but with the addition of `callbacks` attribute
# While `dataclasses` fields are added as a class attribute to the class under `__dataclass_fields__`
# in this implementation, the fields are stored in a `WeakKeyDictionary` as an extra precaution
# to avoid user-side modification of the fields while maintaining a cleaner namespace
_field_registry: dict[type, Mapping[str, Field]] = WeakKeyDictionary()  # type: ignore


def is_treeclass(klass_or_instance: Any) -> bool:
    """Returns `True` if a class or instance is a `treeclass`."""
    if isinstance(klass_or_instance, type):
        return klass_or_instance in _field_registry
    return type(klass_or_instance) in _field_registry


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
    metadata: MappingProxyType[str, Any] | None = None
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
    Args:
        default: The default value of the field.
        default_factory: A 0-argument function called to initialize a field's value.
        init: Whether the field is included in the object's __init__ function.
        repr: Whether the field is included in the object's __repr__ function.
        kw_only: Whether the field is keyword-only.
        pos_only: Whether the field is positional-only.
        metadata: A mapping of user-defined data for the field.
        callbacks: A sequence of functions to call after initialization to modify the field value.

    Example:
        >>> import pytreeclass as pytc
        >>> @pytc.treeclass
        ... class Foo:
        ...     x: int = pytc.field(callbacks=[lambda x: x + 1])  # value is incremented by 1 after initialization
        >>> foo = Foo(x=1)
        >>> foo.x
        2
    """
    if default is not _NOT_SET and default_factory is not None:
        # mutually exclusive arguments
        # this is the similar behavior to `dataclasses`
        msg = "`default` and `default_factory` are mutually exclusive arguments."
        msg += f"got default={default} and default_factory={default_factory}"
        raise ValueError(msg)

    if kw_only is True and pos_only is True:
        # mutually exclusive arguments
        msg = "`kw_only` and `pos_only` are mutually exclusive arguments."
        msg += f"got kw_only={kw_only} and pos_only={pos_only}"
        raise ValueError(msg)

    if isinstance(metadata, dict):
        metadata = MappingProxyType(metadata)  # type: ignore
    elif metadata is not None:
        raise TypeError("`metadata` must be a Mapping or None")

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
        metadata=metadata,  # type: ignore
        callbacks=callbacks,
    )


def _generate_field_map(klass: type) -> dict[str, Field]:
    # get all the fields of the class and its base classes
    # get the fields of the class and its base classes
    field_map = dict()

    for base in reversed(klass.__mro__):
        # get the fields of the base class in the MRO
        # in reverse order to ensure the correct order of the fields
        # are preserved, i.e. the fields of the base class are added first
        # and the fields of the derived class are added last so that
        # in case of name collision, the derived class fields are preserved
        if base in _field_registry:
            field_map.update(_field_registry[base])

    # transform the annotated attributes of the class into Fields
    # while assigning the default values of the Fields to the annotated attributes
    # TODO: use inspect to get annotations, once we are on minimum python version >3.9
    if _ANNOTATIONS not in vars(klass):
        return field_map

    for name in (annotation_map := vars(klass)[_ANNOTATIONS]):
        # get the value associated with the type hint
        # in essence will skip any non type-hinted attributes
        value = getattr(klass, name, _NOT_SET)
        # at this point we stick to the type hint provided by the user
        # inconsistency between the type hint and the value will be handled later
        type = annotation_map[name]

        if isinstance(value, Field):
            # the annotated attribute is a `Field``
            # example case: `x: Any = field(default=1)`
            # assign the name and type to the Field from the annotation
            if isinstance(value.default, _MUTABLE_TYPES):
                # example case: `x: Any = field(default=[1, 2, 3])`
                # https://github.com/ericvsmith/dataclasses/issues/3
                msg = f"Mutable default value of field `{name}` is not allowed, use "
                msg += f"`default_factory=lambda: {value.default}` instead."
                raise ValueError(msg)

            field_map[name] = value._replace(name=name, type=type)

        elif value is _NOT_SET:
            # nothing is assigned to the annotated attribute
            # example case: `x: Any`
            # then we create a Field and assign it to the class
            field_map[name] = Field(name=name, type=type)

        else:
            # the annotated attribute has a non-field default value
            # check for mutable types and raise an error if found
            if isinstance(value, _MUTABLE_TYPES):
                # https://github.com/ericvsmith/dataclasses/issues/3
                # example case: `x: Any = [1, 2, 3]`
                # this is the prime motivation for writing this decorator
                # as from python 3.11, jax arrays `dataclasses` will raise an error if
                # `JAX` arrays are used as default values.
                # the `dataclasses` logic is flawed by using `__hash__` existence
                # as a proxy for immutability, which is not the case for `JAX` arrays
                # which are immutable but do not have a `__hash__` method
                msg = f"Mutable value= {(value)} is not allowed"
                msg += f" for field `{name}` in class `{klass.__name__}`.\n"
                msg += f" use `field(... ,default_factory=lambda:{value})` instead"
                raise TypeError(msg)

            # example case: `x: int = 1`
            # otherwise, we create a Field and assign default value to the class
            field_map[name] = Field(name=name, type=type, default=value)

    # add the field map to the global registry
    _field_registry[klass] = field_map
    return field_map


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
    field_map = _field_registry[klass]
    # generate init method
    local_namespace = dict()  # type: ignore
    global_namespace = vars(sys.modules[klass.__module__])

    # generate the init method code string
    # in here, we generate the function head and body and add `default`/`default_factory`
    exec(_generate_init_code(field_map.values()), global_namespace, local_namespace)
    method = local_namespace["closure"](field_map)

    # inject the method into the class namespace
    return FunctionType(
        code=method.__code__,
        globals=global_namespace,
        name=method.__name__,
        argdefs=method.__defaults__,
        closure=method.__closure__,
    )


def _init_wrapper(init_func: Callable) -> Callable:
    @ft.wraps(init_func)
    def wrapper(self, *a, **k) -> None:
        vars(self)[_FROZEN] = False

        for field in _field_registry[type(self)].values():
            if field.default is not _NOT_SET:
                vars(self)[field.name] = field.default
            elif field.default_factory is not None:
                vars(self)[field.name] = field.default_factory()

        output = init_func(self, *a, **k)

        # in case __post_init__ is defined then call it
        # after the tree is initialized
        # here, we assume that __post_init__ is a method
        if hasattr(type(self), _POST_INIT):
            # in case we found post_init in super class
            # then defreeze it first and call it
            # this behavior is differet to `dataclasses` with `frozen=True`
            # but similar if `frozen=False`
            # vars(self)[_FROZEN] = False
            # the following code will raise FrozenInstanceError in `dataclasses`
            # but it will work in `treeclass`,
            # i.e. `treeclass` defreezes the tree after `__post_init__`
            # >>> @dc.dataclass(frozen=True)
            # ... class Test:
            # ...    a:int = 1
            # ...    def __post_init__(self):
            # ...        self.b = 1
            vars(self)[_FROZEN] = False
            output = getattr(type(self), _POST_INIT)(self)

        # handle uninitialized fields
        for field in _field_registry[type(self)].values():
            if field.name not in vars(self):
                # at this point, all fields should be initialized
                # in principle, this error will be caught when invoking `repr`/`str`
                # like in `dataclasses` but we raise it here for better error message.
                raise AttributeError(f"field=`{field.name}` is not initialized.")

        # delete the shadowing `__dict__` attribute to
        # restore the frozen behavior
        if _FROZEN in vars(self):
            del vars(self)[_FROZEN]
        return output

    return wrapper


def _setattr(self: PyTree, key: str, value: Any) -> None:
    if getattr(self, _FROZEN):
        # Set the attribute of the tree if the tree is not frozen
        msg = f"Cannot set {key}={value!r}. Use `.at['{key}'].set({value!r})` instead."
        raise AttributeError(msg)

    # apply the callbacks on setting the value
    # check if the key is a field name
    if key in _field_registry[type(self)]:
        # check if there is a callback associated with the field
        callbacks = _field_registry[type(self)][key].callbacks

        if callbacks is not None:
            for callback in callbacks:
                try:
                    # callback is a function that takes the value of the field
                    # and returns a modified value
                    value = callback(value)
                except Exception as e:
                    msg = f"Error for field=`{key}`:\n{e}"
                    raise type(e)(msg)

    # set the value
    vars(self)[key] = value  # type: ignore

    if type(value) in _field_registry and key not in _field_registry[type(self)]:
        # auto registers the instance value if it is a registered `treeclass`
        # this behavior is similar to PyTorch behavior in `nn.Module`
        # with `Parameter` class. where registered classes are equivalent to nn.Parameter.
        _field_registry[type(self)][key] = Field(name=key, type=type(value))


def _delattr(self, key: str) -> None:
    # Delete the attribute if tree is not frozen
    if getattr(self, _FROZEN):
        raise AttributeError(f"Cannot delete {key}.")
    del vars(self)[key]


def fields(klass_or_instance: Any) -> Sequence[Field]:
    """Get the fields of a `treeclass` instance."""
    if not is_treeclass(klass_or_instance):
        raise TypeError(f"Cannot get fields from {klass_or_instance!r}.")

    if isinstance(klass_or_instance, type):
        # if the tree is a class, then return the fields of the class
        field_map = _field_registry[klass_or_instance]
    else:
        # if the tree is an instance, then return the fields of the instance
        field_map = _field_registry[type(klass_or_instance)]

    return tuple(field_map[k] for k in field_map if isinstance(field_map[k], Field))
