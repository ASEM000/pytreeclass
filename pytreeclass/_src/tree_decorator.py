from __future__ import annotations

import dataclasses as dc
import functools as ft
import inspect
import sys
from types import FunctionType, MappingProxyType
from typing import Any, Callable, Mapping, NamedTuple, Sequence

_NOT_SET = type("NOT_SET", (), {"__repr__": lambda self: "?"})()
_FROZEN = "__frozen__"
_FIELD_MAP = "__field_map__"
_POST_INIT = "__post_init__"
_MUTABLE_TYPES = (list, dict, set)
_WRAPPED = "__wrapped__"
_VARS = "__dict__"
_ANNOTATIONS = "__annotations__"

PyTree = Any

"""Scritp to define custom fozen `dataclasses.dataclass`-like"""
# similar to dataclass decorator for init code generation
# the motivation for writing this is to avoid the need to use dataclasses
# especially after this update https://github.com/google/jax/issues/14295
# in essence, after this upadte jax arrays are considred mutable by the field logic


def is_treeclass(tree: Any) -> bool:
    """Returns `True` if a class or instance is a `treeclass`."""
    return hasattr(tree, _FROZEN) and hasattr(tree, _FIELD_MAP)


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
    metadata: Mapping[str, Any] | None = None
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
        # this is the similar behavior to `dataclasses`
        raise ValueError("Cannot specify both `default` and `default_factory`")

    if kw_only is True and pos_only is True:
        raise ValueError("Cannot specify both `kw_only=True` and `pos_only=True`")

    if isinstance(metadata, Mapping):
        metadata = MappingProxyType(metadata)
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
        metadata=metadata,
        callbacks=callbacks,
    )


@ft.lru_cache(maxsize=None)
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
        if hasattr(base, _FIELD_MAP):
            field_map.update(getattr(base, _FIELD_MAP))

    # transform the annotated attributes of the class into Fields
    # while assigning the default values of the Fields to the annotated attributes
    # TODO: use inspect to get annotations, once we are on minimum python version >3.9
    if _ANNOTATIONS not in getattr(klass, _VARS):
        return field_map

    for name in (annotation_map := getattr(klass, _VARS)[_ANNOTATIONS]):
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
            field_map[name] = Field(name=name, type=type, default=value)

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
    field_map = _generate_field_map(klass)
    # generate init method
    local_namespace = dict()  # type: ignore
    global_namespace = getattr(sys.modules[klass.__module__], _VARS)

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


def _new_wrapper(new_func: Callable) -> Callable:
    @ft.wraps(new_func)
    def wrapper(klass: type, *_, **__) -> PyTree:
        tree = new_func(klass)
        for field in getattr(klass, _FIELD_MAP).values():
            if field.default is not _NOT_SET:
                getattr(tree, _VARS)[field.name] = field.default
            elif field.default_factory is not None:
                getattr(tree, _VARS)[field.name] = field.default_factory()

        # set the tree as not frozen to enable
        getattr(tree, _VARS)[_FROZEN] = False
        return tree

    # wrap the original `new_func`, to use it later in `tree_unflatten`
    # to avoid repeating iterating over fields and setting default values
    setattr(wrapper, _WRAPPED, new_func)
    return wrapper


def _init_wrapper(init_func: Callable) -> Callable:
    @ft.wraps(init_func)
    def wrapper(self, *a, **k) -> None:
        getattr(self, _VARS)[_FROZEN] = False
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
            getattr(self, _VARS)[_FROZEN] = False
            output = getattr(type(self), _POST_INIT)(self)

        # handle uninitialized fields
        for field in getattr(self, _FIELD_MAP).values():
            if field.name not in getattr(self, _VARS):
                # at this point, all fields should be initialized
                # in principle, this error will be caught when invoking `repr`/`str`
                # like in `dataclasses` but we raise it here for better error message.
                raise AttributeError(f"field=`{field.name}` is not initialized.")

        # delete the shadowing `__dict__` attribute to
        # restore the frozen behavior
        if _FROZEN in getattr(self, _VARS):
            del getattr(self, _VARS)[_FROZEN]
        return output

    return wrapper


def _setattr(tree: PyTree, key: str, value: Any) -> None:
    # Set the attribute of the tree if the tree is not frozen
    if getattr(tree, _FROZEN):
        msg = f"Cannot set {key}={value!r}. Use `.at['{key}'].set({value!r})` instead."
        raise AttributeError(msg)

    # apply the callbacks on setting the value
    # check if the key is a field name
    if key in getattr(tree, _FIELD_MAP):
        # check if there is a callback associated with the field
        callbacks = getattr(tree, _FIELD_MAP)[key].callbacks

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
    getattr(tree, _VARS)[key] = value

    if hasattr(value, _FIELD_MAP) and (key not in getattr(tree, _FIELD_MAP)):
        field = Field(name=key, type=type(value))  # type: ignore
        # register it to field map, to avoid re-registering it in field_map
        getattr(tree, _FIELD_MAP)[key] = field


def _delattr(tree, key: str) -> None:
    # Delete the attribute if tree is not frozen
    if getattr(tree, _FROZEN):
        raise AttributeError(f"Cannot delete {key}.")
    del getattr(tree, _VARS)[key]


def _is_dataclass_like(tree: Any) -> bool:
    # maybe include other dataclass-like objects here? (e.g. attrs)
    return dc.is_dataclass(tree) or is_treeclass(tree)


def fields(tree: Any) -> Sequence[Field]:
    """Get the fields of a `treeclass` instance."""
    if not is_treeclass(tree):
        raise TypeError(f"Cannot get fields from {tree!r}.")
    field_map = getattr(tree, _FIELD_MAP, {})
    return tuple(field_map[k] for k in field_map if isinstance(field_map[k], Field))


def _dataclass_like_fields(tree):
    """Get the fields of a dataclass-like object."""
    # maybe include other dataclass-like objects here? (e.g. attrs)
    if not _is_dataclass_like(tree):
        raise TypeError(f"Cannot get fields from {tree!r}.")
    if dc.is_dataclass(tree):
        return dc.fields(tree)
    return fields(tree)
