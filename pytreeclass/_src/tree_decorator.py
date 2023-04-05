from __future__ import annotations

import functools as ft
import sys
from collections import defaultdict
from types import FunctionType, MappingProxyType
from typing import Any, Callable, Mapping, NamedTuple, Sequence, TypeVar

import jax.tree_util as jtu
from typing_extensions import dataclass_transform

from pytreeclass._src.tree_freeze import ImmutableWrapper, tree_hash
from pytreeclass._src.tree_indexer import (
    _conditional_mutable_method,
    _leafwise_transform,
    _mutable_context,
    is_tree_equal,
    tree_copy,
    tree_indexer,
)
from pytreeclass._src.tree_pprint import tree_repr, tree_str
from pytreeclass._src.tree_trace import register_pytree_node_trace

_NOT_SET = type("NOT_SET", (), {"__repr__": lambda _: "?"})()
_MUTABLE_TYPES = (list, dict, set)
_ANNOTATIONS = "__annotations__"
_POST_INIT = "__post_init__"
_VARS = "__dict__"
T = TypeVar("T")


PyTree = Any


"""Define custom frozen `dataclasses.dataclass`-like decorator"""
# similar to dataclass decorator for init code generation
# the motivation for writing this is to avoid the need to use dataclasses
# especially after this update https://github.com/google/jax/issues/14295
# in essence, after this upadte jax arrays are considred mutable by the field logic


# A registry to store the fields of the `treeclass` wrapped classes. fields are a similar concept to
# `dataclasses.Field` but with the addition of `callbacks` attribute
# While `dataclasses` fields are added as a class attribute to the class under `__dataclass_fields__`
# in this implementation, the fields are stored in a `defaultdict` as an extra precaution
# to avoid user-side modification of the fields while maintaining a cleaner namespace
_field_registry: dict[type, Mapping[str, Field]] = defaultdict(dict)


def register_pytree_field_map(klass: type[T], field_map: Mapping[str, Field]) -> None:
    # register the field map of a class to the registry
    # field map is a mapping of field name to `Field` object
    # the fields are used to generate the `__init__`, `__str__`, `__repr__` functions
    # and to define the leaves in `tree_flatten` and `tree_unflatten` functions
    _field_registry[klass].update(field_map)


def is_treeclass(item: Any) -> bool:
    """Returns `True` if a class or instance is a `treeclass`."""
    klass = item if isinstance(item, type) else type(item)
    return klass in _field_registry


class Field(NamedTuple):
    # Immutable version of dataclasses.Field
    # with the addition `callbacks` attributes
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

    Example:
        >>> import pytreeclass as pytc
        >>> @pytc.treeclass
        ... class Foo:
        ...     x: int = pytc.field(callbacks=[lambda x: x + 1])  # value is incremented by 1 after initialization
        >>> foo = Foo(x=1)
        >>> foo.x
        2
    """
    if default is not _NOT_SET and factory is not None:
        # mutually exclusive arguments
        # this is the similar behavior to `dataclasses`
        msg = "`default` and `factory` are mutually exclusive arguments."
        msg += f"got default={default} and factory={factory}"
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
    if not isinstance(callbacks, Sequence):
        msg = f"`callbacks` must be a Sequence of functions, got {type(callbacks)}"
        raise TypeError(msg)

    # sanity check for callbacks
    for index, callback in enumerate(callbacks):
        if not isinstance(callback, Callable):  # type: ignore
            msg = "`callbacks` must be a Sequence of functions, "
            msg += f"got `{type(callbacks).__name__}` at index={index}"
            raise TypeError(msg)

    # set name and type post initialization
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
    )


def fields(item: Any) -> Sequence[Field]:
    """Get the fields of a `treeclass` instance."""
    if (klass := item if isinstance(item, type) else type(item)) not in _field_registry:
        raise TypeError(f"Cannot get fields of {item!r}.")

    return tuple(_field_registry[klass].values())


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

        if name == "self":
            # while `dataclasses` allows `self` as a field name, its confusing
            # and not recommended. so raise an error
            msg = "Field name cannot be `self`."
            raise ValueError(msg)

        if isinstance(value, Field):
            # the annotated attribute is a `Field``
            # example case: `x: Any = field(default=1)`
            # assign the name and type to the Field from the annotation
            if isinstance(value.default, _MUTABLE_TYPES):
                # example case: `x: Any = field(default=[1, 2, 3])`
                # https://github.com/ericvsmith/dataclasses/issues/3
                msg = f"Mutable default value of field `{name}` is not allowed, use "
                msg += f"`factory=lambda: {value.default}` instead."
                raise TypeError(msg)

            field_map[name] = value._replace(name=name, type=type)

        elif isinstance(value, _MUTABLE_TYPES):
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
            msg += f" use `field(... ,factory=lambda:{value})` instead"
            raise TypeError(msg)

        elif value is _NOT_SET:
            # nothing is assigned to the annotated attribute
            # example case: `x: Any`
            # create a Field and assign it to the class
            field_map[name] = Field(name=name, type=type)

        else:
            # example case: `x: int = 1`
            # create a Field and assign default value to the class
            field_map[name] = Field(name=name, type=type, default=value)

    return field_map


@ft.lru_cache
def _generate_init_code(fields: Sequence[Field]):
    # generate the init method code string
    # in here, we generate the function head and body and add `default`/`factory`
    # for example, if we have a class with fields `x` and `y`
    # then generated code will something like  `def __init__(self, x, y): self.x = x; self.y = y`
    head = body = ""

    for field in fields:
        key = field.name

        mark0 = f"field_map['{key}'].default"
        mark1 = f"field_map['{key}'].factory()"
        mark2 = f"self.{key}"

        if field.kw_only and "*" not in head and field.init:
            # if the field is keyword only, and we have not added the `*` yet
            head += "*, "

        if field.default is not _NOT_SET:
            # we add the default into the function head (def f(.. x= default_value))
            # if the the field require initialization. if not, then omit it from head
            head += f"{key}={mark0}, " if field.init else ""
            # we then add self.x = x for the body function if field is initialized
            # otherwise, define the default value inside the body ( self.x = default_value)
            body += f"\t\t{mark2}=" + (f"{key}\n " if field.init else f"{mark0}\n")
        elif field.factory is not None:
            # same story for functions as above
            head += f"{key}={mark1}, " if field.init else ""
            body += f"\t\t{mark2}=" + (f"{key}\n" if field.init else f"{mark1}\n")
        else:
            # no defaults are added
            head += f"{key}, " if field.init else ""
            body += f"\t\t{mark2}={key}\n " if field.init else ""

        if field.pos_only and field.init:
            # if the field is positional only, we add a "/" marker after it
            if "/" in head:
                head = head.replace("/,", "")

            head += "/, "

    # in case no field is initialized, we add a pass statement to the body
    # to avoid syntax error in the generated code
    body += "\t\tpass"
    # add the body to the head
    body = "\tdef __init__(self, " + head[:-2] + "):\n" + body
    # use closure to be able to reference default values of all types
    body = f"def closure(field_map):\n{body}\n\treturn __init__"
    return body.expandtabs(4)


def _generate_init(klass: type) -> FunctionType:
    # generate the field map for the class
    field_map = _generate_field_map(klass)
    # generate init method
    local_namespace = dict()  # type: ignore
    global_namespace = vars(sys.modules[klass.__module__])

    # generate the init method code string
    # in here, we generate the function head and body and add `default`/`factory`
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


@_conditional_mutable_method
def _setattr(self: PyTree, key: str, value: Any) -> None:
    # setattr under `_mutable_context` context otherwise raise an error
    if key in _field_registry[type(self)]:
        # apply the callbacks on setting the value
        # check if the key is a field name
        for callback in _field_registry[type(self)][key].callbacks:
            try:
                # callback is a function that takes the value of the field
                # and returns a modified value
                value = callback(value)
            except Exception as e:
                msg = f"Error for field=`{key}`:\n{e}"
                raise type(e)(msg)

    vars(self)[key] = value  # type: ignore


@_conditional_mutable_method
def _delattr(self, key: str) -> None:
    # delete the attribute under `_mutable_context` context
    # otherwise raise an error
    del vars(self)[key]


def _init_wrapper(init_func: Callable) -> Callable:
    def register_instance_registered_types(self: PyTree) -> None:
        for key in vars(self):
            # auto registers the instance value if it is a registered `treeclass`
            # this behavior is similar to PyTorch behavior in `nn.Module`
            # with `Parameter` class. where registered classes are equivalent to nn.Parameter.
            # the behavior is useful to avoid repetitive code pattern in field definition and
            # and initialization inside init method.
            if key in _field_registry[type(self)]:
                continue

            if type(value := vars(self)[key]) in _field_registry:
                # register the field with `init=False`.the field is not part of
                # the init method signature, but defined in the `__post_init__` method
                field = Field(name=key, type=type(value), init=False)
                register_pytree_field_map(type(self), {key: field})

    @ft.wraps(init_func)
    def wrapper(self, *a, **k) -> None:
        with _mutable_context(self):
            output = init_func(self, *a, **k)

            if post_init_func := getattr(type(self), _POST_INIT, None):
                # to simplify the logic, we call the post init method
                # even if the init method is not code-generated.
                post_init_func(self)

        # handle non-initialized fields
        if len(keys := set(_field_registry[type(self)]) - set(vars(self))) > 0:
            msg = f"Uninitialized fields: ({', '.join(keys)}) "
            msg += f"in class `{type(self).__name__}`"
            raise AttributeError(msg)

        if wrapper.has_run is False:
            # register instance fields once per class
            register_instance_registered_types(self)
            wrapper.has_run = True

        return output

    wrapper.has_run = False
    return wrapper


def _tree_unflatten(klass: type, treedef: Any, leaves: list[Any]):
    """Unflatten rule for `treeclass` to use with `jax.tree_unflatten`."""
    tree = object.__new__(klass)
    # update through vars, to avoid calling the `setattr` method
    # that will check for callbacks.
    # calling `setattr` will trigger any defined callbacks by the user
    # on each unflattening which is not efficient.
    # however it might be useful to constantly check if the updated value is
    # satisfying the constraints defined by the user in the callbacks.
    vars(tree).update(treedef[1])
    vars(tree).update(zip(treedef[0], leaves))
    return tree


def _tree_flatten(tree: PyTree):
    """Flatten rule for `treeclass` to use with `jax.tree_flatten`."""
    static, dynamic = dict(vars(tree)), dict()
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


def _register_treeclass(klass: T) -> T:
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


def _tree_unwrap(value: Any) -> Any:
    # enables the transparent wrapper behavior iniside `treeclass` wrapped classes
    def is_leaf(x: Any) -> bool:
        return isinstance(x, ImmutableWrapper) or type(x) in _field_registry

    def unwrap(value: Any) -> Any:
        return value.unwrap() if isinstance(value, ImmutableWrapper) else value

    return jtu.tree_map(unwrap, value, is_leaf=is_leaf)


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
    @ft.wraps(getattribute_method)
    def wrapper(self, key: str) -> Any:
        value = getattribute_method(self, key)
        return _tree_unwrap(value) if key in getattribute_method(self, _VARS) else value

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


def _treeclass_transform(klass: T) -> T:
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
def treeclass(klass: T, *, leafwise: bool = False) -> T:
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
