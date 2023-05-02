from __future__ import annotations

import abc
import functools as ft
import sys
from collections.abc import Callable, MutableMapping, MutableSequence
from contextlib import contextmanager
from types import FunctionType, MappingProxyType
from typing import Any, NamedTuple, Sequence, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from typing_extensions import dataclass_transform

from pytreeclass._src.tree_pprint import tree_repr, tree_repr_with_trace, tree_str
from pytreeclass._src.tree_util import (
    NamedSequenceKey,
    TraceType,
    _leafwise_transform,
    is_tree_equal,
    tree_copy,
    tree_hash,
    tree_map_with_trace,
)

"""Define a class that convert a class to a JAX compatible tree structure"""


T = TypeVar("T", bound="TreeClass")
PyTree = Any
EllipsisType = type(Ellipsis)
_no_initializer = object()
_NOT_SET = type("NOT_SET", (), {"__repr__": lambda _: "NOT_SET"})()
_MUTABLE_TYPES = (MutableSequence, MutableMapping, set)

# allow methods in mutable context to be called without raising `AttributeError`
# the instances are registered  during initialization and using `at` property with `__call__
# this is done by registering the instance id in a set before entering the
# mutable context and removing it after exiting the context
_mutable_instance_registry: set[int] = set()


@contextmanager
def _mutable_context(tree, *, kopy: bool = False):
    tree = tree_copy(tree) if kopy else tree
    _mutable_instance_registry.add(id(tree))
    yield tree
    _mutable_instance_registry.discard(id(tree))


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
        raise TypeError(
            "`alias` must be a string describing the alias name of the field"
            "in the constructor or `None` if no alias is provided, "
            f"got type=`{type(alias).__name__}` instead."
        )

    if default is not _NOT_SET and factory is not None:
        raise ValueError(
            "`default` and `factory` are mutually exclusive arguments."
            "Use `default` if the value is immutable or a zero-argument function "
            "returning a mutable value in `factory` otherwise.\n"
            f"got {default=} and {factory=}"
        )

    if kw_only is True and pos_only is True:
        raise ValueError(
            "`kw_only` and `pos_only` are mutually exclusive arguments."
            "Use `kw_only` if the field is a keyword-only argument in the constructor"
            " and `pos_only` if the field is positional only, "
            f"got {kw_only=} and {pos_only=}"
        )

    if isinstance(metadata, dict):
        metadata = MappingProxyType(metadata)  # type: ignore
    elif metadata is not None:
        raise TypeError(
            "`metadata` must be a dictionary describing the metadata of the field"
            "or `None` if no metadata is provided, "
            f"got type=`{type(metadata).__name__}` instead."
        )

    if not isinstance(callbacks, Sequence):
        raise TypeError(
            f"`callbacks` must be a Sequence of one-argument functions "
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
                raise TypeError(
                    f"Mutable default value= {value.default} is not allowed"
                    f" for field `{name}` in class `{klass.__name__}`.\n"
                    f" use `field(... ,factory=lambda:{value.default})` instead"
                )

            field_map[name] = value._replace(name=name, type=type)

        elif isinstance(value, _MUTABLE_TYPES):
            # https://github.com/ericvsmith/dataclasses/issues/3
            # example case: `x: Any = [1, 2, 3]`
            raise TypeError(
                f"Mutable value= {(value)} is not allowed"
                f" for field `{name}` in class `{klass.__name__}`.\n"
                f" use `field(... ,factory=lambda:{value})` instead"
            )

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


@ft.lru_cache(maxsize=128)
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
    setattr(method, "__qualname__", f"{klass.__qualname__}.__init__")
    return method


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


def _generate_path_mask(
    tree: PyTree,
    where: tuple[int | str, ...],
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTree:
    # generate a mask for `where` path in `tree`
    # where path is a tuple of indices or keys, for example
    # where=("a",) wil set all leaves of `tree` with key "a" to True and
    # all other leaves to False
    match = False

    def _unpack_entry(entry) -> tuple[Any, ...]:
        # define rule for indexing matching through `at` property
        # for example `jax` internals uses `jtu.GetAttrKey` to index an attribute, however
        # its not ergonomic to use `tree.at[jtu.GetAttrKey("attr")]` to index an attribute
        # instead `tree.at['attr']` is more ergonomic
        if isinstance(entry, jtu.GetAttrKey):
            return (entry.name,)
        if isinstance(entry, jtu.SequenceKey):
            return (entry.idx,)
        if isinstance(entry, (jtu.DictKey, jtu.FlattenedIndexKey)):
            return (entry.key,)
        if isinstance(entry, NamedSequenceKey):
            return (entry.idx, entry.key)
        return (entry,)

    def map_func(path: TraceType, _: Any):
        keys, _ = path

        if len(where) > len(keys):
            # path is shorter than `where` path. for example
            # where=("a", "b") and the current path is ("a",) then
            # the current path is not a match
            return False

        for wi, key in zip(where, keys):
            if wi not in (..., *_unpack_entry(key)):
                return False

        nonlocal match

        return (match := True)

    mask = tree_map_with_trace(map_func, tree, is_leaf=is_leaf)

    if not match:
        raise LookupError(
            f"No leaf match is found for path={'/'.join(map(str, where))}"
            f"for tree with trace:\n{tree_repr_with_trace(tree)}\n"
            f"with mask={mask}"
        )

    return mask


def _combine_maybe_bool_masks(*masks: PyTree) -> PyTree:
    # combine boolean masks with `&` operator if all masks are boolean
    # otherwise raise an error
    def check_and_return_bool_leaf(leaf: Any) -> bool:
        if hasattr(leaf, "dtype"):
            if leaf.dtype == "bool":
                return leaf
            raise TypeError(f"Expected boolean array mask, got {leaf=}")

        if isinstance(leaf, bool):
            return leaf
        raise TypeError(f"Expected boolean mask, got {leaf=}")

    def map_func(*leaves):
        verdict = True
        for leaf in leaves:
            verdict &= check_and_return_bool_leaf(leaf)
        return verdict

    return jtu.tree_map(map_func, *masks)


def _resolve_where(
    tree: PyTree,
    where: tuple[int | str | PyTree | EllipsisType, ...],
    is_leaf: Callable[[Any], bool] | None = None,
):
    mask = None

    if path := [i for i in where if isinstance(i, (int, str, type(...)))]:
        mask = _generate_path_mask(tree, path, is_leaf=is_leaf)

    if maybe_bool_masks := [i for i in where if isinstance(i, type(tree))]:
        all_masks = [mask, *maybe_bool_masks] if mask else maybe_bool_masks
        mask = _combine_maybe_bool_masks(*all_masks)

    return mask


def _recursive_getattr(tree: Any, where: tuple[str, ...]):
    def step(tree: Any, where: tuple[str, ...]):
        if len(where) == 1:
            return getattr(tree, where[0])
        return step(getattr(tree, where[0]), where[1:])

    return step(tree, where)


class AtIndexer(NamedTuple):
    """Adds `.at` indexing abilities to a PyTree.

    Example:
        >>> import jax.tree_util as jtu
        >>> import pytreeclass as pytc
        >>> @jax.tree_util.register_pytree_with_keys_class
        ... class Tree:
        ...    def __init__(self, a, b):
        ...        self.a = a
        ...        self.b = b
        ...    def tree_flatten_with_keys(self):
        ...        return ((jtu.GetAttrKey("a"), self.a), (jtu.GetAttrKey("b"), self.b)), None
        ...    @classmethod
        ...    def tree_unflatten(cls, aux_data, children):
        ...        return cls(*children)
        ...    @property
        ...    def at(self):
        ...        return pytc.AtIndexer(self, where=())
        ...    def __repr__(self) -> str:
        ...        return f"{self.__class__.__name__}(a={self.a}, b={self.b})"

        >>> Tree(1, 2).at["a"].get()
        Tree(a=1, b=None)
    """

    tree: PyTree
    where: tuple[str | int] | PyTree

    def __getitem__(self, where: str | int | PyTree | EllipsisType) -> AtIndexer:
        if isinstance(where, (type(self.tree), str, int, EllipsisType)):
            return AtIndexer(self.tree, (*self.where, where))

        raise NotImplementedError(
            f"Indexing with {type(where).__name__} is not implemented.\n"
            "Example of supported indexing:\n\n"
            f"class {type(self.tree).__name__}:(pytc.TreeClass)\n"
            "    ...\n\n"
            f">>> tree = {type(self.tree).__name__}(...)\n"
            ">>> # indexing by boolean pytree\n"
            ">>> tree.at[tree > 0].get()\n\n"
            ">>> # indexing by attribute name\n"
            ">>> tree.at[`attribute_name`].get()\n\n"
            ">>> # indexing by attribute index\n"
            ">>> tree.at[`attribute_index`].get()"
        )

    def get(self, *, is_leaf: Callable[[Any], bool] | None = None) -> PyTree:
        """Get the leaf values at the specified location.

        Args:
            is_leaf: a predicate function to determine if a value is a leaf.

        Returns:
            A PyTree of leaf values at the specified location, with the non-selected
            leaf values set to None if the leaf is not an array.

        Example:
            >>> import pytreeclass as pytc
            >>> class Tree(pytc.TreeClass):
            ...     a: int
            ...     b: int
            >>> tree = Tree(a=1, b=2)
            >>> tree.at['a'].get()
            Tree(a=1, b=None)
        """
        where = _resolve_where(self.tree, self.where, is_leaf)

        def leaf_get(leaf: Any, where: Any):
            if isinstance(where, (jax.Array, np.ndarray)) and where.ndim != 0:
                return leaf[jnp.where(where)]
            return leaf if where else None

        return jtu.tree_map(leaf_get, self.tree, where, is_leaf=is_leaf)

    def set(
        self,
        set_value: Any,
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> PyTree:
        """Set the leaf values at the specified location.

        Args:
            set_value: the value to set at the specified location.
            is_leaf: a predicate function to determine if a value is a leaf.

        Returns:
            A PyTree with the leaf values at the specified location set to `set_value`.

        Example:
            >>> import pytreeclass as pytc
            >>> class Tree(pytc.TreeClass):
            ...     a: int
            ...     b: int
            >>> tree = Tree(a=1, b=2)
            >>> tree.at['a'].set(100)
            Tree(a=100, b=2)
        """
        where = _resolve_where(self.tree, self.where, is_leaf)

        def leaf_set(leaf: Any, where: Any, set_value: Any):
            if isinstance(where, (jax.Array, np.ndarray)):
                return jnp.where(where, set_value, leaf)
            return set_value if where else leaf

        if jtu.tree_structure(self.tree) == jtu.tree_structure(set_value):
            # do not broadcast set_value if it is a pytree of same structure
            # for example tree.at[where].set(tree2) will set all tree leaves to tree2 leaves
            # if tree2 is a pytree of same structure as tree
            # instead of making each leaf of tree a copy of tree2
            # is design is similar to `numpy` design `Array.at[...].set(Array)`
            return jtu.tree_map(leaf_set, self.tree, where, set_value, is_leaf=is_leaf)

        # set_value is broadcasted to tree leaves
        # for example tree.at[where].set(1) will set all tree leaves to 1
        partial_leaf_set = lambda leaf, where: leaf_set(leaf, where, set_value)
        return jtu.tree_map(partial_leaf_set, self.tree, where, is_leaf=is_leaf)

    def apply(
        self,
        func: Callable[[Any], Any],
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> PyTree:
        """Apply a function to the leaf values at the specified location.

        Args:
            func: the function to apply to the leaf values.
            is_leaf: a predicate function to determine if a value is a leaf.

        Returns:
            A PyTree with the leaf values at the specified location set to the result
            of applying `func` to the leaf values.

        Example:
            >>> import pytreeclass as pytc
            >>> class Tree(pytc.TreeClass):
            ...     a: int
            ...     b: int
            >>> tree = Tree(a=1, b=2)
            >>> tree.at['a'].apply(lambda _: 100)
            Tree(a=100, b=2)
        """
        where = _resolve_where(self.tree, self.where, is_leaf)

        def leaf_apply(leaf: Any, where: bool):
            if isinstance(where, (jax.Array, np.ndarray)):
                return jnp.where(where, func(leaf), leaf)
            return func(leaf) if where else leaf

        return jtu.tree_map(leaf_apply, self.tree, where, is_leaf=is_leaf)

    def reduce(
        self,
        func: Callable[[Any, Any], Any],
        *,
        initializer: Any = _no_initializer,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> Any:
        """Reduce the leaf values at the specified location.

        Args:
            func: the function to reduce the leaf values.
            initializer: the initializer value for the reduction.
            is_leaf: a predicate function to determine if a value is a leaf.

        Returns:
            The result of reducing the leaf values at the specified location.

        Example:
            >>> import pytreeclass as pytc
            >>> class Tree(pytc.TreeClass):
            ...     a: int
            ...     b: int
            >>> tree = Tree(a=1, b=2)
            >>> tree.at[...].reduce(lambda a, b: a + b, initializer=0)
            3
        """
        where = _resolve_where(self.tree, self.where, is_leaf)
        tree = self.tree.at[where].get(is_leaf=is_leaf)  # type: ignore
        if initializer is _no_initializer:
            return jtu.tree_reduce(func, tree)
        return jtu.tree_reduce(func, tree, initializer)

    def __getattr__(self, name: str) -> AtIndexer:
        """Support nested indexing with `.at`"""
        # support nested `.at``
        # for example `.at[A].at[B]` represents model.A.B
        if name == "at":
            # pass the current tree and the current path to the next `.at`
            return AtIndexer(tree=self.tree, where=self.where)

        raise AttributeError(
            f"{type(self).__name__!r} object has no attribute {name!r}\n"
            f"Did you mean to use .at[{name!r}]?"
        )

    def __call__(self, *a, **k) -> tuple[Any, PyTree]:
        """
        Call the function at the specified location and return a **copy** of the tree.
        with the result of the function call.

        Returns:
            A tuple of the result of the function call and a copy of the a new instance of
            the tree with the modified values.

        Example:
            >>> import pytreeclass as pytc
            >>> class Tree(pytc.TreeClass):
            ...     a: int
            ...     def add(self, b):
            ...         self.a += b
            ...         return self.a
            >>> tree = Tree(a=1)
            >>> tree.at['add'](2)
            (3, Tree(a=3))

        Note:
            If the function mutates the instance, `AttributeError` will be raised.
            Use .at["method_name"](args, kwargs) to call a method that mutates the instance.
        """
        with _mutable_context(self.tree, kopy=True) as tree:
            value = _recursive_getattr(tree, self.where)(*a, **k)  # type: ignore
        return value, tree


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
            raise AttributeError(
                f"Found uninitialized fields=({', '.join(keys)}) in class `{klass.__name__}`"
                f"after calling the defined `__init__` method."
                f"Initialize the fields defined in `{klass.__name__}` in the `__init__` method."
                f"Example:\n"
                f">>> class {klass.__name__}(...):\n"
                f"...   field1: int = 1\n"
                f"...   def __init__(self, field1):\n"
                f"...       self.field1 = field1\n"
                "...       ..."
            )
        return self


@dataclass_transform(field_specifiers=(field, Field))
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
            raise TypeError(
                f"Unable to transform the class `{klass.__name__}` "
                "with resereved methods: `__setattr__` or `__delattr__` defined.\n"
                "Reserved `setters` and `deleters` implements the immutable functionality."
                "and cannot be overriden."
            )

        if "__init__" not in vars(klass):
            # generate the init method if not defined similar to dataclass
            setattr(klass, "__init__", _generate_init_method(klass))

        if leafwise:
            # transform the class to support leafwise operations
            # useful to use with `bcmap` and creating masks by comparisons.
            klass = _leafwise_transform(klass)

        klass = _register_treeclass(klass)

    def __setattr__(self, key: str, value: Any) -> None:
        if id(self) not in _mutable_instance_registry:
            raise AttributeError(
                f"Cannot set attribute `{key}` = {value!r} "
                f"on immutable instance of `{type(self).__name__}`.\n"
                f"Use `.at[`{key}`].set({value!r})` instead."
            )

        if key in (field_map := build_field_map(type(self))):
            for callback in field_map[key].callbacks:
                try:
                    value = callback(value)
                except Exception as e:
                    raise type(e)(f"Error for field=`{key}`:\n{e}")

        getattr(object, "__setattr__")(self, key, value)

    def __delattr__(self, key: str) -> None:
        if id(self) not in _mutable_instance_registry:
            raise AttributeError(
                f"Cannot delete attribute `{key}` "
                f"on immutable instance of `{type(self).__name__}`.\n"
                f"Use `.at[`{key}`].set(None)` instead."
            )

        getattr(object, "__delattr__")(self, key)

    @property
    def at(self) -> AtIndexer:
        return AtIndexer(self, where=())

    def __repr__(self) -> str:
        return tree_repr(self)

    def __str__(self) -> str:
        return tree_str(self)

    def __copy__(self):
        return tree_copy(self)

    def __hash__(self) -> int:
        return tree_hash(self)

    def __eq__(self, other: Any) -> bool | jax.Array:
        return is_tree_equal(self, other)
