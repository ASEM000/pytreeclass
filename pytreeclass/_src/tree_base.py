# Copyright 2023 PyTreeClass authors
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
"""Define a class that convert a class to a JAX compatible tree structure"""

from __future__ import annotations

import abc
import functools as ft
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, Generic, Hashable, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from typing_extensions import Unpack, dataclass_transform

from pytreeclass._src.code_build import (
    Field,
    _build_field_map,
    _build_init_method,
    field,
    fields,
)
from pytreeclass._src.tree_pprint import (
    PPSpec,
    attr_value_pp,
    pp_dispatcher,
    pps,
    tree_repr,
    tree_str,
)
from pytreeclass._src.tree_util import (
    BaseKey,
    EllipsisKey,
    IntKey,
    IsLeafType,
    NamedSequenceKey,
    NameKey,
    _leafwise_transform,
    _resolve_where,
    is_tree_equal,
    tree_copy,
    tree_hash,
)

T = TypeVar("T", bound=Hashable)
PyTree = Any
EllipsisType = type(Ellipsis)
_no_initializer = object()

# allow methods in mutable context to be called without raising `AttributeError`
# the instances are registered  during initialization and using `at`  property
# with `__call__ this is done by registering the instance id in a set before
# entering the mutable context and removing it after exiting the context
_mutable_instance_registry: set[int] = set()


@contextmanager
def _mutable_context(tree, *, kopy: bool = False):
    tree = tree_copy(tree) if kopy else tree
    _mutable_instance_registry.add(id(tree))
    yield tree
    _mutable_instance_registry.discard(id(tree))


def _register_treeclass(klass: type[T]) -> type[T]:
    # handle all registration logic for `treeclass`

    def tree_unflatten(keys: tuple[str, ...], leaves: tuple[Any, ...]) -> T:
        # unflatten rule to use with `jax.tree_unflatten`
        tree = getattr(object, "__new__")(klass)
        vars(tree).update(zip(keys, leaves))
        return tree

    def tree_flatten(tree: T) -> tuple[tuple[Any, ...], tuple[str, ...]]:
        # flatten rule to use with `jax.tree_flatten`
        dynamic = vars(tree)
        return tuple(dynamic.values()), tuple(dynamic.keys())

    def tree_flatten_with_keys(tree: T):
        # flatten rule to use with `jax.tree_util.tree_flatten_with_path`
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
        ...        kva = (jtu.GetAttrKey("a"), self.a)
        ...        kvb = (jtu.GetAttrKey("b"), self.b)
        ...        return (kva, kvb), None
        ...    @classmethod
        ...    def tree_unflatten(cls, aux_data, children):
        ...        return cls(*children)
        ...    @property
        ...    def at(self):
        ...        return pytc.AtIndexer(self)
        ...    def __repr__(self) -> str:
        ...        return f"{self.__class__.__name__}(a={self.a}, b={self.b})"
        >>> Tree(1, 2).at["a"].get()
        Tree(a=1, b=None)
    """

    tree: PyTree
    where: tuple[BaseKey | PyTree] | tuple[()] = ()

    @ft.singledispatchmethod
    def __getitem__(self, where: Any) -> AtIndexer:
        """Index the tree at the specified location.

        Args:
            where: a key or a tree of keys to index the tree.

        Returns:
            A new AtIndexer instance with the specified location.

        Note:
            Use `__getitem__.register` to add conversion logic for custom keys.
            for example, the following code adds support for indexing with
            `str` keys that gets converted to `NameKey`:
                >>> import pytreeclass as pytc
                >>> @pytc.AtIndexer.__getitem__.register(str)
                ... def _(self, where: str) -> AtIndexer:
                ...    return AtIndexer(self.tree, (*self.where, NameKey(where)))
        """
        return AtIndexer(self.tree, (*self.where, where))

    @__getitem__.register(str)
    def _(self, where: str) -> AtIndexer:
        return AtIndexer(self.tree, (*self.where, NameKey(where)))

    @__getitem__.register(int)
    def _(self, where: int) -> AtIndexer:
        return AtIndexer(self.tree, (*self.where, IntKey(where)))

    @__getitem__.register(type(...))
    def _(self, _: EllipsisType) -> AtIndexer:
        return AtIndexer(self.tree, (*self.where, EllipsisKey()))

    def get(self, *, is_leaf: IsLeafType = None) -> PyTree:
        """Get the leaf values at the specified location.

        Args:
            is_leaf: a predicate function to determine if a value is a leaf.

        Returns:
            A PyTree of leaf values at the specified location, with the
            non-selected leaf values set to None if the leaf is not an array.

        Example:
            >>> import pytreeclass as pytc
            >>> class Tree(pytc.TreeClass):
            ...     a: int
            ...     b: int
            >>> tree = Tree(a=1, b=2)
            >>> # get `a` and return a new instance
            >>> # with `None` for all other leaves
            >>> tree.at['a'].get()
            Tree(a=1, b=None)
        """
        where = _resolve_where(self.tree, self.where, is_leaf)

        def leaf_get(leaf: Any, where: Any):
            if isinstance(where, (jax.Array, np.ndarray)) and where.ndim != 0:
                return leaf[jnp.where(where)]
            return leaf if where else None

        return jtu.tree_map(leaf_get, self.tree, where, is_leaf=is_leaf)

    def set(self, set_value: Any, *, is_leaf: IsLeafType = None):
        """Set the leaf values at the specified location.

        Args:
            set_value: the value to set at the specified location.
            is_leaf: a predicate function to determine if a value is a leaf.

        Returns:
            A PyTree with the leaf values at the specified location
            set to `set_value`.

        Example:
            >>> import pytreeclass as pytc
            >>> class Tree(pytc.TreeClass):
            ...     a: int
            ...     b: int
            >>> tree = Tree(a=1, b=2)
            >>> # set `a` and return a new instance
            >>> # with all other leaves unchanged
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
            # for example tree.at[where].set(tree2) will set all tree leaves
            # to tree2 leaves if tree2 is a pytree of same structure as tree
            # instead of making each leaf of tree a copy of tree2
            # is design is similar to `numpy` design `Array.at[...].set(Array)`
            return jtu.tree_map(leaf_set, self.tree, where, set_value, is_leaf=is_leaf)

        # set_value is broadcasted to tree leaves
        # for example tree.at[where].set(1) will set all tree leaves to 1
        partial_leaf_set = lambda leaf, where: leaf_set(leaf, where, set_value)
        return jtu.tree_map(partial_leaf_set, self.tree, where, is_leaf=is_leaf)

    def apply(self, func: Callable[[Any], Any], *, is_leaf: IsLeafType = None):
        """Apply a function to the leaf values at the specified location.

        Args:
            func: the function to apply to the leaf values.
            is_leaf: a predicate function to determine if a value is a leaf.

        Returns:
            A PyTree with the leaf values at the specified location set to
            the result of applying `func` to the leaf values.

        Example:
            >>> import pytreeclass as pytc
            >>> class Tree(pytc.TreeClass):
            ...     a: int
            ...     b: int
            >>> tree = Tree(a=1, b=2)
            >>> # apply to `a` and return a new instance
            >>> # with all other leaves unchanged
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
        is_leaf: IsLeafType = None,
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
        """Support nested indexing"""
        if name == "at":
            # pass the current tree and the current path to the next `.at`
            return AtIndexer(tree=self.tree, where=self.where)

        raise AttributeError(f"`{type(self).__name__!r}` has no attribute {name!r}.")

    def __call__(self, *a, **k) -> tuple[Any, PyTree]:
        """
        Call the function at the specified location and return a **copy**
        of the tree. with the result of the function call.

        Returns:
            A tuple of the result of the function call and a copy of the a
            new instance of the tree with the modified values.

        Example:
            >>> import pytreeclass as pytc
            >>> class Tree(pytc.TreeClass):
            ...     a: int
            ...     def add(self, x:int) -> int:
            ...         self.a += x
            ...         return self.a
            >>> tree = Tree(a=1)
            >>> # call `add` and return a tuple of
            >>> # (return value, new instance)
            >>> tree.at['add'](99)
            (100, Tree(a=100))

        Note:
            - `AttributeError` is raised, If the function mutates the instance.
            - Use .at["method_name"](*, **) to call a method that mutates the instance.
        """

        def recursive_getattr(tree: Any, where: tuple[NameKey, ...]):
            if not isinstance(where[0], NameKey):
                raise TypeError(f"Expected `NameKey` got {type(where[0])!r}.")
            if len(where) == 1:
                return getattr(tree, where[0].name)
            return recursive_getattr(getattr(tree, where[0].name), where[1:])

        with _mutable_context(self.tree, kopy=True) as tree:
            value = recursive_getattr(tree, self.where)(*a, **k)  # type: ignore
        return value, tree


class TreeClassMeta(abc.ABCMeta):
    def __call__(klass: type[T], *a, **k) -> T:
        self = getattr(klass, "__new__")(klass, *a, **k)

        with _mutable_context(self):
            # initialize the instance under the mutable context
            # to allow setting instance attributes without
            # throwing an `AttributeError`
            getattr(klass, "__init__")(self, *a, **k)

        if keys := set(_build_field_map(klass)) - set(vars(self)):
            raise AttributeError(f"Found uninitialized fields {keys}.")
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

    def __init_subclass__(
        klass: type[T],
        *a,
        leafwise: bool = False,
        **k,
    ) -> None:
        if "__setattr__" in vars(klass) or "__delattr__" in vars(klass):
            raise TypeError(
                f"Unable to transform the class `{klass.__name__}` "
                "with resereved methods: `__setattr__` or `__delattr__` "
                "defined.\nReserved `setters` and `deleters` implements "
                "the immutable functionality and cannot be overriden."
            )

        super().__init_subclass__(*a, **k)

        if "__init__" not in vars(klass):
            # generate the init method if not defined similar to `dataclass`
            setattr(klass, "__init__", _build_init_method(klass))

        if leafwise:
            # transform the class to support leafwise operations
            # useful to use with `bcmap` and creating masks by comparisons.
            klass = _leafwise_transform(klass)

        klass = _register_treeclass(klass)

    def __setattr__(self, key: str, value: Any) -> None:
        if id(self) not in _mutable_instance_registry:
            # instance is not under a mutable context
            # mutable context is used for setting instance attributes
            # during initialization and when using the `at` property
            # with call method.
            raise AttributeError(
                f"Cannot set attribute {value=} to `{key=}`  "
                f"on an immutable instance of `{type(self).__name__}`.\n"
                f"Use `.at['{key}'].set({value})` "
                "to set the value immutably.\nExample:\n"
                f">>> tree1 = {type(self).__name__}(...)\n"
                f">>> tree2 = tree1.at['{key}'].set({value!r})\n"
                ">>> assert not tree1 is tree2\n"
                f">>> tree2.{key}\n{value}"
            )

        if key in (field_map := _build_field_map(type(self))):
            # apply field callbacks on the value before setting
            value = field_map[key](value)

        getattr(object, "__setattr__")(self, key, value)

    def __delattr__(self, key: str) -> None:
        if id(self) not in _mutable_instance_registry:
            # instance is not under a mutable context
            raise AttributeError(
                f"Cannot delete attribute `{key}` "
                f"on immutable instance of `{type(self).__name__}`.\n"
                f"Use `.at['{key}'].set(None)` instead."
            )

        getattr(object, "__delattr__")(self, key)

    @property
    def at(self) -> AtIndexer:
        """Immutable out-of-place indexing

        - `.at[***].get()`:
            Return a new instance with the value at the index otherwise None.
        - `.at[***].set(value)`:
            Set the `value` and return a new instance with the updated value.
        - `.at[***].apply(func)`:
            Apply a `func` and return a new instance with the updated value.
        - `.at['method'](*a, **k)`:
            Call a `method` and return a (return value, new instance) tuple.

        `***` acceptable index types are `str` for mapping keys or
        class attributes, `int` for positional indexing, `...` to select all leaves,
        , a boolean mask of the same structure as the tree.

        Example:
            >>> import pytreeclass as pytc
            >>> class Tree(pytc.TreeClass):
            ...     a:int = 1
            ...     b:float = 2.0
            ...     def add(self, x:int) -> int:
            ...         self.a += x
            ...         return self.a
            >>> tree = Tree()
            >>> # get `a` and return a new instance
            >>> # with `None` for all other leaves
            >>> tree.at["a"].get()
            Tree(a=1, b=None)
            >>> # set `a` and return a new instance
            >>> # with all other leaves unchanged
            >>> tree.at["a"].set(100)
            Tree(a=100, b=2.0)
            >>> # apply to `a` and return a new instance
            >>> # with all other leaves unchanged
            >>> tree.at["a"].apply(lambda x: 100)
            Tree(a=100, b=2.0)
            >>> # call `add` and return a tuple of
            >>> # (return value, new instance)
            >>> tree.at["add"](99)
            (100, Tree(a=100, b=2.0))
        """
        return AtIndexer(self)

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


def _frozen_error(opname: str, tree):
    raise NotImplementedError(
        f"Cannot apply `{opname}` operation to a frozen object `{tree!r}`.\n"
        "Unfreeze the object first to apply operations to it\n"
        "Example:\n"
        ">>> import jax\n"
        ">>> import pytreeclass as pytc\n"
        ">>> tree = jax.tree_map(pytc.unfreeze, tree, is_leaf=pytc.is_frozen)"
    )


class _Frozen(Generic[T]):
    __slots__ = ["__wrapped__", "__weakref__"]
    __wrapped__: T

    def __init__(self, x: T) -> None:
        object.__setattr__(self, "__wrapped__", x)

    def __setattr__(self, _, __) -> None:
        raise AttributeError("Cannot assign to frozen instance.")

    def __delattr__(self, _: str) -> None:
        raise AttributeError("Cannot delete from frozen instance.")

    def __repr__(self) -> str:
        return "#" + tree_repr(self.__wrapped__)

    def __str__(self) -> str:
        return "#" + tree_str(self.__wrapped__)

    def __copy__(self) -> _Frozen[T]:
        return _Frozen(tree_copy(self.__wrapped__))

    def __eq__(self, rhs: Any) -> bool | jax.Array:
        if not isinstance(rhs, _Frozen):
            return False
        return is_tree_equal(self.__wrapped__, rhs.__wrapped__)

    def __hash__(self) -> int:
        return tree_hash(self.__wrapped__)

    # raise helpful error message when trying to interact with frozen object
    __add__ = __radd__ = __iadd__ = lambda x, _: _frozen_error("+", x)
    __sub__ = __rsub__ = __isub__ = lambda x, _: _frozen_error("-", x)
    __mul__ = __rmul__ = __imul__ = lambda x, _: _frozen_error("*", x)
    __matmul__ = __rmatmul__ = __imatmul__ = lambda x, _: _frozen_error("@", x)
    __truediv__ = __rtruediv__ = __itruediv__ = lambda x, _: _frozen_error("/", x)
    __floordiv__ = __rfloordiv__ = __ifloordiv__ = lambda x, _: _frozen_error("//", x)
    __mod__ = __rmod__ = __imod__ = lambda x, _: _frozen_error("%", x)
    __pow__ = __rpow__ = __ipow__ = lambda x, _: _frozen_error("**", x)
    __lshift__ = __rlshift__ = __ilshift__ = lambda x, _: _frozen_error("<<", x)
    __rshift__ = __rrshift__ = __irshift__ = lambda x, _: _frozen_error(">>", x)
    __and__ = __rand__ = __iand__ = lambda x, _: _frozen_error("and", x)
    __xor__ = __rxor__ = __ixor__ = lambda x, _: _frozen_error("xor", x)
    __or__ = __ror__ = __ior__ = lambda x, _: _frozen_error("or", x)
    __neg__ = __pos__ = __abs__ = __invert__ = lambda x: _frozen_error("unary op", x)
    __call__ = lambda x, *_, **__: _frozen_error("call", x)


jtu.register_pytree_node(
    nodetype=_Frozen,
    flatten_func=lambda tree: ((), tree),
    unflatten_func=lambda treedef, _: treedef,
)


def freeze(wrapped: _Frozen[T] | T) -> _Frozen[T]:
    """Freeze a value to avoid updating it by `jax` transformations.

    Example:
        >>> import jax
        >>> import pytreeclass as pytc
        >>> import jax.tree_util as jtu
        >>> # Usage with `jax.tree_util.tree_leaves`
        >>> # no leaves for a wrapped value
        >>> jtu.tree_leaves(pytc.freeze(2.))
        []

        >>> # retrieve the frozen wrapper value using `is_leaf=pytc.is_frozen`
        >>> jtu.tree_leaves(pytc.freeze(2.), is_leaf=pytc.is_frozen)
        [#2.0]

        >>> # Usage with `jax.tree_util.tree_map`
        >>> a= [1,2,3]
        >>> a[1] = pytc.freeze(a[1])
        >>> jtu.tree_map(lambda x:x+100, a)
        [101, #2, 103]
    """
    return wrapped if is_frozen(wrapped) else _Frozen(wrapped)  # type: ignore


def is_frozen(wrapped: Any) -> bool:
    """Returns True if the value is a frozen wrapper."""
    return isinstance(wrapped, _Frozen)


def unfreeze(wrapped: _Frozen[T] | T) -> T:
    """Unfreeze `frozen` value, otherwise return the value itself.

    - use `is_leaf=pytc.is_frozen` with `jax.tree_map` to unfreeze a tree.**

    Example:
        >>> import pytreeclass as pytc
        >>> import jax
        >>> frozen_value = pytc.freeze(1)
        >>> pytc.unfreeze(frozen_value)
        1
        >>> # usage with `jax.tree_map`
        >>> frozen_tree = jax.tree_map(pytc.freeze, {"a": 1, "b": 2})
        >>> unfrozen_tree = jax.tree_map(pytc.unfreeze, frozen_tree, is_leaf=pytc.is_frozen)
        >>> unfrozen_tree
        {'a': 1, 'b': 2}
    """
    return getattr(wrapped, "__wrapped__") if is_frozen(wrapped) else wrapped


def is_nondiff(wrapped: Any) -> bool:
    """
    Returns True if the node is a non-differentiable node, and False for if the
    node is of type float, complex number, or a numpy array of floats or
    complex numbers.

    Example:
        >>> import pytreeclass as pytc
        >>> import jax.numpy as jnp
        >>> pytc.is_nondiff(jnp.array(1))  # int array is non-diff type
        True
        >>> pytc.is_nondiff(jnp.array(1.))  # float array is diff type
        False
        >>> pytc.is_nondiff(1)  # int is non-diff type
        True
        >>> pytc.is_nondiff(1.)  # float is diff type
        False

    Note:
        This function is meant to be used with `jax.tree_map` to
        create a mask for non-differentiable nodes in a tree, that can be used
        to freeze the non-differentiable nodes before passing the tree to a
        `jax` transformation.
    """
    if hasattr(wrapped, "dtype") and np.issubdtype(wrapped.dtype, np.inexact):
        return False
    if isinstance(wrapped, (float, complex)):
        return False
    return True


@pp_dispatcher.register(TreeClass)
def treeclass_pp(node: Any, **spec: Unpack[PPSpec]) -> str:
    name = type(node).__name__
    skip = [f.name for f in fields(node) if not f.repr]
    kvs = ((k, v) for k, v in vars(node).items() if k not in skip)
    return name + "(" + pps(kvs, pp=attr_value_pp, **spec) + ")"
