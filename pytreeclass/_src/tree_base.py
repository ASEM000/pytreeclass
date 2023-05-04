from __future__ import annotations

import abc
from collections.abc import Callable
from contextlib import contextmanager
from typing import Any, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from typing_extensions import dataclass_transform

from pytreeclass._src.code_build import (
    Field,
    _build_field_map,
    _generate_init_method,
    field,
)
from pytreeclass._src.tree_pprint import tree_repr, tree_str
from pytreeclass._src.tree_util import (
    IsLeafType,
    NamedSequenceKey,
    _leafwise_transform,
    _resolve_where,
    is_tree_equal,
    tree_copy,
    tree_hash,
)

"""Define a class that convert a class to a JAX compatible tree structure"""


T = TypeVar("T", bound="TreeClass")
PyTree = Any
EllipsisType = type(Ellipsis)
_no_initializer = object()

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
            ">>> import jax\n"
            ">>> import pytreeclass as pytc\n"
            f"class {type(self.tree).__name__}:(pytc.TreeClass)\n"
            "    ...\n\n"
            f">>> tree = {type(self.tree).__name__}(...)\n"
            ">>> # indexing by boolean pytree\n"
            ">>> mask = jax.tree_map(lambda x: x > 0, tree)\n"
            ">>> tree.at[mask].get()\n\n"
            ">>> # indexing by attribute name\n"
            ">>> tree.at[`attribute_name`].get()\n\n"
            ">>> # indexing by leaf index\n"
            ">>> tree.at[index].get()"
        )

    def get(self, *, is_leaf: IsLeafType = None) -> PyTree:
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
            A PyTree with the leaf values at the specified location set to `set_value`.

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
            # for example tree.at[where].set(tree2) will set all tree leaves to tree2 leaves
            # if tree2 is a pytree of same structure as tree
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
            A PyTree with the leaf values at the specified location set to the result
            of applying `func` to the leaf values.

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
        Call the function at the specified location and return a **copy** of the tree.
        with the result of the function call.

        Returns:
            A tuple of the result of the function call and a copy of the a new instance of
            the tree with the modified values.

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
            If the function mutates the instance, `AttributeError` will be raised.
            Use .at["method_name"](args, kwargs) to call a method that mutates the instance.
        """

        def recursive_getattr(tree: Any, where: tuple[str, ...]):
            if len(where) == 1:
                return getattr(tree, where[0])
            return recursive_getattr(getattr(tree, where[0]), where[1:])

        with _mutable_context(self.tree, kopy=True) as tree:
            value = recursive_getattr(tree, self.where)(*a, **k)  # type: ignore
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

        if len(keys := set(_build_field_map(klass)) - set(vars(self))):
            # handle non-initialized fields
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

    def __init_subclass__(klass: type[T], *a, leafwise: bool = False, **k) -> None:
        super().__init_subclass__(*a, **k)

        if "__setattr__" in vars(klass) or "__delattr__" in vars(klass):
            # conflicting methods with the immutable functionality
            raise TypeError(
                f"Unable to transform the class `{klass.__name__}` "
                "with resereved methods: `__setattr__` or `__delattr__` defined.\n"
                "Reserved `setters` and `deleters` implements "
                "the immutable functionality and cannot be overriden."
            )

        if "__init__" not in vars(klass):
            # generate the init method if not defined similar to `dataclasses.dataclass`
            setattr(klass, "__init__", _generate_init_method(klass))

        if leafwise:
            # transform the class to support leafwise operations
            # useful to use with `bcmap` and creating masks by comparisons.
            klass = _leafwise_transform(klass)

        klass = _register_treeclass(klass)

    def __setattr__(self, key: str, value: Any) -> None:
        if id(self) not in _mutable_instance_registry:
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
                f"Use `.at['{key}'].set(None)` instead."
            )

        getattr(object, "__delattr__")(self, key)

    @property
    def at(self) -> AtIndexer:
        """Immutable out-of-place indexing

        `.at[***].get()`: Return a new instance with the value at the index otherwise None.
        `.at[***].set(value)`: Set the `value` and return a new instance with the updated value.
        `.at[***].apply(func)`: Apply a `func` and return a new instance with the updated value.
        `.at['method'](*a, **k)`: Call a `method` and return a (return value, new instance) tuple.

        `***` acceptable index types are `str` for mapping keys or class attributes, `int`
        for positional indexing, `...` for all leaves, and a boolean mask of the
        same structure as the tree.

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
