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

"""Define lens-like indexing/masking for pytrees."""

from __future__ import annotations

import abc
import functools as ft
import re
from typing import Any, Callable, Hashable, NamedTuple, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

T = TypeVar("T")
S = TypeVar("S")
PyTree = Any
# TODO: swich to EllipsisType for python 3.10
EllipsisType = TypeVar("EllipsisType")
KeyEntry = TypeVar("KeyEntry", bound=Hashable)
TypeEntry = TypeVar("TypeEntry", bound=type)
TraceEntry = Tuple[KeyEntry, TypeEntry]
KeyPath = Tuple[KeyEntry, ...]
TypePath = Tuple[TypeEntry, ...]
TraceType = Tuple[KeyPath, TypePath]
IsLeafType = Union[None, Callable[[Any], bool]]
_no_initializer = object()


class BaseKey(abc.ABC):
    """Parent class for all match classes.

    - Subclass this class to create custom match keys by implementing
      the `__eq__` method. The ``__eq__`` method should return True if the
      key matches the given path entry and False otherwise. The path entry
      refers to the entry defined in the ``tree_flatten_with_keys`` method of
      the pytree class.

    - Typical path entries are:

        - ``jax.tree_util.GetAttrKey`` for attributes
        - ``jax.tree_util.DictKey`` for mapping keys
        - ``jax.tree_util.SequenceKey`` for sequence indices

    - When implementing the ``__eq__`` method you can use the ``singledispatchmethod``
      to unpack the path entry for example:

        - ``jax.tree_util.GetAttrKey`` -> `key.name`
        - ``jax.tree_util.DictKey`` -> `key.key`
        - ``jax.tree_util.SequenceKey`` -> `key.index`


        See Examples for more details.

    Example:
        >>> # define an match strategy to match a leaf with a given name and type
        >>> import pytreeclass as pytc
        >>> from typing import NamedTuple
        >>> import jax
        >>> class NameTypeContainer(NamedTuple):
        ...     name: str
        ...     type: type
        >>> @jax.tree_util.register_pytree_with_keys_class
        ... class Tree:
        ...    def __init__(self, a, b) -> None:
        ...        self.a = a
        ...        self.b = b
        ...    def tree_flatten_with_keys(self):
        ...        ak = (NameTypeContainer("a", type(self.a)), self.a)
        ...        bk = (NameTypeContainer("b", type(self.b)), self.b)
        ...        return (ak, bk), None
        ...    @classmethod
        ...    def tree_unflatten(cls, aux_data, children):
        ...        return cls(*children)
        ...    @property
        ...    def at(self):
        ...        return pytc.AtIndexer(self)
        >>> tree = Tree(1, 2)
        >>> class MatchNameType(pytc.BaseKey):
        ...    def __init__(self, name, type):
        ...        self.name = name
        ...        self.type = type
        ...    def __eq__(self, other):
        ...        if isinstance(other, NameTypeContainer):
        ...            return other == (self.name, self.type)
        ...        return False
        >>> tree = tree.at[MatchNameType("a", int)].get()
        >>> assert jax.tree_util.tree_leaves(tree) == [1]

    Note:
        - use ``BaseKey.def_alias(type, func)`` to define an index type alias
          for `BaseKey` subclasses. This is useful for convience when
          creating new match strategies.

            >>> import pytreeclass as pytc
            >>> import functools as ft
            >>> from types import FunctionType
            >>> import jax.tree_util as jtu
            >>> # lets define a new match strategy called `FuncKey` that applies
            >>> # a function to the path entry and returns True if the function
            >>> # returns True and False otherwise.
            >>> # for example `FuncKey(lambda x: x.startswith("a"))` will match
            >>> # all leaves that start with "a".
            >>> class FuncKey(pytc.BaseKey):
            ...    def __init__(self, func):
            ...        self.func = func
            ...    @ft.singledispatchmethod
            ...    def __eq__(self, key):
            ...        return self.func(key)
            ...    @__eq__.register(jtu.GetAttrKey)
            ...    def _(self, key: jtu.GetAttrKey):
            ...        # unpack the GetAttrKey
            ...        return self.func(key.name)
            ...    @__eq__.register(jtu.DictKey)
            ...    def _(self, key: jtu.DictKey):
            ...        # unpack the DictKey
            ...        return self.func(key.key)
            ...    @__eq__.register(jtu.SequenceKey)
            ...    def _(self, key: jtu.SequenceKey):
            ...        return self.func(key.index)

            >>> # instead of using ``FuncKey(function)`` we can define an alias
            >>> # for `FuncKey`, for this example we will define any FunctionType
            >>> # as a `FuncKey` by default.
            >>> @pytc.BaseKey.def_alias(FunctionType)
            ... def _(func):
            ...    return FuncKey(func)
            >>> # create a simple pytree
            >>> @pytc.autoinit
            ... class Tree(pytc.TreeClass):
            ...    a: int
            ...    b: str
            >>> tree = Tree(1, "string")
            >>> # now we can use the `FuncKey` alias to match all leaves that
            >>> # are strings and start with "a"
            >>> tree.at[lambda x: isinstance(x, str) and x.startswith("a")].get()
            Tree(a=1, b=None)
    """

    @abc.abstractmethod
    def __eq__(self, entry: KeyEntry) -> bool:
        pass


class IntKey(BaseKey):
    def __init__(self, idx: int) -> None:
        self.idx = idx

    @ft.singledispatchmethod
    def __eq__(self, _: KeyEntry) -> bool:
        return False

    @__eq__.register(int)
    def _(self, other: int) -> bool:
        return self.idx == other

    @__eq__.register(jtu.SequenceKey)
    def _(self, other: jtu.SequenceKey) -> bool:
        return self.idx == other.idx


class NameKey(BaseKey):
    def __init__(self, name: str) -> None:
        self.name = name

    @ft.singledispatchmethod
    def __eq__(self, _: KeyEntry) -> bool:
        return False

    @__eq__.register(str)
    def _(self, other: str) -> bool:
        return self.name == other

    @__eq__.register(jtu.GetAttrKey)
    def _(self, other: jtu.GetAttrKey) -> bool:
        return self.name == other.name

    @__eq__.register(jtu.DictKey)
    def _(self, other: jtu.DictKey) -> bool:
        return self.name == other.key


class EllipsisKey(BaseKey):
    def __init__(self, _):
        del _

    def __eq__(self, _: KeyEntry) -> bool:
        return True


class MultiKey(BaseKey):
    """Match a leaf with multiple keys at the same level."""

    def __init__(self, *keys: tuple[BaseKey, ...]):
        self.keys = tuple(keys)

    def __eq__(self, entry) -> bool:
        return any(entry == key for key in self.keys)


class RegexKey(BaseKey):
    """Match a leaf with a regex pattern inside 'at' property.

    Args:
        pattern: regex pattern to match.

    Example:
        >>> import pytreeclass as pytc
        >>> import re
        >>> @pytc.autoinit
        ... class Tree(pytc.TreeClass):
        ...     weight_1: float = 1.0
        ...     weight_2: float = 2.0
        ...     weight_3: float = 3.0
        ...     bias: float = 0.0
        >>> tree = Tree()
        >>> tree.at[re.compile(r"weight_.*")].set(100.0)  # set all weights to 100.0
        Tree(weight_1=100.0, weight_2=100.0, weight_3=100.0, bias=0.0)
    """

    def __init__(self, pattern: str) -> None:
        self.pattern = pattern

    @ft.singledispatchmethod
    def __eq__(self, _: KeyEntry) -> bool:
        return False

    @__eq__.register(str)
    def _(self, other: str) -> bool:
        return re.fullmatch(self.pattern, other) is not None

    @__eq__.register(jtu.GetAttrKey)
    def _(self, other) -> bool:
        return re.fullmatch(self.pattern, other.name) is not None

    @__eq__.register(jtu.DictKey)
    def _(self, other) -> bool:
        return re.fullmatch(self.pattern, other.key) is not None


# dispatch on type of indexer to convert input item to at indexer
# `__getitem__` to the appropriate key
# avoid using container pytree types to avoid conflict between
# matching as a mask or as an instance of `BaseKey`
indexer_dispatcher = ft.singledispatch(lambda x: x)
indexer_dispatcher.register(type(...), EllipsisKey)
indexer_dispatcher.register(int, IntKey)
indexer_dispatcher.register(str, NameKey)
indexer_dispatcher.register(re.Pattern, RegexKey)

BaseKey.def_alias = indexer_dispatcher.register


_NOT_IMPLEMENTED_INDEXING = """Indexing with {} is not implemented, supported indexing types are:
- `str` for mapping keys or class attributes.
- `int` for positional indexing for sequences.
- `...` to select all leaves.
- Boolean mask of the same structure as the tree
- `re.Pattern` to index all keys matching a regex pattern.
- Instance of `BaseKey` with custom logic to index a pytree.
- `tuple` of the above types to match multiple leaves at the same level.
"""


def _generate_path_mask(
    tree: PyTree,
    where: tuple[BaseKey, ...],
    is_leaf: IsLeafType = None,
) -> PyTree:
    # generate a boolean mask for `where` path in `tree`
    # where path is a tuple of indices or keys, for example
    # where=("a",) wil set all leaves of `tree` with key "a" to True and
    # all other leaves to False
    match = False

    def map_func(path, _: Any):
        if len(where) > len(path):
            # path is shorter than `where` path. for example
            # where=("a", "b") and the current path is ("a",) then
            # the current path is not a match
            return False
        for wi, ki in zip(where, path):
            if not (wi == ki):
                return False

        nonlocal match
        match = True
        return match

    mask = jtu.tree_map_with_path(map_func, tree, is_leaf=is_leaf)

    if not match:
        raise LookupError(f"No leaf match is found for {where=}.")

    return mask


def _combine_bool_leaves(*leaves):
    verdict = True
    for leaf in leaves:
        verdict &= leaf
    return verdict


def _is_bool_leaf(leaf: Any) -> bool:
    if hasattr(leaf, "dtype"):
        return leaf.dtype == "bool"
    return isinstance(leaf, bool)


def _resolve_where(
    tree: T,
    where: tuple[Any, ...],  # type: ignore
    is_leaf: IsLeafType = None,
) -> T | None:
    mask = None
    bool_masks: list[T] = []
    path_masks: list[BaseKey] = []
    treedef0 = jtu.tree_structure(tree, is_leaf=is_leaf)
    seen_tuple = False  # handle multiple keys at the same level
    level_paths = []

    def verify_and_aggregate_is_leaf(x) -> bool:
        # use is_leaf with non-local to traverse the tree depth-first manner
        # required for verifying if a pytree is a valid indexing pytree
        nonlocal seen_tuple, level_paths, bool_masks
        # used to check if a pytree is a valid indexing pytree
        # used with `is_leaf` argument of any `jtu.tree_*` function
        leaves, treedef = jtu.tree_flatten(x)

        if treedef == treedef0 and all(map(_is_bool_leaf, leaves)):
            # boolean pytrees of same structure as `tree` is a valid indexing pytree
            bool_masks += [x]
            return True

        if isinstance(resolved_key := indexer_dispatcher(x), BaseKey):
            # valid resolution of `BaseKey` is a valid indexing leaf
            # makes it possible to dispatch on multi-leaf pytree
            level_paths += [resolved_key]
            return False

        if type(x) is tuple and seen_tuple is False:
            # e.g. `at[1,2,3]` but not `at[1,(2,3)]``
            seen_tuple = True
            return False

        # not a container of other keys or a pytree of same structure
        raise NotImplementedError(_NOT_IMPLEMENTED_INDEXING.format(x))

    for level_keys in where:
        # each for loop iteration is a level in the where path
        jtu.tree_leaves(level_keys, is_leaf=verify_and_aggregate_is_leaf)
        path_masks += [MultiKey(*level_paths)] if len(level_paths) > 1 else level_paths
        level_paths = []
        seen_tuple = False

    if path_masks:
        mask = _generate_path_mask(tree, path_masks, is_leaf=is_leaf)

    if bool_masks:
        all_masks = [mask, *bool_masks] if mask else bool_masks
        mask = jax.tree_map(_combine_bool_leaves, *all_masks)

    return mask


class AtIndexer(NamedTuple):
    """Index a pytree at a given path using a path or mask.

    Args:
        tree: pytree to index
        where: one of the following:

            - ``str`` for mapping keys or class attributes.
            - ``int`` for positional indexing for sequences.
            - ``...`` to select all leaves.
            - a boolean mask of the same structure as the tree
            - ``re.Pattern`` to index all keys matching a regex pattern.
            - an instance of ``BaseKey`` with custom logic to index a pytree.
            - a tuple of the above to match multiple keys at the same level.

    Example:
        >>> # use `AtIndexer` on a pytree (e.g. dict,list,tuple,etc.)
        >>> import jax
        >>> import pytreeclass as pytc
        >>> tree = {"level1_0": {"level2_0": 100, "level2_1": 200}, "level1_1": 300}
        >>> tree = pytc.AtIndexer(tree)
        >>> tree.at["level1_0"].at["level2_0"].get()
        {'level1_0': {'level2_0': 100, 'level2_1': None}, 'level1_1': None}
        >>> # get multiple keys at once at the same level
        >>> tree.at["level1_0"].at["level2_0", "level2_1"].get()
        {'level1_0': {'level2_0': 100, 'level2_1': 200}, 'level1_1': None}
        >>> # get with a mask
        >>> mask = {"level1_0": {"level2_0": True, "level2_1": False}, "level1_1": True}
        >>> tree.at[mask].get()
        {'level1_0': {'level2_0': 100, 'level2_1': None}, 'level1_1': 300}

    Example:
        >>> # use ``AtIndexer`` in a class
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

    def __getitem__(self, where: Any) -> AtIndexer:
        return type(self)(self.tree, (*self.where, where))

    def __getattr__(self, name: str) -> AtIndexer:
        """Support nested indexing."""
        if name == "at":
            # pass the current tree and the current path to the next `.at`
            return type(self)(tree=self.tree, where=self.where)

        raise AttributeError(f"`{self!r}` has no attribute {name!r}.")

    def get(self, *, is_leaf: IsLeafType = None) -> PyTree:
        """Get the leaf values at the specified location.

        Args:
            is_leaf: a predicate function to determine if a value is a leaf.

        Returns:
            A _new_ pytree of leaf values at the specified location, with the
            non-selected leaf values set to None if the leaf is not an array.

        Example:
            >>> import pytreeclass as pytc
            >>> tree = {"level1_0": {"level2_0": 100, "level2_1": 200}, "level1_1": 300}
            >>> tree = pytc.AtIndexer(tree)
            >>> tree.at["level1_0"].at["level2_0"].get()
            {'level1_0': {'level2_0': 100, 'level2_1': None}, 'level1_1': None}

        Example:
            >>> import pytreeclass as pytc
            >>> @pytc.autoinit
            ... class Tree(pytc.TreeClass):
            ...     a: int
            ...     b: int
            >>> tree = Tree(a=1, b=2)
            >>> # get ``a`` and return a new instance
            >>> # with ``None`` for all other leaves
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
            A pytree with the leaf values at the specified location
            set to ``set_value``.

        Example:
            >>> import pytreeclass as pytc
            >>> tree = {"level1_0": {"level2_0": 100, "level2_1": 200}, "level1_1": 300}
            >>> tree = pytc.AtIndexer(tree)
            >>> tree.at["level1_0"].at["level2_0"].set('SET')
            {'level1_0': {'level2_0': 'SET', 'level2_1': 200}, 'level1_1': 300}

        Example:
            >>> import pytreeclass as pytc
            >>> @pytc.autoinit
            ... class Tree(pytc.TreeClass):
            ...     a: int
            ...     b: int
            >>> tree = Tree(a=1, b=2)
            >>> # set ``a`` and return a new instance
            >>> # with all other leaves unchanged
            >>> tree.at['a'].set(100)
            Tree(a=100, b=2)
        """
        where = _resolve_where(self.tree, self.where, is_leaf)

        def leaf_set(leaf: Any, where: Any, set_value: Any):
            if isinstance(where, (jax.Array, np.ndarray)):
                return jnp.where(where, set_value, leaf)
            return set_value if where else leaf

        if jtu.tree_structure(self.tree, is_leaf) == jtu.tree_structure(set_value):
            # do not broadcast set_value if it is a pytree of same structure
            # for example tree.at[where].set(tree2) will set all tree leaves
            # to tree2 leaves if tree2 is a pytree of same structure as tree
            # instead of making each leaf of tree a copy of tree2
            # is design is similar to ``numpy`` design `Array.at[...].set(Array)`
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
            A pytree with the leaf values at the specified location set to
            the result of applying ``func`` to the leaf values.

        Example:
            >>> import pytreeclass as pytc
            >>> tree = {"level1_0": {"level2_0": 100, "level2_1": 200}, "level1_1": 300}
            >>> tree = pytc.AtIndexer(tree)
            >>> tree.at["level1_0"].at["level2_0"].apply(lambda _: 'SET')
            {'level1_0': {'level2_0': 'SET', 'level2_1': 200}, 'level1_1': 300}

        Example:
            >>> import pytreeclass as pytc
            >>> @pytc.autoinit
            ... class Tree(pytc.TreeClass):
            ...     a: int
            ...     b: int
            >>> tree = Tree(a=1, b=2)
            >>> # apply to ``a`` and return a new instance
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

    def scan(
        self,
        func: Callable[[Any, S], tuple[Any, S]],
        state: S,
        *,
        is_leaf: IsLeafType = None,
    ) -> tuple[PyTree, S]:
        """Apply a function with carrtying a state.

        Args:
            func: the function to apply to the leaf values. the function accepts
                a running state and leaf value and returns a tuple of the new
                leaf value and the new state.
            state: the initial state to carry.
            is_leaf: a predicate function to determine if a value is a leaf. for
                example, ``lambda x: isinstance(x, list)`` will treat all lists
                as leaves and will not recurse into list items.

        Returns:
            A tuple of the final state and pytree with the leaf values at the
            specified location set to the result of applying ``func`` to the leaf
            values.

        Example:
            >>> import pytreeclass as pytc
            >>> tree = {"level1_0": {"level2_0": 100, "level2_1": 200}, "level1_1": 300}
            >>> def scan_func(leaf, state):
            ...     return 'SET', state + 1
            >>> init_state = 0
            >>> tree = pytc.AtIndexer(tree)
            >>> tree.at["level1_0"].at["level2_0"].scan(scan_func, state=init_state)
            ({'level1_0': {'level2_0': 'SET', 'level2_1': 200}, 'level1_1': 300}, 1)

        Example:
            >>> import pytreeclass as pytc
            >>> from typing import NamedTuple
            >>> class State(NamedTuple):
            ...     func_evals: int = 0
            >>> @pytc.autoinit
            ... class Tree(pytc.TreeClass):
            ...     a: int
            ...     b: int
            ...     c: int
            >>> tree = Tree(a=1, b=2, c=3)
            >>> def scan_func(leaf, state: State):
            ...     state = State(state.func_evals + 1)
            ...     return leaf + 1, state
            >>> # apply to ``a`` and ``b`` and return a new instance with all other
            >>> # leaves unchanged and the new state that counts the number of
            >>> # function evaluations
            >>> tree.at['a','b'].scan(scan_func, state=State())
            (Tree(a=2, b=3, c=3), State(func_evals=2))

        Note:
            ``scan`` applies a binary ``func`` to the leaf values while carrying
            a state and returning a tree leaves with the the ``func`` applied to
            them with final state. While ``reduce`` applies a binary ``func`` to the
            leaf values while carrying a state and returning a single value.
        """
        where = _resolve_where(self.tree, self.where, is_leaf)

        running_state = state

        def stateless_func(leaf):
            nonlocal running_state
            leaf, running_state = func(leaf, running_state)
            return leaf

        def leaf_apply(leaf: Any, where: bool):
            if isinstance(where, (jax.Array, np.ndarray)):
                return jnp.where(where, stateless_func(leaf), leaf)
            return stateless_func(leaf) if where else leaf

        out = jtu.tree_map(leaf_apply, self.tree, where, is_leaf=is_leaf)
        return out, running_state

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

        Note:
            - If ``initializer`` is not specified, the first leaf value is used as
              the initializer.
            - ``reduce`` applies a binary ``func`` to each leaf values while accumulating
              a state a returns the final result. while ``scan`` applies ``func`` to each
              leaf value while carrying a state and returns the final state and
              the leaves of the tree with the result of applying ``func`` to each leaf.

        Example:
            >>> import pytreeclass as pytc
            >>> @pytc.autoinit
            ... class Tree(pytc.TreeClass):
            ...     a: int
            ...     b: int
            >>> tree = Tree(a=1, b=2)
            >>> tree.at[...].reduce(lambda a, b: a + b, initializer=0)
            3
        """
        where = _resolve_where(self.tree, self.where, is_leaf)
        tree = self.at[where].get(is_leaf=is_leaf)  # type: ignore
        if initializer is _no_initializer:
            return jtu.tree_reduce(func, tree)
        return jtu.tree_reduce(func, tree, initializer)
