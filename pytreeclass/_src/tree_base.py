# Copyright 2023 pytreeclass authors
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

"""Define a class that convert a class to a JAX compatible tree structure."""

from __future__ import annotations

import abc
from typing import Any, Hashable, TypeVar

import jax
import jax.tree_util as jtu
from typing_extensions import Unpack

from pytreeclass._src.code_build import fields
from pytreeclass._src.tree_index import AtIndexer
from pytreeclass._src.tree_pprint import (
    PPSpec,
    attr_value_pp,
    pp_dispatcher,
    pps,
    tree_repr,
    tree_str,
)
from pytreeclass._src.tree_util import (
    NamedSequenceKey,
    is_tree_equal,
    tree_copy,
    tree_hash,
)

T = TypeVar("T", bound=Hashable)
S = TypeVar("S")
PyTree = Any
EllipsisType = type(Ellipsis)


# allow methods in mutable registry to be called without raising `AttributeError`
_mutable_instance_registry: set[int] = set()


def add_mutable_entry(node) -> None:
    _mutable_instance_registry.add(id(node))


def discard_mutable_entry(node) -> None:
    _mutable_instance_registry.discard(id(node))


def recursive_getattr(tree: Any, where: tuple[str, ...]):
    if not isinstance(where[0], str):
        raise TypeError(f"Expected string, got {type(where[0])!r}.")
    if len(where) == 1:
        return getattr(tree, where[0])
    return recursive_getattr(getattr(tree, where[0]), where[1:])


class TreeClassIndexer(AtIndexer):
    def __call__(self, *a, **k) -> tuple[Any, PyTree]:
        """Call a method on the tree instance and return result and new instance.

        Returns:
            A tuple of the result of the function call and a copy of the a
             new instance of the tree with the modified values.

        Example:
            >>> import pytreeclass as pytc
            >>> @pytc.autoinit
            ... class Tree(pytc.TreeClass):
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
        tree = tree_copy(self.tree)
        jtu.tree_map(lambda _: _, tree, is_leaf=add_mutable_entry)
        value = recursive_getattr(tree, self.where)(*a, **k)  # type: ignore
        jtu.tree_map(lambda _: _, tree, is_leaf=discard_mutable_entry)
        return value, tree


class TreeClassMeta(abc.ABCMeta):
    def __call__(klass: type[T], *a, **k) -> T:
        tree = getattr(klass, "__new__")(klass, *a, **k)
        add_mutable_entry(tree)
        getattr(klass, "__init__")(tree, *a, **k)
        discard_mutable_entry(tree)
        return tree


class TreeClass(metaclass=TreeClassMeta):
    """Convert a class to a ``jax``-compatible pytree by inheriting from :class:`.TreeClass`.

    A pytree is any nested structure that can be used with ``jax`` functions.
    A pytree can be a container or a leaf. Container examples are: a ``tuple``,
    ``list``, or ``dict``. A leaf is a non-container data structure like an
    ``int``, ``float``, ``string``, or ``jax.Array``. :class:`.TreeClass` is a
    container pytree that holds other pytrees in its attributes.

    Note:
        ``pytreeclass`` offers two methods to define the ``__init__`` method:

        1. Manual ``__init__`` method

           >>> import pytreeclass as pytc
           >>> class Tree(pytc.TreeClass):
           ...     def __init__(self, a:int, b:float):
           ...         self.a = a
           ...         self.b = b
           >>> tree = Tree(a=1, b=2.0)

        2. Auto generated ``__init__`` method

           Either by ``dataclasses.dataclasss`` or by using :func:`.autoinit` decorator
           where the type annotations are used to generate the ``__init__`` method
           similar to ``dataclasses.dataclass``. Compared to ``dataclasses.dataclass``,
           ``autoinit`` with :func:`field` objects can be used to apply functions on
           the field values during initialization, and/or support multiple argument kinds.
           For more details see :func:`.autoinit` and :func:`.field`.

           >>> import pytreeclass as pytc
           >>> @pytc.autoinit
           ... class Tree(pytc.TreeClass):
           ...     a:int
           ...     b:float
           >>> tree = Tree(a=1, b=2.0)

    Note:
        Leaf-wise math operations are supported  using ``leafwise`` decorator.
        ``leafwise`` decorator applies math operations to each leaf of the tree.
        for example:

        >>> @pytc.leafwise
        ... @pytc.autoinit
        ... class Tree(pytc.TreeClass):
        ...     a:int = 1
        ...     b:float = 2.0
        >>> tree = Tree()
        >>> tree + 1
        Tree(a=2, b=3.0)

    Note:
        Advanced indexing is supported using ``at`` property. Indexing can be
        used to ``get``, ``set``, or ``apply`` a function to a leaf or a group of
        leaves using ``leaf`` name, index or by a boolean mask.

        >>> @pytc.autoinit
        ... class Tree(pytc.TreeClass):
        ...     a:int = 1
        ...     b:float = 2.0
        >>> tree = Tree()
        >>> tree.at["a"].get()
        Tree(a=1, b=None)
        >>> tree.at[0].get()
        Tree(a=1, b=None)

    Note:
        - Under ``jax.tree_util.***`` all :class:`.TreeClass` attributes are
          treated as leaves.
        - To hide/ignore a specific attribute from the tree leaves, during
          ``jax.tree_util.***`` operations, freeze the leaf using :func:`.freeze`
          or :func:`.tree_mask`.

        >>> # freeze(exclude) a leaf from the tree leaves:
        >>> import jax
        >>> import pytreeclass as pytc
        >>> @pytc.autoinit
        ... class Tree(pytc.TreeClass):
        ...     a:int = 1
        ...     b:float = 2.0
        >>> tree = Tree()
        >>> tree = tree.at["a"].apply(pytc.freeze)
        >>> jax.tree_util.tree_leaves(tree)
        [2.0]

        >>> # undo the freeze
        >>> tree = tree.at["a"].apply(pytc.unfreeze, is_leaf=pytc.is_frozen)
        >>> jax.tree_util.tree_leaves(tree)
        [1, 2.0]

        >>> # using `tree_mask` to exclude a leaf from the tree leaves
        >>> freeze_mask = Tree(a=True, b=False)
        >>> jax.tree_util.tree_leaves(pytc.tree_mask(tree, freeze_mask))
        [2.0]

    Note:
        - :class:`.TreeClass` inherits from ``abc.ABC`` so ``@abstract...`` decorators
          can be used to define abstract behavior.

    Warning:
        The structure should be organized as a tree. In essence, *cyclic references*
        are not allowed. The leaves of the tree are the values of the tree and
        the branches are the containers that hold the leaves.
    """

    def __init_subclass__(klass: type[T], **k):
        if "__setattr__" in vars(klass):
            raise TypeError(f"Reserved methods: `__setattr__` defined in `{klass}`.")
        if "__delattr__" in vars(klass):
            raise TypeError(f"Reserved methods: `__delattr__` defined in `{klass}`.")

        super().__init_subclass__(**k)

        def tree_unflatten(keys: tuple[str, ...], leaves: tuple[Any, ...]) -> T:
            # unflatten rule to use with `jax.tree_unflatten`
            vars(tree := getattr(object, "__new__")(klass)).update(zip(keys, leaves))
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
    def at(self) -> TreeClassIndexer:
        """Immutable out-of-place indexing.

        - ``.at[***].get()``:
            Return a new instance with the value at the index otherwise None.
        - ``.at[***].set(value)``:
            Set the `value` and return a new instance with the updated value.
        - ``.at[***].apply(func)``:
            Apply a ``func`` and return a new instance with the updated value.
        - ``.at['method'](*a, **k)``:
            Call a ``method`` and return a (return value, new instance) tuple.

        `***` acceptable indexing types are:
            - ``str`` for mapping keys or class attributes.
            - ``int`` for positional indexing for sequences.
            - ``...`` to select all leaves.
            - a boolean mask of the same structure as the tree
            - ``re.Pattern`` to index all keys matching a regex pattern.
            - an instance of ``BaseKey`` with custom logic to index a pytree.
            - a tuple of the above types to index multiple keys at same level.

        Example:
            >>> import pytreeclass as pytc
            >>> @pytc.autoinit
            ... class Tree(pytc.TreeClass):
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

        Note:
            - ``pytree.at[*].at[**]`` is equivalent to selecting pytree.*.**
            - ``pytree.at[*, **]`` is equivalent selecting pytree.* and pytree.**
        """
        return TreeClassIndexer(self)

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


@pp_dispatcher.register(TreeClass)
def treeclass_pp(node: TreeClass, **spec: Unpack[PPSpec]) -> str:
    name = type(node).__name__
    skip = [f.name for f in fields(node) if not f.repr]
    kvs = tuple((k, v) for k, v in vars(node).items() if k not in skip)
    return name + "(" + pps(kvs, pp=attr_value_pp, **spec) + ")"
