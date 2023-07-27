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

"""Define a class that convert a class to a JAX compatible tree structure."""

from __future__ import annotations

import abc
from contextlib import contextmanager
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


# allow methods in mutable context to be called without raising `AttributeError`
# the instances are registered  during initialization and using `at`  property
# with `__call__` this is done by registering the instance id in a set before
# entering the mutable context and removing it after exiting the context
_mutable_instance_registry: set[int] = set()


@contextmanager
def _mutable_context(tree, *, kopy: bool = False):
    tree = tree_copy(tree) if kopy else tree
    _mutable_instance_registry.add(id(tree))
    yield tree
    _mutable_instance_registry.discard(id(tree))


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

        def recursive_getattr(tree: Any, where: tuple[str, ...]):
            if not isinstance(where[0], str):
                raise TypeError(f"Expected string, got {type(where[0])!r}.")
            if len(where) == 1:
                return getattr(tree, where[0])
            return recursive_getattr(getattr(tree, where[0]), where[1:])

        with _mutable_context(self.tree, kopy=True) as tree:
            value = recursive_getattr(tree, self.where)(*a, **k)  # type: ignore
        return value, tree


class TreeClassMeta(abc.ABCMeta):
    def __call__(klass: type[T], *a, **k) -> T:
        with _mutable_context(self := getattr(klass, "__new__")(klass, *a, **k)):
            getattr(klass, "__init__")(self, *a, **k)
        return self


class TreeClass(metaclass=TreeClassMeta):
    """Convert a class to a JAX compatible tree structure.

    Example:
        >>> import jax
        >>> import pytreeclass as pytc
        >>> # Tree leaves are instance attributes
        >>> @pytc.autoinit
        ... class Tree(pytc.TreeClass):
        ...     a:int = 1
        ...     b:float = 2.0
        >>> tree = Tree()
        >>> jax.tree_util.tree_leaves(tree)
        [1, 2.0]

        >>> # Leaf-wise math operations are supported by setting `leafwise=True`
        >>> @pytc.leafwise
        ... @pytc.autoinit
        ... class Tree(pytc.TreeClass):
        ...     a:int = 1
        ...     b:float = 2.0
        >>> tree = Tree()
        >>> tree + 1
        Tree(a=2, b=3.0)

        >>> # Advanced indexing is supported using `at` property
        >>> @pytc.autoinit
        ... class Tree(pytc.TreeClass):
        ...     a:int = 1
        ...     b:float = 2.0
        >>> tree = Tree()
        >>> tree.at["a"].get()
        Tree(a=1, b=None)
        >>> tree.at[0].get()
        Tree(a=1, b=None)
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
    kvs = ((k, v) for k, v in vars(node).items() if k not in skip)
    return name + "(" + pps(kvs, pp=attr_value_pp, **spec) + ")"
