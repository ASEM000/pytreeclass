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

"""Backend tools for pytreeclass."""

from __future__ import annotations

import abc
import dataclasses as dc
import importlib
import os
from typing import Any, Callable, Hashable, Iterable, Literal, Tuple, TypeVar, get_args

BackendsLiteral = Literal["jax", "numpy"]
backend: BackendsLiteral = os.environ.get("PYTREECLASS_BACKEND", "jax").lower()
namespace: str = os.environ.get("PYTREECLASS_NAMESPACE", "PYTREECLASS")
BACKENDS = get_args(BackendsLiteral)

TreeDef = TypeVar("TreeDef")
Leaf = TypeVar("Leaf", bound=Any)
KeyEntry = TypeVar("KeyEntry", bound=Hashable)
KeyPath = Tuple[KeyEntry, ...]
KeyPathLeaf = Tuple[KeyPath, Leaf]
T = TypeVar("T")


class BackendTreeUtil(abc.ABC):
    """The minimal interface for tree operations used by pytreeclass."""

    @staticmethod
    @abc.abstractmethod
    def tree_map(
        func: Callable[..., Any],
        tree: Any,
        *rest: Any,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> Any:
        ...

    @staticmethod
    @abc.abstractmethod
    def tree_map_with_path(
        func: Callable[..., Any],
        tree: Any,
        *rest: Any,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> Any:
        ...

    @staticmethod
    @abc.abstractmethod
    def tree_flatten(
        self,
        tree: Any,
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> tuple[Iterable[Leaf], TreeDef]:
        ...

    @staticmethod
    @abc.abstractmethod
    def tree_flatten_with_path(
        tree: Any,
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> tuple[Iterable[KeyPathLeaf], TreeDef]:
        ...

    @staticmethod
    @abc.abstractmethod
    def tree_unflatten(self, treedef: TreeDef, leaves: Iterable[Any]) -> Any:
        ...

    @staticmethod
    @abc.abstractmethod
    def register_pytree_node(
        nodetype: type[T],
        flatten_func: Callable[[T], tuple[Iterable[Leaf], TreeDef]],
        unflatten_func: Callable[[TreeDef, Iterable[Leaf]], T],
    ) -> None:
        ...

    @staticmethod
    @abc.abstractmethod
    def register_pytree_node_with_path(
        nodetype: type[T],
        flatten_func_with_path: Callable[[T], tuple[Iterable[KeyPathLeaf], TreeDef]],
        unflatten_func: Callable[[TreeDef, Iterable[KeyPathLeaf]], T],
        flatten_func: Callable[[T], tuple[Iterable[Leaf], TreeDef]],
    ) -> None:
        ...

    @staticmethod
    @abc.abstractclassmethod
    def attribute_key(name: str) -> Any:
        ...

    @staticmethod
    @abc.abstractmethod
    def sequence_key(index: int) -> Any:
        ...

    @staticmethod
    @abc.abstractmethod
    def dict_key(key: Hashable) -> Any:
        ...

    @staticmethod
    @abc.abstractmethod
    def named_sequence_key(name: str, index: int) -> Any:
        ...

    @staticmethod
    @abc.abstractmethod
    def keystr(keys: Any) -> str:
        ...


if backend == "jax":
    if importlib.util.find_spec("jax") is None:
        raise ImportError("`jax` backend requires `jax` to be installed.")

    # tree utils and array utils
    import jax.numpy as jnp
    import jax.tree_util as jtu

    @dc.dataclass(frozen=True)
    class NamedSequenceKey(jtu.GetAttrKey, jtu.SequenceKey):
        def __str__(self):
            return f".{self.name}"

    class TreeUtil(BackendTreeUtil):
        @staticmethod
        def tree_map(
            func: Callable[..., Any],
            tree: Any,
            *rest: Any,
            is_leaf: Callable[[Any], bool] | None = None,
        ) -> Any:
            return jtu.tree_map(func, tree, *rest, is_leaf=is_leaf)

        @staticmethod
        def tree_map_with_path(
            func: Callable[..., Any],
            tree: Any,
            *rest: Any,
            is_leaf: Callable[[Any], bool] | None = None,
        ) -> Any:
            return jtu.tree_map_with_path(func, tree, *rest, is_leaf=is_leaf)

        @staticmethod
        def tree_flatten(
            tree: Any,
            *,
            is_leaf: Callable[[Any], bool] | None = None,
        ) -> tuple[Iterable[Leaf], TreeDef]:
            return jtu.tree_flatten(tree, is_leaf=is_leaf)

        @staticmethod
        def tree_flatten_with_path(
            tree: Any,
            *,
            is_leaf: Callable[[Any], bool] | None = None,
        ) -> tuple[Iterable[KeyPathLeaf], TreeDef]:
            return jtu.tree_flatten_with_path(tree, is_leaf=is_leaf)

        @staticmethod
        def tree_unflatten(treedef: TreeDef, leaves: Iterable[Any]) -> Any:
            return jtu.tree_unflatten(treedef, leaves)

        @staticmethod
        def register_pytree_node(
            nodetype: type[T],
            flatten_func: Callable[[T], tuple[Iterable[Leaf], TreeDef]],
            unflatten_func: Callable[[TreeDef, Iterable[Leaf]], T],
        ) -> None:
            jtu.register_pytree_node(nodetype, flatten_func, unflatten_func)

        @staticmethod
        def register_pytree_node_with_path(
            nodetype: type[T],
            flatten_func_with_path: Callable[
                [T], tuple[Iterable[KeyPathLeaf], TreeDef]
            ],
            unflatten_func: Callable[[TreeDef, Iterable[KeyPathLeaf]], T],
            flatten_func: Callable[[T], tuple[Iterable[Leaf], TreeDef]],
        ) -> None:
            args = (nodetype, flatten_func_with_path, unflatten_func, flatten_func)
            jtu.register_pytree_with_keys(*args)

        @staticmethod
        def attribute_key(name: str) -> jtu.GetAttrKey:
            return jtu.GetAttrKey(name)

        @staticmethod
        def sequence_key(index: int) -> jtu.SequenceKey:
            return jtu.SequenceKey(index)

        @staticmethod
        def dict_key(key: Hashable) -> jtu.DictKey:
            return jtu.DictKey(key)

        @staticmethod
        def named_sequence_key(name: str, index: int) -> NamedSequenceKey:
            return NamedSequenceKey(name, index)

        @staticmethod
        def keystr(keys: Any) -> str:
            return jtu.keystr(keys)

    numpy = jnp

elif backend == "numpy":
    # numpy backend for array utils
    # and optree backend for tree utils
    if importlib.util.find_spec("numpy") is None:
        raise ImportError("`numpy` backend requires `numpy` to be installed.")
    if importlib.util.find_spec("optree") is None:
        raise ImportError("`numpy` backend requires `optree` to be installed.")

    import numpy
    import optree as ot

    numpy = numpy

    @dc.dataclass(frozen=True)
    class NamedSequenceKey:
        name: str
        idx: int

        def pprint(self):
            return f".{self.name}"

    class TreeUtil(BackendTreeUtil):
        @staticmethod
        def tree_map(
            func: Callable[..., Any],
            tree: Any,
            *rest: Any,
            is_leaf: Callable[[Any], bool] | None = None,
        ) -> Any:
            return ot.tree_map(func, tree, *rest, is_leaf=is_leaf, namespace=namespace)

        @staticmethod
        def tree_map_with_path(
            func: Callable[..., Any],
            tree: Any,
            *rest: Any,
            is_leaf: Callable[[Any], bool] | None = None,
        ) -> Any:
            return ot.tree_map_with_path(
                func,
                tree,
                *rest,
                is_leaf=is_leaf,
                namespace=namespace,
            )

        @staticmethod
        def tree_flatten(
            tree: Any,
            *,
            is_leaf: Callable[[Any], bool] | None = None,
        ) -> tuple[Iterable[Leaf], TreeDef]:
            return ot.tree_flatten(tree, is_leaf=is_leaf, namespace=namespace)

        @staticmethod
        def tree_flatten_with_path(
            tree: Any, *, is_leaf: Callable[[Any], bool] | None = None
        ) -> tuple[Iterable[KeyPathLeaf], TreeDef]:
            # optree returns a tuple of (leaves, paths, treedef) while jax returns
            # a tuple of (keys_leaves, treedef)
            out = ot.tree_flatten_with_path(tree, is_leaf=is_leaf, namespace=namespace)
            paths, leaves, treedef = out
            return list(zip(paths, leaves)), treedef

        @staticmethod
        def tree_unflatten(treedef: TreeDef, leaves: Iterable[Any]) -> Any:
            return ot.tree_unflatten(treedef, leaves)

        @staticmethod
        def register_pytree_node(
            nodetype: type[T],
            flatten_func: Callable[[T], tuple[Iterable[Leaf], TreeDef]],
            unflatten_func: Callable[[TreeDef, Iterable[Leaf]], T],
        ) -> None:
            ot.register_pytree_node(nodetype, flatten_func, unflatten_func, namespace)

        @staticmethod
        def register_pytree_node_with_path(
            nodetype: type[T],
            flatten_func_with_path: Callable[
                [T], tuple[Iterable[KeyPathLeaf], TreeDef]
            ],
            unflatten_func: Callable[[TreeDef, Iterable[KeyPathLeaf]], T],
            flatten_func: Callable[[T], tuple[Iterable[Leaf], TreeDef]],
        ) -> None:
            def keypath_func(tree):
                keys_leaves, treedef = flatten_func_with_path(tree)
                return [k for k, _ in keys_leaves]

            ot.register_keypaths(nodetype, keypath_func)
            ot.register_pytree_node(nodetype, flatten_func, unflatten_func, namespace)

        @staticmethod
        def attribute_key(name: str) -> ot.GetitemKeyPathEntry:
            return ot.GetitemKeyPathEntry(name)

        @staticmethod
        def sequence_key(index: int) -> ot.GetitemKeyPathEntry:
            return ot.GetitemKeyPathEntry(index)

        @staticmethod
        def dict_key(key: Hashable) -> ot.GetitemKeyPathEntry:
            return ot.GetitemKeyPathEntry(key)

        @staticmethod
        def named_sequence_key(name: str, index: int) -> NamedSequenceKey:
            return NamedSequenceKey(name, index)

        @staticmethod
        def keystr(keys: Any) -> str:
            return "".join([k.pprint() for k in keys])

else:
    raise ImportError(f"None of the {BACKENDS=} are installed.")
