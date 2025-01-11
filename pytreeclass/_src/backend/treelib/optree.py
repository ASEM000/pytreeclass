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

from __future__ import annotations

import dataclasses as dc
from typing import Any, Callable, Hashable, Iterable

import optree as ot

from pytreeclass._src.backend.treelib.base import (
    AbstractTreeLib,
    KeyPathLeaf,
    ParallelConfig,
    Tree,
    concurrent_map,
    namespace,
)


@dc.dataclass(frozen=True)
class SequenceKey:
    idx: int


@dc.dataclass(frozen=True)
class DictKey:
    key: Hashable


@dc.dataclass(frozen=True)
class GetAttrKey:
    name: str


class OpTreeTreeLib(AbstractTreeLib):
    @staticmethod
    def tree_map(
        func: Callable[..., Any],
        tree: Any,
        *rest: Any,
        is_leaf: Callable[[Any], bool] | None = None,
        is_parallel: bool | ParallelConfig = False,
    ) -> Any:
        leaves, treedef = ot.tree_flatten(tree, is_leaf, namespace=namespace)
        flat = [leaves] + [treedef.flatten_up_to(r) for r in rest]
        if not is_parallel:
            return ot.tree_unflatten(treedef, [func(*args) for args in zip(*flat)])
        config = dict() if is_parallel is True else is_parallel
        return ot.tree_unflatten(treedef, concurrent_map(func, flat, **config))

    @staticmethod
    def tree_path_map(
        func: Callable[..., Any],
        tree: Any,
        *rest: Any,
        is_leaf: Callable[[Any], bool] | None = None,
        is_parallel: bool | ParallelConfig = False,
    ) -> Any:
        leaves, treedef = ot.tree_flatten(tree, is_leaf, namespace=namespace)
        flat = [leaves] + [treedef.flatten_up_to(r) for r in rest]
        flat = (ot.treespec_paths(treedef), *flat)
        if not is_parallel:
            return ot.tree_unflatten(treedef, [func(*args) for args in zip(*flat)])
        config = dict() if is_parallel is True else is_parallel
        return ot.tree_unflatten(treedef, concurrent_map(func, flat, **config))

    @staticmethod
    def tree_flatten(
        tree: Any,
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> tuple[Iterable[Any], ot.PyTreeDef]:
        leaves, treedef = ot.tree_flatten(tree, is_leaf, namespace=namespace)
        return (leaves, treedef)

    @staticmethod
    def tree_path_flatten(
        tree: Any,
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> tuple[Iterable[KeyPathLeaf], ot.PyTreeDef]:
        leaves, treedef = ot.tree_flatten(tree, is_leaf, namespace=namespace)
        return (list(zip(ot.treespec_paths(treedef), leaves)), treedef)

    @staticmethod
    def tree_unflatten(treedef: ot.PyTreeDef, leaves: Iterable[Any]) -> Any:
        return ot.tree_unflatten(treedef, leaves)

    @staticmethod
    def register_treeclass(klass: type[Tree]) -> None:
        def unflatten(keys: tuple[str, ...], leaves: tuple[Any, ...]) -> Tree:
            vars(tree := getattr(object, "__new__")(klass)).update(zip(keys, leaves))
            return tree

        def flatten(tree: Tree):
            dynamic = dict(vars(tree))
            keys = tuple(dynamic.keys())
            entries = tuple(GetAttrKey(k) for _, k in enumerate(keys))
            return (tuple(dynamic.values()), keys, entries)

        ot.register_pytree_node(klass, flatten, unflatten, namespace=namespace)

    @staticmethod
    def register_static(klass: type[Tree]) -> None:
        ot.register_pytree_node(
            klass,
            lambda x: ((), x),
            lambda x, _: x,
            namespace=namespace,
        )

    @staticmethod
    def attribute_key(name: str) -> GetAttrKey:
        return GetAttrKey(name)

    @staticmethod
    def sequence_key(index: int) -> SequenceKey:
        return SequenceKey(index)

    @staticmethod
    def dict_key(key: Hashable) -> DictKey:
        return DictKey(key)
