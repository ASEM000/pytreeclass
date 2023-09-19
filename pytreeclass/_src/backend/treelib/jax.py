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

import jax.tree_util as jtu

from pytreeclass._src.backend.treelib.base import (
    AbstractTreeLib,
    KeyPathLeaf,
    ParallelConfig,
    Tree,
    concurrent_map,
)


@dc.dataclass(frozen=True)
class NamedSequenceKey(jtu.GetAttrKey, jtu.SequenceKey):
    def __str__(self):
        return f".{self.name}"


class JaxTreeLib(AbstractTreeLib):
    @staticmethod
    def tree_map(
        func: Callable[..., Any],
        tree: Any,
        *rest: Any,
        is_leaf: Callable[[Any], bool] | None = None,
        is_parallel: bool | ParallelConfig = False,
    ) -> Any:
        leaves, treedef = jtu.tree_flatten(tree, is_leaf)
        flat: list[Any] = [leaves] + [treedef.flatten_up_to(r) for r in rest]
        if is_parallel is False:
            return jtu.tree_unflatten(treedef, [func(*args) for args in zip(*flat)])
        config = dict() if is_parallel is True else is_parallel
        return jtu.tree_unflatten(treedef, concurrent_map(func, flat, **config))

    @staticmethod
    def tree_path_map(
        func: Callable[..., Any],
        tree: Any,
        *rest: Any,
        is_leaf: Callable[[Any], bool] | None = None,
        is_parallel: bool | ParallelConfig = False,
    ) -> Any:
        leaves, treedef = jtu.tree_flatten_with_path(tree, is_leaf)
        flat: list[Any] = list(zip(*leaves)) + [treedef.flatten_up_to(r) for r in rest]
        if not is_parallel:
            return jtu.tree_unflatten(treedef, [func(*args) for args in zip(*flat)])
        config = dict() if is_parallel is True else is_parallel
        return jtu.tree_unflatten(treedef, concurrent_map(func, flat, **config))

    @staticmethod
    def tree_flatten(
        tree: Any,
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> tuple[Iterable[Any], jtu.PyTreeDef]:
        return jtu.tree_flatten(tree, is_leaf=is_leaf)

    @staticmethod
    def tree_path_flatten(
        tree: Any,
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> tuple[Iterable[KeyPathLeaf], jtu.PyTreeDef]:
        return jtu.tree_flatten_with_path(tree, is_leaf=is_leaf)

    @staticmethod
    def tree_unflatten(treedef: jtu.PyTreeDef, leaves: Iterable[Any]) -> Any:
        return jtu.tree_unflatten(treedef, leaves)

    @staticmethod
    def register_treeclass(klass: type[Tree]) -> None:
        def unflatten(keys: tuple[str, ...], leaves: tuple[Any, ...]) -> Tree:
            vars(tree := getattr(object, "__new__")(klass)).update(zip(keys, leaves))
            return tree

        def flatten(tree: Tree) -> tuple[tuple[Any, ...], tuple[str, ...]]:
            return tuple((dynamic := vars(tree)).values()), tuple(dynamic.keys())

        def flatten_with_keys(tree: Tree):
            dynamic = dict(vars(tree))
            for idx, key in enumerate(vars(tree)):
                dynamic[key] = (NamedSequenceKey(idx, key), dynamic[key])
            return tuple(dynamic.values()), tuple(dynamic.keys())

        jtu.register_pytree_with_keys(klass, flatten_with_keys, unflatten, flatten)

    @staticmethod
    def register_static(klass: type[Tree]) -> None:
        jtu.register_pytree_node(klass, lambda x: ((), x), lambda x, _: x)

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
    def keystr(keys: Any) -> str:
        return jtu.keystr(keys)
