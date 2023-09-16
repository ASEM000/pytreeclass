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
    def register_treeclass(klass: type[T]) -> None:
        ...

    @staticmethod
    @abc.abstractmethod
    def register_static(klass: type[T]) -> None:
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
        def register_treeclass(klass: type[T]) -> None:
            def tree_unflatten(keys: tuple[str, ...], leaves: tuple[Any, ...]) -> T:
                # unflatten rule to use with `tree_unflatten`
                tree = getattr(object, "__new__")(klass)
                vars(tree).update(zip(keys, leaves))
                return tree

            def tree_flatten(tree: T) -> tuple[tuple[Any, ...], tuple[str, ...]]:
                # flatten rule to use with `tree_flatten`
                dynamic = vars(tree)
                return tuple(dynamic.values()), tuple(dynamic.keys())

            def tree_flatten_with_keys(tree: T):
                # flatten rule to use with `tree_flatten_with_path`
                dynamic = dict(vars(tree))
                for idx, key in enumerate(vars(tree)):
                    dynamic[key] = (NamedSequenceKey(idx, key), dynamic[key])
                return tuple(dynamic.values()), tuple(dynamic.keys())

            jtu.register_pytree_with_keys(
                nodetype=klass,
                flatten_func=tree_flatten,
                flatten_with_keys=tree_flatten_with_keys,
                unflatten_func=tree_unflatten,
            )

        @staticmethod
        def register_static(klass: type[T]) -> None:
            jtu.register_pytree_node(
                nodetype=klass,
                flatten_func=lambda tree: ((), tree),
                unflatten_func=lambda treedef, _: treedef,
            )

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

    tree_util = TreeUtil()
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

    @dc.dataclass(frozen=True)
    class SequenceKey:
        idx: int

        def __str__(self):
            return f"[{repr(self.idx)}]"

    @dc.dataclass(frozen=True)
    class DictKey:
        key: Hashable

        def __str__(self):
            return f"[{repr(self.key)}]"

    @dc.dataclass(frozen=True)
    class GetAttrKey:
        name: str

        def __str__(self):
            return f".{self.name}"

    @dc.dataclass(frozen=True)
    class NamedSequenceKey(GetAttrKey, SequenceKey):
        def __str__(self) -> str:
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
            leaves, treedef = ot.tree_flatten(tree, is_leaf, namespace=namespace)
            paths = ot.treespec_paths(treedef)
            flat_args = [leaves] + [treedef.flatten_up_to(r) for r in rest]
            flat_results = map(func, paths, *flat_args)
            return treedef.unflatten(flat_results)

        @staticmethod
        def tree_flatten(
            tree: Any, *, is_leaf: Callable[[Any], bool] | None = None
        ) -> tuple[Iterable[Leaf], TreeDef]:
            return ot.tree_flatten(tree, is_leaf=is_leaf, namespace=namespace)

        @staticmethod
        def tree_flatten_with_path(
            tree: Any, *, is_leaf: Callable[[Any], bool] | None = None
        ) -> tuple[Iterable[KeyPathLeaf], TreeDef]:
            # optree returns a tuple of (leaves, paths, treedef) while jax returns
            # a tuple of (keys_leaves, treedef)
            leaves, treedef = ot.tree_flatten(tree, is_leaf, namespace=namespace)
            return list(zip(ot.treespec_paths(treedef), leaves)), treedef

        @staticmethod
        def tree_unflatten(treedef: TreeDef, leaves: Iterable[Any]) -> Any:
            return ot.tree_unflatten(treedef, leaves)

        @staticmethod
        def register_treeclass(klass: type[T]) -> None:
            def tree_unflatten(keys: tuple[str, ...], leaves: tuple[Any, ...]) -> T:
                # unflatten rule to use with `tree_unflatten`
                tree = getattr(object, "__new__")(klass)
                vars(tree).update(zip(keys, leaves))
                return tree

            def tree_flatten(tree: T):
                dynamic = dict(vars(tree))
                keys = tuple(dynamic.keys())
                entries = tuple(NamedSequenceKey(*ik) for ik in enumerate(keys))
                return (tuple(dynamic.values()), keys, entries)

            ot.register_pytree_node(
                klass,
                flatten_func=tree_flatten,
                unflatten_func=tree_unflatten,
                namespace=namespace,
            )

        @staticmethod
        def register_static(klass: type[T]) -> None:
            ot.register_pytree_node(
                klass,
                flatten_func=lambda tree: ((), tree),
                unflatten_func=lambda treedef, _: treedef,
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

        @staticmethod
        def keystr(keys: Any) -> str:
            return ".".join(str(key) for key in keys)

    tree_util = TreeUtil()
    numpy = numpy

else:
    raise ImportError(f"None of the {BACKENDS=} are installed.")
