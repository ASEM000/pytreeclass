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
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from importlib.util import find_spec
from typing import (
    Any,
    Callable,
    Hashable,
    Iterable,
    Literal,
    Tuple,
    TypedDict,
    TypeVar,
    Union,
    get_args,
)

BackendsLiteral = Literal["jax", "numpy"]
backend: BackendsLiteral = os.environ.get("PYTREECLASS_BACKEND", "jax").lower()
namespace: str = os.environ.get("PYTREECLASS_NAMESPACE", "PYTREECLASS")
BACKENDS = get_args(BackendsLiteral)
TreeDef = TypeVar("TreeDef")
Leaf = TypeVar("Leaf", bound=Any)
KeyEntry = TypeVar("KeyEntry", bound=Hashable)
KeyPath = Tuple[KeyEntry, ...]
KeyPathLeaf = Tuple[KeyPath, Leaf]
IsLeaf = Union[Callable[[Any], bool], None]
FlattenOut = Tuple[Iterable[Leaf], TreeDef]
PathFlattenOut = Tuple[Iterable[KeyPathLeaf], TreeDef]
T = TypeVar("T")
pool_map = dict(thread=ThreadPoolExecutor, process=ProcessPoolExecutor)


if backend not in BACKENDS:
    raise ImportError(
        f"Invalid backend {backend=}. Must be one of {BACKENDS=}."
        f"Set the environment variable PYTREECLASS_BACKEND to one of {BACKENDS=}."
    )


class IsParallel(TypedDict):
    max_workers: int | None
    kind: Literal["thread", "process"]


def raise_future_execption(future):
    raise future.exception()


def concurrent_map(
    func: Callable[..., Any],
    flat: Iterable[Any],
    is_parallel: bool | IsParallel,
) -> Iterable[Any]:
    is_parallel = dict() if is_parallel is True else is_parallel
    workers = is_parallel.get("max_workers", None)
    kind = is_parallel.get("kind", "thread")

    with (executor := pool_map[kind](workers)) as executor:
        futures = [executor.submit(func, *args) for args in zip(*flat)]

    return [
        future.result()
        if future.exception() is None
        else raise_future_execption(future)
        for future in futures
    ]


class AbstractTreeUtil(abc.ABC):
    """The minimal interface for tree operations used by pytreeclass."""

    @staticmethod
    @abc.abstractmethod
    def tree_map(
        func: Callable[..., Any],
        tree: Any,
        *rest: Any,
        is_leaf: IsLeaf = None,
        is_parallel: bool | IsParallel = False,
    ) -> Any:
        ...

    @staticmethod
    @abc.abstractmethod
    def tree_path_map(
        func: Callable[..., Any],
        tree: Any,
        *rest: Any,
        is_leaf: IsLeaf = None,
        is_parallel: bool | IsParallel = False,
    ) -> Any:
        ...

    @staticmethod
    @abc.abstractmethod
    def tree_flatten(self, tree: Any, *, is_leaf: IsLeaf = None) -> FlattenOut:
        ...

    @staticmethod
    @abc.abstractmethod
    def tree_path_flatten(self, tree: Any, *, is_leaf: IsLeaf = None) -> PathFlattenOut:
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
    if find_spec("jax") is None:
        import logging

        logging.info("[PYTREECLASS]: switching to `numpy` backend.")
        backend = "numpy"

if backend == "numpy":
    if find_spec("numpy") is None:
        raise ImportError("`numpy` backend requires `numpy` to be installed.")
    if find_spec("optree") is None:
        raise ImportError("`numpy` backend requires `optree` to be installed.")


if backend == "jax":
    import jax.numpy as jnp
    import jax.tree_util as jtu

    @dc.dataclass(frozen=True)
    class NamedSequenceKey(jtu.GetAttrKey, jtu.SequenceKey):
        def __str__(self):
            return f".{self.name}"

    class JaxTreeUtil(AbstractTreeUtil):
        @staticmethod
        def tree_map(
            func: Callable[..., Any],
            tree: Any,
            *rest: Any,
            is_leaf: IsLeaf = None,
            is_parallel: bool | IsParallel = True,
        ) -> Any:
            leaves, treedef = jtu.tree_flatten(tree, is_leaf)
            flat = [leaves] + [treedef.flatten_up_to(r) for r in rest]
            if not is_parallel:
                return jtu.tree_unflatten(treedef, [func(*args) for args in zip(*flat)])
            return jtu.tree_unflatten(treedef, concurrent_map(func, flat, is_parallel))

        @staticmethod
        def tree_path_map(
            func: Callable[..., Any],
            tree: Any,
            *rest: Any,
            is_leaf: IsLeaf = None,
            is_parallel: bool | IsParallel = True,
        ) -> Any:
            leaves, treedef = jtu.tree_flatten_with_path(tree, is_leaf)
            flat = list(zip(*leaves)) + [treedef.flatten_up_to(r) for r in rest]
            if not is_parallel:
                return jtu.tree_unflatten(treedef, [func(*args) for args in zip(*flat)])
            return jtu.tree_unflatten(treedef, concurrent_map(func, flat, is_parallel))

        @staticmethod
        def tree_flatten(tree: Any, *, is_leaf: IsLeaf = None) -> FlattenOut:
            return jtu.tree_flatten(tree, is_leaf=is_leaf)

        @staticmethod
        def tree_path_flatten(tree: Any, *, is_leaf: IsLeaf = None) -> PathFlattenOut:
            return jtu.tree_flatten_with_path(tree, is_leaf=is_leaf)

        @staticmethod
        def tree_unflatten(treedef: TreeDef, leaves: Iterable[Any]) -> Any:
            return jtu.tree_unflatten(treedef, leaves)

        @staticmethod
        def register_treeclass(klass: type[T]) -> None:
            def unflatten(keys: tuple[str, ...], leaves: tuple[Any, ...]) -> T:
                tree = getattr(object, "__new__")(klass)
                vars(tree).update(zip(keys, leaves))
                return tree

            def flatten(tree: T) -> tuple[tuple[Any, ...], tuple[str, ...]]:
                dynamic = vars(tree)
                return tuple(dynamic.values()), tuple(dynamic.keys())

            def flatten_with_keys(tree: T):
                dynamic = dict(vars(tree))
                for idx, key in enumerate(vars(tree)):
                    dynamic[key] = (NamedSequenceKey(idx, key), dynamic[key])
                return tuple(dynamic.values()), tuple(dynamic.keys())

            jtu.register_pytree_with_keys(klass, flatten_with_keys, unflatten, flatten)

        @staticmethod
        def register_static(klass: type[T]) -> None:
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

    tree_util = JaxTreeUtil()
    numpy = jnp


elif backend == "numpy":
    import numpy as np
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

    class OpTreeTreeUtil(AbstractTreeUtil):
        @staticmethod
        def tree_map(
            func: Callable[..., Any],
            tree: Any,
            *rest: Any,
            is_leaf: IsLeaf = None,
            is_parallel: bool | IsParallel = False,
        ) -> Any:
            leaves, treedef = ot.tree_flatten(tree, is_leaf, namespace=namespace)
            flat = [leaves] + [treedef.flatten_up_to(r) for r in rest]
            if not is_parallel:
                return ot.tree_unflatten(treedef, [func(*args) for args in zip(*flat)])
            return ot.tree_unflatten(treedef, concurrent_map(func, flat, is_parallel))

        @staticmethod
        def tree_path_map(
            func: Callable[..., Any],
            tree: Any,
            *rest: Any,
            is_leaf: IsLeaf = None,
            is_parallel: bool | IsParallel = False,
        ) -> Any:
            leaves, treedef = ot.tree_flatten(tree, is_leaf, namespace=namespace)
            flat = [leaves] + [treedef.flatten_up_to(r) for r in rest]
            flat = (ot.treespec_paths(treedef), *flat)
            if not is_parallel:
                return ot.tree_unflatten(treedef, [func(*args) for args in zip(*flat)])
            return ot.tree_unflatten(treedef, concurrent_map(func, flat, is_parallel))

        @staticmethod
        def tree_flatten(tree: Any, *, is_leaf: IsLeaf = None) -> FlattenOut:
            leaves, treedef = ot.tree_flatten(tree, is_leaf, namespace=namespace)
            return (leaves, treedef)

        @staticmethod
        def tree_path_flatten(tree: Any, *, is_leaf: IsLeaf = None) -> PathFlattenOut:
            leaves, treedef = ot.tree_flatten(tree, is_leaf, namespace=namespace)
            return (list(zip(ot.treespec_paths(treedef), leaves)), treedef)

        @staticmethod
        def tree_unflatten(treedef: TreeDef, leaves: Iterable[Any]) -> Any:
            return ot.tree_unflatten(treedef, leaves)

        @staticmethod
        def register_treeclass(klass: type[T]) -> None:
            def unflatten(keys: tuple[str, ...], leaves: tuple[Any, ...]) -> T:
                tree = getattr(object, "__new__")(klass)
                vars(tree).update(zip(keys, leaves))
                return tree

            def flatten(tree: T):
                dynamic = dict(vars(tree))
                keys = tuple(dynamic.keys())
                entries = tuple(NamedSequenceKey(*ik) for ik in enumerate(keys))
                return (tuple(dynamic.values()), keys, entries)

            ot.register_pytree_node(klass, flatten, unflatten, namespace)

        @staticmethod
        def register_static(klass: type[T]) -> None:
            ot.register_pytree_node(klass, lambda x: ((), x), lambda x, _: x, namespace)

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
            return "".join(str(key) for key in keys)

    tree_util = OpTreeTreeUtil()
    numpy = np
