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
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from typing import Any, Callable, Hashable, Iterable, Literal, Tuple, TypedDict, TypeVar

namespace: str = os.environ.get("PYTREECLASS_NAMESPACE", "PYTREECLASS")

Tree = TypeVar("Tree", bound=Any)
Leaf = TypeVar("Leaf", bound=Any)
KeyEntry = TypeVar("KeyEntry", bound=Hashable)
KeyPath = Tuple[KeyEntry, ...]
KeyPathLeaf = Tuple[KeyPath, Leaf]
pool_map = dict(thread=ThreadPoolExecutor, process=ProcessPoolExecutor)


class ParallelConfig(TypedDict):
    max_workers: int | None
    kind: Literal["thread", "process"]


def raise_future_execption(future):
    raise future.exception()


def concurrent_map(
    func: Callable[..., Any],
    flat: Iterable[Any],
    max_workers: int | None = None,
    kind: Literal["thread", "process"] = "thread",
) -> Iterable[Any]:
    with (executor := pool_map[kind](max_workers)) as executor:
        futures = [executor.submit(func, *args) for args in zip(*flat)]

    return [
        future.result()
        if future.exception() is None
        else raise_future_execption(future)
        for future in futures
    ]


class AbstractTreeLib(abc.ABC):
    """The minimal interface for tree operations used by pytreeclass."""

    @staticmethod
    @abc.abstractmethod
    def tree_map(
        func: Callable[..., Any],
        tree: Any,
        *rest: Any,
        is_leaf: Callable[[Any], None] | None = None,
        is_parallel: bool | ParallelConfig = False,
    ) -> Any:
        ...

    @staticmethod
    @abc.abstractmethod
    def tree_path_map(
        func: Callable[..., Any],
        tree: Any,
        *rest: Any,
        is_leaf: Callable[[Any], bool] | None = None,
        is_parallel: bool | ParallelConfig = False,
    ) -> Any:
        ...

    @staticmethod
    @abc.abstractmethod
    def tree_flatten(
        tree: Any,
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> tuple[Iterable[Leaf], Any]:
        ...

    @staticmethod
    @abc.abstractmethod
    def tree_path_flatten(
        tree: Any,
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> tuple[Iterable[KeyPathLeaf], Any]:
        ...

    @staticmethod
    @abc.abstractmethod
    def tree_unflatten(treedef: Any, leaves: Iterable[Any]) -> Any:
        ...

    @staticmethod
    @abc.abstractmethod
    def register_treeclass(klass: type[Tree]) -> None:
        ...

    @staticmethod
    @abc.abstractmethod
    def register_static(klass: type[Tree]) -> None:
        ...

    @staticmethod
    @abc.abstractmethod
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
