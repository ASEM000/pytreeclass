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

import jax.numpy as jnp

from pytreeclass._src.backend.arraylib.base import AbstractArray


class JaxArray(AbstractArray):
    @staticmethod
    def tobytes(array: jnp.ndarray) -> bytes:
        return jnp.array(array).tobytes()

    @property
    def ndarray(self) -> type[jnp.ndarray]:
        return jnp.ndarray

    @staticmethod
    def where(condition, x, y) -> jnp.ndarray:
        return jnp.where(condition, x, y)

    @staticmethod
    def nbytes(array: jnp.ndarray) -> int:
        return array.nbytes

    @staticmethod
    def size(array: jnp.ndarray) -> int:
        return array.size

    @staticmethod
    def ndim(array: jnp.ndarray) -> int:
        return array.ndim

    @staticmethod
    def shape(array: jnp.ndarray) -> tuple[int, ...]:
        return array.shape

    @staticmethod
    def dtype(array: jnp.ndarray):
        return array.dtype

    @staticmethod
    def min(array: jnp.ndarray) -> jnp.ndarray:
        return jnp.min(array)

    @staticmethod
    def max(array: jnp.ndarray) -> jnp.ndarray:
        return jnp.max(array)

    @staticmethod
    def mean(array: jnp.ndarray) -> jnp.ndarray:
        return jnp.mean(array)

    @staticmethod
    def std(array: jnp.ndarray) -> jnp.ndarray:
        return jnp.std(array)

    @staticmethod
    def all(array: jnp.ndarray) -> jnp.ndarray:
        return jnp.all(array)

    @staticmethod
    def is_floating(array: jnp.ndarray) -> bool:
        return jnp.issubdtype(array.dtype, jnp.floating)

    @staticmethod
    def is_integer(array: jnp.ndarray) -> bool:
        return jnp.issubdtype(array.dtype, jnp.integer)

    @staticmethod
    def is_inexact(array: jnp.ndarray) -> bool:
        return jnp.issubdtype(array.dtype, jnp.inexact)

    @staticmethod
    def is_bool(array: jnp.ndarray) -> bool:
        return jnp.issubdtype(array.dtype, jnp.bool_)
