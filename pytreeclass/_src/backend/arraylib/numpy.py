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

import numpy as np

from pytreeclass._src.backend.arraylib.base import AbstractArray


class NumpyArray(AbstractArray):
    @staticmethod
    def tobytes(array: np.ndarray) -> bytes:
        return np.array(array).tobytes()

    @property
    def ndarray(self) -> type[np.ndarray]:
        return np.ndarray

    @staticmethod
    def where(condition, x, y):
        return np.where(condition, x, y)

    @staticmethod
    def nbytes(array: np.ndarray) -> np.ndarray:
        return array.nbytes

    @staticmethod
    def size(array: np.ndarray) -> np.ndarray:
        return array.size

    @staticmethod
    def ndim(array: np.ndarray) -> np.ndarray:
        return array.ndim

    @staticmethod
    def shape(array: np.ndarray) -> np.ndarray:
        return array.shape

    @staticmethod
    def dtype(array: np.ndarray) -> np.ndarray:
        return array.dtype

    @staticmethod
    def min(array: np.ndarray) -> np.ndarray:
        return np.min(array)

    @staticmethod
    def max(array: np.ndarray) -> np.ndarray:
        return np.max(array)

    @staticmethod
    def mean(array: np.ndarray) -> np.ndarray:
        return np.mean(array)

    @staticmethod
    def std(array: np.ndarray) -> np.ndarray:
        return np.std(array)

    @staticmethod
    def all(array: np.ndarray) -> np.ndarray:
        return np.all(array)

    @staticmethod
    def is_floating(array: np.ndarray) -> bool:
        return np.issubdtype(array.dtype, np.floating)

    @staticmethod
    def is_integer(array: np.ndarray) -> bool:
        return np.issubdtype(array.dtype, np.integer)

    @staticmethod
    def is_inexact(array: np.ndarray) -> bool:
        return np.issubdtype(array.dtype, np.inexact)

    @staticmethod
    def is_bool(array: np.ndarray) -> bool:
        return np.issubdtype(array.dtype, np.bool_)
