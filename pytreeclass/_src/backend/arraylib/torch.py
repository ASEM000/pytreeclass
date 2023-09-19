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
import torch

from pytreeclass._src.backend.arraylib.base import AbstractArray

floatings = [torch.float16, torch.float32, torch.float64]
complexes = [torch.complex32, torch.complex64, torch.complex128]
integers = [torch.int8, torch.int16, torch.int32, torch.int64]


class TorchArray(AbstractArray):
    @staticmethod
    def tobytes(array: torch.Tensor) -> bytes:
        return np.from_dlpack(array).tobytes()

    @property
    def ndarray(self) -> type[torch.Tensor]:
        return torch.Tensor

    @staticmethod
    def where(condition, x, y):
        return torch.where(condition, x, y)

    @staticmethod
    def nbytes(array: torch.Tensor) -> int:
        return array.nbytes

    @staticmethod
    def size(array: torch.Tensor) -> int:
        return array.size().numel()

    @staticmethod
    def ndim(array: torch.Tensor) -> int:
        return array.ndim

    @staticmethod
    def shape(array: torch.Tensor) -> tuple[int, ...]:
        return tuple(array.shape)

    @staticmethod
    def dtype(array):
        return array.dtype

    @staticmethod
    def min(array: torch.Tensor) -> torch.Tensor:
        return torch.min(array)

    @staticmethod
    def max(array: torch.Tensor) -> torch.Tensor:
        return torch.max(array)

    @staticmethod
    def mean(array: torch.Tensor) -> torch.Tensor:
        return torch.mean(array)

    @staticmethod
    def std(array: torch.Tensor) -> torch.Tensor:
        return torch.std(array)

    @staticmethod
    def all(array: torch.Tensor) -> torch.Tensor:
        return torch.all(array)

    @staticmethod
    def is_floating(array: torch.Tensor) -> bool:
        return array.dtype in floatings

    @staticmethod
    def is_integer(array: torch.Tensor) -> bool:
        return array.dtype in integers

    @staticmethod
    def is_inexact(array):
        return array.dtype in floatings + complexes

    @staticmethod
    def is_bool(array):
        return array.dtype == torch.bool
