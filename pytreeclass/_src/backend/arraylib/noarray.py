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

from typing import Any

from pytreeclass._src.backend.arraylib.base import AbstractArray


class NoArray(AbstractArray):
    @staticmethod
    def tobytes(array: Any) -> bytes:
        raise NotImplementedError

    @property
    def ndarray(self) -> "NoArray":
        return type(self)

    @staticmethod
    def where(condition, x, y):
        raise NotImplementedError

    @staticmethod
    def nbytes(array: Any):
        raise NotImplementedError

    @staticmethod
    def size(array: Any):
        raise NotImplementedError

    @staticmethod
    def ndim(array: Any):
        raise NotImplementedError

    @staticmethod
    def shape(array: None):
        raise NotImplementedError

    @staticmethod
    def dtype(array: Any):
        raise NotImplementedError

    @staticmethod
    def min(array: Any):
        raise NotImplementedError

    @staticmethod
    def max(array: Any):
        raise NotImplementedError

    @staticmethod
    def mean(array: Any):
        raise NotImplementedError

    @staticmethod
    def std(array: Any):
        raise NotImplementedError

    @staticmethod
    def all(array: Any):
        raise NotImplementedError

    @staticmethod
    def is_floating(array: Any):
        raise NotImplementedError

    @staticmethod
    def is_integer(array: Any):
        raise NotImplementedError

    @staticmethod
    def is_inexact(array: Any):
        raise NotImplementedError

    @staticmethod
    def is_bool(array: Any):
        raise NotImplementedError
