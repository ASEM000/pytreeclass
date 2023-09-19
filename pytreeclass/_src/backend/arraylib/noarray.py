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

from pytreeclass._src.backend.arraylib.base import AbstractArray


class GenericArray(AbstractArray):
    ...


class NoArray(AbstractArray):
    @staticmethod
    def tobytes(array: GenericArray) -> bytes:
        raise NotImplementedError

    @property
    def ndarray(self) -> GenericArray:
        return GenericArray

    @staticmethod
    def where(condition, x, y):
        raise NotImplementedError

    @staticmethod
    def nbytes(array: GenericArray):
        raise NotImplementedError

    @staticmethod
    def size(array: GenericArray):
        raise NotImplementedError

    @staticmethod
    def ndim(array: GenericArray):
        raise NotImplementedError

    @staticmethod
    def shape(array: None):
        raise NotImplementedError

    @staticmethod
    def dtype(array: GenericArray):
        raise NotImplementedError

    @staticmethod
    def min(array: GenericArray):
        raise NotImplementedError

    @staticmethod
    def max(array: GenericArray):
        raise NotImplementedError

    @staticmethod
    def mean(array: GenericArray):
        raise NotImplementedError

    @staticmethod
    def std(array: GenericArray):
        raise NotImplementedError

    @staticmethod
    def all(array: GenericArray):
        raise NotImplementedError

    @staticmethod
    def is_floating(array: GenericArray):
        raise NotImplementedError

    @staticmethod
    def is_integer(array: GenericArray):
        raise NotImplementedError

    @staticmethod
    def is_inexact(array: GenericArray):
        raise NotImplementedError

    @staticmethod
    def is_bool(array: GenericArray):
        raise NotImplementedError
