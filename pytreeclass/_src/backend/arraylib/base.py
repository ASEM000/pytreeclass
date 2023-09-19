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


class AbstractArray(abc.ABC):
    """The minimal array operations used by pytreeclass."""

    @staticmethod
    @abc.abstractmethod
    def tobytes(array):
        ...

    @property
    @abc.abstractmethod
    def ndarray(self):
        ...

    @staticmethod
    @abc.abstractmethod
    def where(condition, x, y):
        ...

    @staticmethod
    @abc.abstractmethod
    def nbytes(array):
        ...

    @staticmethod
    @abc.abstractmethod
    def size(array):
        ...

    @staticmethod
    @abc.abstractmethod
    def ndim(array):
        ...

    @staticmethod
    @abc.abstractmethod
    def shape(array):
        ...

    @staticmethod
    @abc.abstractmethod
    def dtype(array):
        ...

    @staticmethod
    @abc.abstractmethod
    def min(array):
        ...

    @staticmethod
    @abc.abstractmethod
    def max(array):
        ...

    @staticmethod
    @abc.abstractmethod
    def mean(array):
        ...

    @staticmethod
    @abc.abstractmethod
    def std(array):
        ...

    @staticmethod
    @abc.abstractmethod
    def all(array):
        ...

    @staticmethod
    @abc.abstractmethod
    def is_floating(array):
        ...

    @staticmethod
    @abc.abstractmethod
    def is_integer(array):
        ...

    @staticmethod
    @abc.abstractmethod
    def is_inexact(array):
        ...

    @staticmethod
    @abc.abstractmethod
    def is_bool(array):
        ...
