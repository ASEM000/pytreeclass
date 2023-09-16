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

import math
from typing import Any

import pytest

from pytreeclass._src.backend import numpy as np
from pytreeclass._src.code_build import autoinit, field
from pytreeclass._src.tree_base import TreeClass
from pytreeclass._src.tree_mask import freeze
from pytreeclass._src.tree_util import bcmap, is_tree_equal, leafwise


@leafwise
@autoinit
class Tree(TreeClass):
    """A simple tree class for math testing."""

    a: float
    b: float
    c: float
    name: str = field(on_setattr=[freeze])  # exclude it from math operations


tree1 = Tree(-10, 20, 30, "A")
tree2 = Tree(np.array([-10]), np.array([20]), np.array([30]), "A")


@pytest.mark.parametrize(
    ["tree", "expected"],
    [
        [abs(tree1), Tree(10, 20, 30, "A")],
        [tree1 + tree1, Tree(-20, 40, 60, "A")],
        [tree1 + 1, Tree(-9, 21, 31, "A")],
        [math.ceil(tree1), Tree(-10, 20, 30, "A")],
        [
            divmod(tree1, tree1),
            Tree(divmod(-10, -10), divmod(20, 20), divmod(30, 30), "A"),
        ],
        [tree1, tree1],
        [math.floor(tree1), Tree(-10, 20, 30, "A")],
        [tree1 // tree1, Tree(-10 // (-10), 20 // (20), 30 // (30), "A")],
        [tree1 // 2, Tree(-5, 10, 15, "A")],
        [tree1 >= tree1, Tree(-10 >= -10, 20 >= 20, 30 >= 30, "A")],
        [tree1 >= 1, Tree(-10 >= 1, 20 >= 1, 30 >= 1, "A")],
        [tree1 > tree1, Tree(-10 > -10, 20 > 20, 30 > 30, "A")],
        [tree1 > 1, Tree(-10 > 1, 20 > 1, 30 > 1, "A")],
        [~tree1, Tree(~-10, ~20, ~30, "A")],
        [tree1 <= tree1, Tree(-10 <= -10, 20 <= 20, 30 <= 30, "A")],
        [tree1 <= 1, Tree(-10 <= 1, 20 <= 1, 30 <= 1, "A")],
        [tree1 << 1, Tree(-10 << 1, 20 << 1, 30 << 1, "A")],
        [tree1 < tree1, Tree(-10 < -10, 20 < 20, 30 < 30, "A")],
        [tree1 != tree1, Tree(-10 != -10, 20 != 20, 30 != 30, "A")],
        [tree1 != 1, Tree(-10 != 1, 20 != 1, 30 != 1, "A")],
        [tree1 or tree1, Tree(-10 or -10, 20 or 20, 30 or 30, "A")],
        [tree1 or 1, Tree(-10 or 1, 20 or 1, 30 or 1, "A")],
        [+tree1, Tree(+(-10), +20, +30, "A")],
        [tree1**2, Tree(100, 400, 900, "A")],
        [1 + tree1, Tree(-9, 21, 31, "A")],
        [1 and tree1, Tree(1 and -10, 1 and 20, 1 and 30, "A")],
        [divmod(1, tree1), Tree(divmod(1, -10), divmod(1, 20), divmod(1, 30), "A")],
        [1 | tree1, Tree(1 | -10, 1 | 20, 1 | 30, "A")],
        [2**tree1, Tree(2**-10, 2**20, 2**30, "A")],
        [1 ^ tree1, Tree(1 ^ -10, 1 ^ 20, 1 ^ 30, "A")],
        [1 - tree1, Tree(1 + 10, 1 - 20, 1 - 30, "A")],
        [1 / tree1, Tree(1 / -10, 1 / 20, 1 / 30, "A")],
        [1 * tree1, Tree(1 * -10, 1 * 20, 1 * 30, "A")],
        [math.trunc(tree1), Tree(-10, 20, 30, "A")],
        [round(tree1), Tree(-10, 20, 30, "A")],
        [tree1 ^ tree1, Tree(-10 ^ -10, 20 ^ 20, 30 ^ 30, "A")],
        [
            tree2 @ tree2,
            Tree(
                np.array([10]) @ np.array([10]),
                np.array([20]) @ np.array([20]),
                np.array([30]) @ np.array([30]),
                "A",
            ),
        ],
    ],
)
def test_tree_math(tree, expected):
    assert is_tree_equal(tree, expected)
    assert tree == expected  # same as above


@leafwise
@autoinit
class Tree(TreeClass):
    """A simple tree class for math testing."""

    a: Any
    b: Any
    c: Any
    d: Any


tree = Tree(a=(10, 20, 30), b=(40, 50, 60), c=(70, 80, 90), d=(100, 110, 120))


@pytest.mark.parametrize(
    ["tree", "expected"],
    [
        [
            bcmap(np.where)(tree > 10, 0, tree),
            bcmap(np.array)(Tree(a=(10, 0, 0), b=(0, 0, 0), c=(0, 0, 0), d=(0, 0, 0))),
        ],
        # [
        #     bcmap(np.where)(tree > 10, 0, y=tree),
        #     bcmap(np.array)(Tree(a=(10, 0, 0), b=(0, 0, 0), c=(0, 0, 0), d=(0, 0, 0))),
        # ],
        # [
        #     bcmap(np.where)(tree > 10, x=0, y=tree),
        #     bcmap(np.array)(Tree(a=(10, 0, 0), b=(0, 0, 0), c=(0, 0, 0), d=(0, 0, 0))),
        # ],
        # [
        #     bcmap(np.where)(condition=tree > 10, x=0, y=tree),
        #     bcmap(np.array)(Tree(a=(10, 0, 0), b=(0, 0, 0), c=(0, 0, 0), d=(0, 0, 0))),
        # ],
    ],
)
def test_bcmap(tree, expected):
    assert is_tree_equal(tree, expected)


def test_math_operations_errors():
    with pytest.raises(TypeError):
        tree1 + "s"
