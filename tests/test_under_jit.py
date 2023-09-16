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


import pytest

from pytreeclass._src.backend import backend
from pytreeclass._src.backend import numpy as np
from pytreeclass._src.code_build import autoinit
from pytreeclass._src.tree_base import TreeClass
from pytreeclass._src.tree_util import is_tree_equal, leafwise


@pytest.mark.skipif(backend != "jax", reason="jax backend is not installed")
def test_ops_with_jit():
    import jax

    @autoinit
    @leafwise
    class T0(TreeClass):
        a: jax.Array = np.array(1)
        b: jax.Array = np.array(2)
        c: jax.Array = np.array(3)

    @autoinit
    @leafwise
    class T1(TreeClass):
        a: jax.Array = np.array(1)
        b: jax.Array = np.array(2)
        c: jax.Array = np.array(3)
        d: jax.Array = np.array([1, 2, 3])

    @jax.jit
    def getter(tree):
        return tree.at[...].get()

    @jax.jit
    def setter(tree):
        return tree.at[...].set(0)

    @jax.jit
    def applier(tree):
        return tree.at[...].apply(lambda _: 0)

    assert is_tree_equal(getter(T0()), T0())
    assert is_tree_equal(T0(0, 0, 0), setter(T0()))
    assert is_tree_equal(T0(0, 0, 0), applier(T0()))
    assert is_tree_equal(getter(T1()), T1())
    assert is_tree_equal(T1(0, 0, 0, 0), setter(T1()))
    assert is_tree_equal(T1(0, 0, 0, 0), applier(T1()))
    assert jax.jit(is_tree_equal)(T1(0, 0, 0, 0), applier(T1()))
