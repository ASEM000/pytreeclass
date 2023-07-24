# Copyright 2023 PyTreeClass authors
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


import jax
from jax import numpy as jnp

import pytreeclass as pytc


def test_ops_with_jit():
    @pytc.autoinit
    @pytc.leafwise
    class T0(pytc.TreeClass):
        a: jax.Array = jnp.array(1)
        b: jax.Array = jnp.array(2)
        c: jax.Array = jnp.array(3)

    @pytc.autoinit
    @pytc.leafwise
    class T1(pytc.TreeClass):
        a: jax.Array = jnp.array(1)
        b: jax.Array = jnp.array(2)
        c: jax.Array = jnp.array(3)
        d: jax.Array = jnp.array([1, 2, 3])

    @jax.jit
    def getter(tree):
        return tree.at[...].get()

    @jax.jit
    def setter(tree):
        return tree.at[...].set(0)

    @jax.jit
    def applier(tree):
        return tree.at[...].apply(lambda _: 0)

    # with pytest.raises(jax.errors.ConcretizationTypeError):
    pytc.is_tree_equal(getter(T0()), T0())

    assert pytc.is_tree_equal(T0(0, 0, 0), setter(T0()))

    assert pytc.is_tree_equal(T0(0, 0, 0), applier(T0()))

    # with pytest.raises(jax.errors.ConcretizationTypeError):
    pytc.is_tree_equal(getter(T1()), T1())

    assert pytc.is_tree_equal(T1(0, 0, 0, 0), setter(T1()))

    assert pytc.is_tree_equal(T1(0, 0, 0, 0), applier(T1()))

    assert jax.jit(pytc.is_tree_equal)(T1(0, 0, 0, 0), applier(T1()))
