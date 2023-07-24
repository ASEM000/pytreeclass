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
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import optax
import pytest

import pytreeclass as pytc


def test_nn():
    class Linear(pytc.TreeClass):
        def __init__(self, key, in_dim, out_dim):
            self.weight = jr.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(2 / in_dim)
            self.bias = jnp.ones((1, out_dim))

        def __call__(self, x):
            return x @ self.weight + self.bias

    class StackedLinear(pytc.TreeClass):
        def __init__(self, key, layers):
            keys = jr.split(key, len(layers) - 1)

            self.layers = ()

            for ki, in_dim, out_dim in zip(keys, layers[:-1], layers[1:]):
                self.layers += (Linear(ki, in_dim, out_dim),)

        def __call__(self, x):
            *layers, last = self.layers
            for layer in layers:
                x = layer(x)
                x = jax.nn.tanh(x)
            return last(x)

    x = jnp.linspace(0, 1, 100)[:, None]
    y = x**3 + jr.uniform(jr.PRNGKey(0), (100, 1)) * 0.01

    nn = StackedLinear(layers=[1, 128, 128, 1], key=jr.PRNGKey(0))

    def loss_func(nn, x, y):
        return jnp.mean((nn(x) - y) ** 2)

    @jax.jit
    def update(nn, x, y):
        value, grads = jax.value_and_grad(loss_func)(nn, x, y)
        return value, jax.tree_map(lambda x, y: x - 1e-3 * y, nn, grads)

    for _ in range(1, 2001):
        value, nn = update(nn, x, y)

    np.testing.assert_allclose(value, jnp.array(0.00103019), atol=1e-5)


def test_nn_with_func_input():
    class Linear(pytc.TreeClass):
        def __init__(self, key, in_dim, out_dim, act_func):
            self.act_func = act_func
            self.weight = jr.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(2 / in_dim)
            self.bias = jnp.ones((1, out_dim))

        def __call__(self, x):
            return self.act_func(x @ self.weight + self.bias)

    layer = Linear(key=jr.PRNGKey(0), in_dim=1, out_dim=1, act_func=jax.nn.tanh)
    x = jnp.linspace(0, 1, 100)[:, None]
    # y = x**3 + jr.uniform(jr.PRNGKey(0), (100, 1)) * 0.01
    layer(x)
    return True


def test_compact_nn():
    class Linear(pytc.TreeClass):
        def __init__(self, key, in_dim, out_dim):
            self.weight = jr.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(2 / in_dim)
            self.bias = jnp.ones((1, out_dim))

        def __call__(self, x):
            return x @ self.weight + self.bias

    @pytc.leafwise
    class StackedLinear(pytc.TreeClass):
        def __init__(self, key, in_dim, out_dim, hidden_dim):
            keys = jr.split(key, 3)

            self.l1 = Linear(key=keys[0], in_dim=in_dim, out_dim=hidden_dim)
            self.l2 = Linear(key=keys[1], in_dim=hidden_dim, out_dim=hidden_dim)
            self.l3 = Linear(key=keys[2], in_dim=hidden_dim, out_dim=out_dim)

        def __call__(self, x):
            x = self.l1(x)
            x = jax.nn.tanh(x)
            x = self.l2(x)
            x = jax.nn.tanh(x)
            x = self.l3(x)

            return x

    x = jnp.linspace(0, 1, 100)[:, None]
    y = x**3 + jr.uniform(jr.PRNGKey(0), (100, 1)) * 0.01

    nn = StackedLinear(in_dim=1, out_dim=1, hidden_dim=10, key=jr.PRNGKey(0))

    def loss_func(nn, x, y):
        return jnp.mean((nn(x) - y) ** 2)

    @jax.jit
    def update(nn, x, y):
        value, grads = jax.value_and_grad(loss_func)(nn, x, y)

        # no need to use `jax.tree_map` to update the nn
        #  as it nn is wrapped by @ft.partial(pytc.treeclass, leafwise=True)
        return value, nn - 1e-3 * grads

    for _ in range(1, 10_001):
        value, nn = update(nn, x, y)

    np.testing.assert_allclose(value, jnp.array(0.0031012), atol=1e-5)


def test_freeze_nondiff():
    @pytc.leafwise
    class Linear(pytc.TreeClass):
        def __init__(self, key, in_dim, out_dim):
            self.weight = jr.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(2 / in_dim)
            self.bias = jnp.ones((1, out_dim))
            self.use_bias = True
            self.count = 0

        def __call__(self, x):
            return x @ self.weight + self.bias

    @pytc.leafwise
    class StackedLinear(pytc.TreeClass):
        def __init__(self, key, in_dim, out_dim, hidden_dim):
            self.name = "stack"
            keys = jr.split(key, 3)

            self.l1 = Linear(key=keys[0], in_dim=in_dim, out_dim=hidden_dim)
            self.l2 = Linear(key=keys[1], in_dim=hidden_dim, out_dim=hidden_dim)
            self.l3 = Linear(key=keys[2], in_dim=hidden_dim, out_dim=out_dim)

        def __call__(self, x):
            x = self.l1(x)
            x = jax.nn.tanh(x)
            x = self.l2(x)
            x = jax.nn.tanh(x)
            x = self.l3(x)

            return x

    x = jnp.linspace(0, 1, 100)[:, None]
    y = x**3 + jr.uniform(jr.PRNGKey(0), (100, 1)) * 0.01

    nn = StackedLinear(in_dim=1, out_dim=1, hidden_dim=10, key=jr.PRNGKey(0))

    def loss_func(nn, x, y):
        return jnp.mean((nn(x) - y) ** 2)

    with pytest.raises(TypeError):

        @jax.jit
        def update(nn, x, y):
            value, grads = jax.value_and_grad(loss_func)(nn, x, y)
            return value, nn - 1e-3 * grads

        for _ in range(1, 10_001):
            value, nn = update(nn, x, y)

    @jax.jit
    def update(nn, x, y):
        value, grads = jax.value_and_grad(loss_func)(nn, x, y)
        return value, nn - 1e-3 * grads

    mask = jtu.tree_map(pytc.is_nondiff, nn)
    freezeed_nn = nn.at[mask].apply(pytc.freeze)

    for _ in range(1, 10_001):
        value, freezeed_nn = update(freezeed_nn, x, y)

    np.testing.assert_allclose(value, jnp.array(0.0031012), atol=1e-5)

    nn = StackedLinear(in_dim=1, out_dim=1, hidden_dim=10, key=jr.PRNGKey(0))

    frozen_ = jtu.tree_map(pytc.freeze, nn)
    unfrozen_ = jtu.tree_map(pytc.unfreeze, frozen_, is_leaf=pytc.is_frozen)
    assert jtu.tree_leaves(nn) == jtu.tree_leaves(unfrozen_)

    class Linear(pytc.TreeClass):
        def __init__(self, key, in_dim, out_dim):
            self.weight = jr.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(2 / in_dim)
            self.bias = jnp.ones((1, out_dim))

        def __call__(self, x):
            return x @ self.weight + self.bias

    nn = Linear(jr.PRNGKey(0), 1, 1)
    nn_ = nn
    nn = nn.at["weight"].apply(pytc.freeze)

    optim = optax.sgd(1e-3)
    optim_state = optim.init(nn)

    x = jnp.linspace(0, 1, 100)[:, None]
    y = x**3 + jr.uniform(jr.PRNGKey(0), (100, 1)) * 0.01

    def loss_func(nn, x, y):
        nn = pytc.tree_unmask(nn)
        return jnp.mean((nn(x) - y) ** 2)

    @jax.jit
    def train_step(nn, optim_state, x, y):
        value, dnn = jax.value_and_grad(loss_func)(nn, x, y)
        dnn, optim_state = optim.update(dnn, optim_state)
        nn = optax.apply_updates(nn, dnn)
        return nn, optim_state, value

    for _ in range(1, 100):
        nn, optim_state, value = train_step(nn, optim_state, x, y)

    nn = pytc.tree_unmask(nn)
    assert nn_.weight == nn.weight, nn


@pytest.mark.benchmark
def test_nn_benchmark(benchmark):
    benchmark(test_nn)


@pytest.mark.benchmark
def test_freeze_nondiff_benchmark(benchmark):
    benchmark(test_freeze_nondiff)
