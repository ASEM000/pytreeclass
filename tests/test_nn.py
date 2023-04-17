from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import pytest

import pytreeclass as pytc


def test_nn():
    class Linear(pytc.TreeClass, leafwise=True):
        weight: jax.Array
        bias: jax.Array

        def __init__(self, key, in_dim, out_dim):
            self.weight = jr.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(2 / in_dim)
            self.bias = jnp.ones((1, out_dim))

        def __call__(self, x):
            return x @ self.weight + self.bias

    class StackedLinear(pytc.TreeClass, leafwise=True):
        layers: Sequence[Linear]

        def __init__(self, key, layers):
            keys = jr.split(key, len(layers) - 1)

            self.layers = []

            for ki, in_dim, out_dim in zip(keys, layers[:-1], layers[1:]):
                self.layers += [Linear(ki, in_dim, out_dim)]

        def __call__(self, x):
            for layer in self.layers[:-1]:
                x = layer(x)
                x = jax.nn.tanh(x)

            return self.layers[-1](x)

    x = jnp.linspace(0, 1, 100)[:, None]
    y = x**3 + jr.uniform(jr.PRNGKey(0), (100, 1)) * 0.01

    NN = StackedLinear(layers=[1, 128, 128, 1], key=jr.PRNGKey(0))

    def loss_func(NN, x, y):
        return jnp.mean((NN(x) - y) ** 2)

    @jax.jit
    def update(NN, x, y):
        value, grads = jax.value_and_grad(loss_func)(NN, x, y)
        return value, jax.tree_map(lambda x, y: x - 1e-3 * y, NN, grads)

    for _ in range(1, 2001):
        value, NN = update(NN, x, y)

    np.testing.assert_allclose(value, jnp.array(0.00103019), atol=1e-5)


def test_nn_with_func_input():
    class Linear(pytc.TreeClass, leafwise=True):
        weight: jax.Array
        bias: jax.Array
        act_func: Callable

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
    class Linear(pytc.TreeClass, leafwise=True):
        weight: jax.Array
        bias: jax.Array

        def __init__(self, key, in_dim, out_dim):
            self.weight = jr.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(2 / in_dim)
            self.bias = jnp.ones((1, out_dim))

        def __call__(self, x):
            return x @ self.weight + self.bias

    class StackedLinear(pytc.TreeClass, leafwise=True):
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

    NN = StackedLinear(in_dim=1, out_dim=1, hidden_dim=10, key=jr.PRNGKey(0))

    def loss_func(NN, x, y):
        return jnp.mean((NN(x) - y) ** 2)

    @jax.jit
    def update(NN, x, y):
        value, grads = jax.value_and_grad(loss_func)(NN, x, y)

        # no need to use `jax.tree_map` to update the NN
        #  as it NN is wrapped by @ft.partial(pytc.treeclass, leafwise=True)
        return value, NN - 1e-3 * grads

    for _ in range(1, 10_001):
        value, NN = update(NN, x, y)

    np.testing.assert_allclose(value, jnp.array(0.0031012), atol=1e-5)


def test_freeze_nondiff():
    class Linear(pytc.TreeClass, leafwise=True):
        weight: jax.Array
        bias: jax.Array
        count: int
        use_bias: bool

        def __init__(self, key, in_dim, out_dim):
            self.weight = jr.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(2 / in_dim)
            self.bias = jnp.ones((1, out_dim))
            self.use_bias = True
            self.count = 0

        def __call__(self, x):
            return x @ self.weight + self.bias

    class StackedLinear(pytc.TreeClass, leafwise=True):
        name: str
        exact_array: jax.Array
        bool_array: jax.Array

        def __init__(self, key, in_dim, out_dim, hidden_dim):
            self.name = "stack"
            self.exact_array = jnp.array([1, 2, 3])
            self.bool_array = jnp.array([True, True])

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

    NN = StackedLinear(in_dim=1, out_dim=1, hidden_dim=10, key=jr.PRNGKey(0))

    def loss_func(NN, x, y):
        return jnp.mean((NN(x) - y) ** 2)

    with pytest.raises(TypeError):

        @jax.jit
        def update(NN, x, y):
            value, grads = jax.value_and_grad(loss_func)(NN, x, y)
            return value, NN - 1e-3 * grads

        for _ in range(1, 10_001):
            value, NN = update(NN, x, y)

    @jax.jit
    def update(NN, x, y):
        value, grads = jax.value_and_grad(loss_func)(NN, x, y)
        return value, NN - 1e-3 * grads

    mask = jtu.tree_map(pytc.is_nondiff, NN)
    freezeed_NN = NN.at[mask].apply(pytc.freeze)

    for _ in range(1, 10_001):
        value, freezeed_NN = update(freezeed_NN, x, y)

    np.testing.assert_allclose(value, jnp.array(0.0031012), atol=1e-5)

    X = StackedLinear(in_dim=1, out_dim=1, hidden_dim=10, key=jr.PRNGKey(0))

    frozen_ = jtu.tree_map(pytc.freeze, X)
    unfrozen_ = jtu.tree_map(pytc.unfreeze, frozen_, is_leaf=pytc.is_frozen)
    assert jtu.tree_leaves(X) == jtu.tree_leaves(unfrozen_)


@pytest.mark.benchmark
def test_nn_benchmark(benchmark):
    benchmark(test_nn)


@pytest.mark.benchmark
def test_freeze_nondiff_benchmark(benchmark):
    benchmark(test_freeze_nondiff)
