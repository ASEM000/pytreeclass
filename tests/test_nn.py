from typing import Callable, Sequence

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
import pytest

import pytreeclass as pytc
from pytreeclass._src.tree_util import filter_nondiff, unfilter_nondiff


def test_nn():
    @pytc.treeclass
    class Linear:

        weight: jnp.ndarray
        bias: jnp.ndarray

        def __init__(self, key, in_dim, out_dim):
            self.weight = jax.random.normal(
                key, shape=(in_dim, out_dim)) * jnp.sqrt(2 / in_dim)  # fmt: skip
            self.bias = jnp.ones((1, out_dim))

        def __call__(self, x):
            return x @ self.weight + self.bias

    @pytc.treeclass
    class StackedLinear:
        layers: Sequence[Linear]

        def __init__(self, key, layers):
            keys = jax.random.split(key, len(layers) - 1)

            self.layers = []

            for ki, in_dim, out_dim in zip(keys, layers[:-1], layers[1:]):
                self.layers += [Linear(ki, in_dim, out_dim)]

        def __call__(self, x):
            for layer in self.layers[:-1]:
                x = layer(x)
                x = jax.nn.tanh(x)

            return self.layers[-1](x)

    x = jnp.linspace(0, 1, 100)[:, None]
    y = x**3 + jax.random.uniform(jax.random.PRNGKey(0), (100, 1)) * 0.01

    model = StackedLinear(layers=[1, 128, 128, 1], key=jax.random.PRNGKey(0))

    def loss_func(model, x, y):
        return jnp.mean((model(x) - y) ** 2)

    @jax.jit
    def update(model, x, y):
        value, grads = jax.value_and_grad(loss_func)(model, x, y)
        return value, jax.tree_map(lambda x, y: x - 1e-3 * y, model, grads)

    for _ in range(1, 2001):
        value, model = update(model, x, y)

    np.testing.assert_allclose(value, jnp.array(0.00103019), atol=1e-5)


def test_nn_with_func_input():
    @pytc.treeclass
    class Linear:

        weight: jnp.ndarray
        bias: jnp.ndarray
        act_func: Callable

        def __init__(self, key, in_dim, out_dim, act_func):
            self.act_func = act_func
            self.weight = jax.random.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(
                2 / in_dim
            )
            self.bias = jnp.ones((1, out_dim))

        def __call__(self, x):
            return self.act_func(x @ self.weight + self.bias)

    layer = Linear(key=jax.random.PRNGKey(0), in_dim=1, out_dim=1, act_func=jax.nn.tanh)
    x = jnp.linspace(0, 1, 100)[:, None]
    # y = x**3 + jax.random.uniform(jax.random.PRNGKey(0), (100, 1)) * 0.01
    layer(x)
    return True


def test_compact_nn():
    @pytc.treeclass
    class Linear:

        weight: jnp.ndarray
        bias: jnp.ndarray

        def __init__(self, key, in_dim, out_dim):
            self.weight = jax.random.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(
                2 / in_dim
            )
            self.bias = jnp.ones((1, out_dim))

        def __call__(self, x):
            return x @ self.weight + self.bias

    @pytc.treeclass
    class StackedLinear:
        def __init__(self, key, in_dim, out_dim, hidden_dim):

            keys = jax.random.split(key, 3)

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
    y = x**3 + jax.random.uniform(jax.random.PRNGKey(0), (100, 1)) * 0.01

    model = StackedLinear(in_dim=1, out_dim=1, hidden_dim=10, key=jax.random.PRNGKey(0))

    def loss_func(model, x, y):
        return jnp.mean((model(x) - y) ** 2)

    @jax.jit
    def update(model, x, y):
        value, grads = jax.value_and_grad(loss_func)(model, x, y)

        # no need to use `jax.tree_map` to update the model
        #  as it model is wrapped by @pytc.treeclass
        return value, model - 1e-3 * grads

    for _ in range(1, 10_001):
        value, model = update(model, x, y)

    np.testing.assert_allclose(value, jnp.array(0.0031012), atol=1e-5)


def test_filter_nondiff():
    @pytc.treeclass
    class Linear:

        weight: jnp.ndarray
        bias: jnp.ndarray
        count: int = 0
        use_bias: bool = True

        def __init__(self, key, in_dim, out_dim):
            self.weight = jax.random.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(
                2 / in_dim
            )
            self.bias = jnp.ones((1, out_dim))

        def __call__(self, x):
            return x @ self.weight + self.bias

    @pytc.treeclass
    class StackedLinear:
        name: str = "stack"
        exact_array: jnp.ndarray = jnp.array([1, 2, 3])
        bool_array: jnp.ndarray = jnp.array([True, True])

        def __init__(self, key, in_dim, out_dim, hidden_dim):

            keys = jax.random.split(key, 3)

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
    y = x**3 + jax.random.uniform(jax.random.PRNGKey(0), (100, 1)) * 0.01

    model = StackedLinear(in_dim=1, out_dim=1, hidden_dim=10, key=jax.random.PRNGKey(0))

    def loss_func(model, x, y):
        return jnp.mean((model(x) - y) ** 2)

    with pytest.raises(TypeError):

        @jax.jit
        def update(model, x, y):
            value, grads = jax.value_and_grad(loss_func)(model, x, y)
            return value, model - 1e-3 * grads

        for _ in range(1, 10_001):
            value, model = update(model, x, y)

    @jax.jit
    def update(model, x, y):
        value, grads = jax.value_and_grad(loss_func)(model, x, y)
        return value, model - 1e-3 * grads

    filtered_model = filter_nondiff(model)

    for _ in range(1, 10_001):
        value, filtered_model = update(filtered_model, x, y)

    np.testing.assert_allclose(value, jnp.array(0.0031012), atol=1e-5)

    X = StackedLinear(in_dim=1, out_dim=1, hidden_dim=10, key=jax.random.PRNGKey(0))
    assert jtu.tree_leaves(X) == jtu.tree_leaves(unfilter_nondiff(filter_nondiff(X)))
