import jax
import jax.tree_util as jtu
from jax import numpy as jnp

import pytreeclass as pytc


import numpy.testing as npt

def test_jit_freeze():
    @pytc.treeclass
    class Linear:
        weight: jnp.ndarray
        bias: jnp.ndarray
        name: str = "a"

        def __init__(self, key, in_dim, out_dim):
            self.weight = jax.random.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(
                2 / in_dim
            )
            self.bias = jnp.ones((1, out_dim))

        def __call__(self, x):
            return x @ self.weight + self.bias

    @pytc.treeclass
    class StackedLinear:
        l1: Linear
        l2: Linear
        l3: Linear

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

    model = StackedLinear(in_dim=1, out_dim=1, hidden_dim=10, key=jax.random.PRNGKey(0))
    x = jnp.linspace(0, 1, 100)[:, None]
    y = x**3 + jax.random.uniform(jax.random.PRNGKey(0), (100, 1)) * 0.01

    @jax.value_and_grad
    def loss_func(model, x, y):
        return jnp.mean((model(x) - y) ** 2)

    @jax.jit
    def update(model, x, y):
        value, grads = loss_func(model, x, y)
        return value, jtu.tree_map(lambda x, y: x - 1e-3 * y, model, grads)

    # freeze l1
    def train_step(x, y, epochs=20_000):
        model = StackedLinear(
            in_dim=1, out_dim=1, hidden_dim=10, key=jax.random.PRNGKey(0)
        )
        model.l1 = model.l1.freeze()
        for i in range(1, epochs + 1):
            value, model = update(model, x, y)

        npt.assert_allclose(value, jnp.array(0.0012702086))
        return value, model

    for _ in range(2):
        value, model = train_step(x, y, epochs=20_000)

    # freeze l2
    def train_step(x, y, epochs=20_000):
        model = StackedLinear(
            in_dim=1, out_dim=1, hidden_dim=10, key=jax.random.PRNGKey(0)
        )
        model.l2 = model.l2.freeze()
        for i in range(1, epochs + 1):
            value, model = update(model, x, y)

        npt.assert_allclose(value, jnp.array(0.00619382))
        return value, model

    for _ in range(2):
        value, model = train_step(x, y, epochs=20_000)

    # freeze all
    def train_step(x, y, epochs=20_000):
        model = StackedLinear(
            in_dim=1, out_dim=1, hidden_dim=10, key=jax.random.PRNGKey(0)
        )
        model = model.freeze()
        for i in range(1, epochs + 1):
            value, model = update(model, x, y)

        npt.assert_allclose(value, jnp.array(3.9368904))
        return value, model

    for _ in range(2):
        value, model = train_step(x, y, epochs=20_000)
