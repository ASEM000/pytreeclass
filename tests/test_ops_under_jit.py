import jax
import jax.tree_util as jtu
import numpy.testing as npt
import pytest
from jax import numpy as jnp

import pytreeclass as pytc
from pytreeclass._src.tree_util import tree_freeze


def test_jit_freeze():
    @pytc.treeclass
    class Linear:
        weight: jnp.ndarray
        bias: jnp.ndarray
        name: str = pytc.field(nondiff=True, default=("a"))

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
        model = model.at["l1"].set(tree_freeze(model.l1))
        for i in range(1, epochs + 1):
            value, model = update(model, x, y)

        npt.assert_allclose(value, jnp.array(0.0012702086), atol=1e-4)
        return value, model

    for _ in range(2):
        value, model = train_step(x, y, epochs=20_000)

    # freeze l2
    def train_step(x, y, epochs=20_000):
        model = StackedLinear(
            in_dim=1, out_dim=1, hidden_dim=10, key=jax.random.PRNGKey(0)
        )
        model = model.at["l2"].set(tree_freeze(model.l2))
        for i in range(1, epochs + 1):
            value, model = update(model, x, y)

        npt.assert_allclose(value, jnp.array(0.00619382), atol=1e-4)
        return value, model

    for _ in range(2):
        value, model = train_step(x, y, epochs=20_000)

    # freeze all
    def train_step(x, y, epochs=20_000):
        model = StackedLinear(
            in_dim=1, out_dim=1, hidden_dim=10, key=jax.random.PRNGKey(0)
        )
        model = tree_freeze(model)
        for i in range(1, epochs + 1):
            value, model = update(model, x, y)

        npt.assert_allclose(value, jnp.array(3.9368904), atol=1e-4)
        return value, model

    for _ in range(2):
        value, model = train_step(x, y, epochs=20_000)


def test_ops_with_jit():
    @pytc.treeclass
    class T0:
        a: int = 1
        b: int = 2
        c: int = 3

    @pytc.treeclass
    class T1:
        a: int = 1
        b: int = 2
        c: int = 3
        d: jnp.ndarray = jnp.array([1, 2, 3])

    @jax.jit
    def getter(tree):
        return tree.at[...].get()

    @jax.jit
    def setter(tree):
        return tree.at[...].set(0)

    @jax.jit
    def applier(tree):
        return tree.at[...].apply(lambda _: 0)

    with pytest.raises(jax.errors.ConcretizationTypeError):
        getter(T0())

    assert pytc.is_treeclass_equal(T0(0, 0, 0), setter(T0()))

    assert pytc.is_treeclass_equal(T0(0, 0, 0), applier(T0()))

    with pytest.raises(jax.errors.ConcretizationTypeError):
        getter(T1())

    assert pytc.is_treeclass_equal(T1(0, 0, 0, jnp.array([0, 0, 0])), setter(T1()))

    assert pytc.is_treeclass_equal(T1(0, 0, 0, jnp.array([0, 0, 0])), applier(T1()))
