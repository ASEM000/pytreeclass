import jax
import jax.numpy as jnp

import pytreeclass as pytc


def test_node():
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
    class MLP:
        def __init__(self, layers, key=jax.random.PRNGKey(0)):

            keys = jax.random.split(key, len(layers))

            for i, (ki, in_dim, out_dim) in enumerate(
                zip(keys, layers[:-1], layers[1:])
            ):
                self.param(
                    Linear(key=ki, in_dim=in_dim, out_dim=out_dim), name=f"l{i}"
                )

        def __call__(self, x):
            x = self.l0(x)
            x = jax.nn.tanh(x)
            x = self.l1(x)
            x = jax.nn.tanh(x)
            x = self.l2(x)
            return x

    model = MLP(layers=[1, 128, 128, 1])
    x = jnp.linspace(0, 1, 100)[:, None]
    y = x**3 + jax.random.uniform(jax.random.PRNGKey(0), (100, 1)) * 0.01
    print(pytc.tree_viz.tree_diagram(model))

    # leaves,treedef=jax.tree_flatten(model)
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

    assert value < 1e-3
