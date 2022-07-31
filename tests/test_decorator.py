import jax
import pytest
from jax import numpy as jnp

import pytreeclass as pytc


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


def test_op():
    @pytc.treeclass(op=False)
    class StackedLinear:
        def __init__(self, key, in_dim, out_dim, hidden_dim):
            keys = jax.random.split(key, 3)

            # Declaring l1,l2,l3 as dataclass_fields is optional
            # as l1,l2,l3 are Linear class that is wrapped with @pytc.treeclass
            self.l1 = Linear(key=keys[0], in_dim=in_dim, out_dim=hidden_dim)
            self.l2 = Linear(key=keys[1], in_dim=hidden_dim, out_dim=hidden_dim)
            self.l3 = Linear(key=keys[2], in_dim=hidden_dim, out_dim=out_dim)

    with pytest.raises(TypeError):
        model = StackedLinear(
            key=jax.random.PRNGKey(0), in_dim=2, out_dim=2, hidden_dim=2
        )
        model + model


def test_field_only():
    @pytc.treeclass(field_only=True)
    class StackedLinear:
        def __init__(self, key, in_dim, out_dim, hidden_dim):
            keys = jax.random.split(key, 3)

            # Declaring l1,l2,l3 as dataclass_fields is optional
            # as l1,l2,l3 are Linear class that is wrapped with @pytc.treeclass
            self.l1 = Linear(key=keys[0], in_dim=in_dim, out_dim=hidden_dim)
            self.l2 = Linear(key=keys[1], in_dim=hidden_dim, out_dim=hidden_dim)
            self.l3 = Linear(key=keys[2], in_dim=hidden_dim, out_dim=out_dim)

    model = StackedLinear(key=jax.random.PRNGKey(0), in_dim=2, out_dim=2, hidden_dim=2)
    model.tree_fields

    assert "l1" not in model.__dataclass_fields__

    @pytc.treeclass(field_only=False)
    class StackedLinear:
        def __init__(self, key, in_dim, out_dim, hidden_dim):
            keys = jax.random.split(key, 3)

            # Declaring l1,l2,l3 as dataclass_fields is optional
            # as l1,l2,l3 are Linear class that is wrapped with @pytc.treeclass
            self.l1 = Linear(key=keys[0], in_dim=in_dim, out_dim=hidden_dim)
            self.l2 = Linear(key=keys[1], in_dim=hidden_dim, out_dim=hidden_dim)
            self.l3 = Linear(key=keys[2], in_dim=hidden_dim, out_dim=out_dim)

    model = StackedLinear(key=jax.random.PRNGKey(0), in_dim=2, out_dim=2, hidden_dim=2)
    model.tree_fields

    assert "l1" in model.__dataclass_fields__
