from dataclasses import field

import jax
import pytest
from jax import numpy as jnp

import pytreeclass as pytc


def test_true_field_only():
    @pytc.treeclass
    class Linear:
        weight: jnp.ndarray
        bias: jnp.ndarray

        def __init__(self, key, in_dim, out_dim):
            self.weight = jax.random.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(
                2 / in_dim
            )
            self.bias = jnp.ones((1, out_dim))

    @pytc.treeclass(field_only=True)
    class StackedLinear:
        def __init__(self, key, in_dim, out_dim, hidden_dim):
            keys = jax.random.split(key, 3)
            self.l1 = Linear(key=keys[0], in_dim=in_dim, out_dim=hidden_dim)
            self.l2 = Linear(key=keys[1], in_dim=hidden_dim, out_dim=hidden_dim)
            self.l3 = Linear(key=keys[2], in_dim=hidden_dim, out_dim=out_dim)

    model = StackedLinear(key=jax.random.PRNGKey(0), in_dim=2, out_dim=2, hidden_dim=2)
    model.__pytree_structure__

    assert "l1" not in model.__dataclass_fields__
    assert "l1" not in model.__undeclared_fields__

    @pytc.treeclass(field_only=True)
    class StackedLinear:
        l4: Linear = field(default=1)
        l5: Linear

        def __init__(self, key, in_dim, out_dim, hidden_dim):
            ...

    with pytest.raises(AttributeError):
        # l5 is not declared
        model = StackedLinear(
            key=jax.random.PRNGKey(0), in_dim=2, out_dim=2, hidden_dim=2
        )
        model.__pytree_structure__


def test_false_field_only():
    @pytc.treeclass
    class Linear:
        weight: jnp.ndarray
        bias: jnp.ndarray

        def __init__(self, key, in_dim, out_dim):
            self.weight = jax.random.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(
                2 / in_dim
            )
            self.bias = jnp.ones((1, out_dim))

    @pytc.treeclass(field_only=False)
    class StackedLinear:
        def __init__(self, key, in_dim, out_dim, hidden_dim):
            keys = jax.random.split(key, 3)
            self.l1 = Linear(key=keys[0], in_dim=in_dim, out_dim=hidden_dim)
            self.l2 = Linear(key=keys[1], in_dim=hidden_dim, out_dim=hidden_dim)
            self.l3 = Linear(key=keys[2], in_dim=hidden_dim, out_dim=out_dim)

    model = StackedLinear(key=jax.random.PRNGKey(0), in_dim=2, out_dim=2, hidden_dim=2)

    assert "l1" in model.__pytree_fields__

    @pytc.treeclass(field_only=False)
    class StackedLinear:
        l4: Linear

        def __init__(self, key, in_dim, out_dim, hidden_dim):
            keys = jax.random.split(key, 3)

            # Declaring l1,l2,l3 as dataclass_fields is optional
            # as l1,l2,l3 are Linear class that is wrapped with @pytc.treeclass
            self.l1 = Linear(key=keys[0], in_dim=in_dim, out_dim=hidden_dim)
            self.l2 = Linear(key=keys[1], in_dim=hidden_dim, out_dim=hidden_dim)
            self.l3 = Linear(key=keys[2], in_dim=hidden_dim, out_dim=out_dim)

    with pytest.raises(AttributeError):
        # l4 is not declared
        model = StackedLinear(
            key=jax.random.PRNGKey(0), in_dim=2, out_dim=2, hidden_dim=2
        )
        model.__pytree_structure__


def test_hash():
    @pytc.treeclass
    class T:
        a: jnp.ndarray

    with pytest.raises(TypeError):
        hash(T(jnp.array([1, 2, 3])))


def test_post_init():
    @pytc.treeclass
    class Test:
        a: int = 1

        def __post_init__(self):
            self.a = 2

    t = Test()

    assert t.a == 2
