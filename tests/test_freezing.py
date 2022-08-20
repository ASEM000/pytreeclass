from dataclasses import field

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pytreeclass as pytc
from pytreeclass.src.tree_util import (
    _freeze_nodes,
    _unfreeze_nodes,
    is_treeclass_equal,
    static_value,
)


def test_freezing_unfreezing():
    @pytc.treeclass
    class A:
        a: int
        b: int

    a = A(1, 2)
    b = a.freeze()
    c = a.unfreeze()

    assert jax.tree_util.tree_leaves(a) == [1, 2]
    assert jax.tree_util.tree_leaves(b) == []
    assert jax.tree_util.tree_leaves(c) == [1, 2]

    a = A(1, 2)
    b = a.at[...].static()

    assert jax.tree_util.tree_leaves(a) == [1, 2]
    assert jax.tree_util.tree_leaves(b) == []

    @pytc.treeclass
    class A:
        a: int
        b: int

    @pytc.treeclass
    class B:
        c: int = 3
        d: A = A(1, 2)

    a = B()
    a.d = a.d.freeze()
    assert a.d.frozen is True
    assert (
        pytc.tree_viz.tree_diagram(a)
    ) == "B\n    ├── c=3\n    └#─ d=A\n        ├#─ a=1\n        └#─ b=2     "

    @pytc.treeclass(field_only=True)
    class A:
        a: int
        b: int

    a = A(1, 2)
    b = a.freeze()
    c = a.unfreeze()

    assert jax.tree_util.tree_leaves(a) == [1, 2]
    assert jax.tree_util.tree_leaves(b) == []
    assert jax.tree_util.tree_leaves(c) == [1, 2]

    @pytc.treeclass
    class A:
        a: int
        b: int

    @pytc.treeclass
    class B:
        c: int = 3
        d: A = A(1, 2)

    a = B()
    a.d = a.d.freeze()
    assert a.d.frozen is True
    assert (
        pytc.tree_viz.tree_diagram(a)
    ) == "B\n    ├── c=3\n    └#─ d=A\n        ├#─ a=1\n        └#─ b=2     "
    assert (
        a.tree_diagram()
    ) == "B\n    ├── c=3\n    └#─ d=A\n        ├#─ a=1\n        └#─ b=2     "

    @pytc.treeclass
    class Linear:

        weight: jnp.ndarray
        bias: jnp.ndarray
        notes: str = field(default=pytc.static_value("string"))

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

    model = model.freeze()

    def loss_func(model, x, y):
        return jnp.mean((model(x) - y) ** 2)

    @jax.jit
    def update(model, x, y):
        value, grads = jax.value_and_grad(loss_func)(model, x, y)
        return value, model - 1e-3 * grads

    for _ in range(1, 10_001):
        value, model = update(model, x, y)

    np.testing.assert_allclose(value, jnp.array(3.9368), atol=1e-3)

    @pytc.treeclass
    class Stacked:
        def __init__(self):
            self.mdl1 = StackedLinear(
                in_dim=1, out_dim=1, hidden_dim=10, key=jax.random.PRNGKey(0)
            )
            self.mdl2 = StackedLinear(
                in_dim=1, out_dim=1, hidden_dim=10, key=jax.random.PRNGKey(0)
            )

    mdl = Stacked()

    frozen_diagram = pytc.tree_viz.tree_diagram((_freeze_nodes(mdl)))
    # trunk-ignore(flake8/E501)
    fmt = "Stacked\n    ├#─ mdl1=StackedLinear\n    │   ├#─ l1=Linear\n    │   │   ├#─ weight=f32[1,10]\n    │   │   ├#─ bias=f32[1,10]\n    │   │   └#─ notes=*'string'    \n    │   ├#─ l2=Linear\n    │   │   ├#─ weight=f32[10,10]\n    │   │   ├#─ bias=f32[1,10]\n    │   │   └#─ notes=*'string'    \n    │   └#─ l3=Linear\n    │       ├#─ weight=f32[10,1]\n    │       ├#─ bias=f32[1,1]\n    │       └#─ notes=*'string'        \n    └#─ mdl2=StackedLinear\n        ├#─ l1=Linear\n        │   ├#─ weight=f32[1,10]\n        │   ├#─ bias=f32[1,10]\n        │   └#─ notes=*'string'    \n        ├#─ l2=Linear\n        │   ├#─ weight=f32[10,10]\n        │   ├#─ bias=f32[1,10]\n        │   └#─ notes=*'string'    \n        └#─ l3=Linear\n            ├#─ weight=f32[10,1]\n            ├#─ bias=f32[1,1]\n            └#─ notes=*'string'            "

    unfrozen_diagram = pytc.tree_viz.tree_diagram(_unfreeze_nodes(_freeze_nodes(mdl)))
    # trunk-ignore(flake8/E501)
    fmt = "Stacked\n    ├── mdl1=StackedLinear\n    │   ├── l1=Linear\n    │   │   ├── weight=f32[1,10]\n    │   │   ├── bias=f32[1,10]\n    │   │   └── notes=*'string' \n    │   ├── l2=Linear\n    │   │   ├── weight=f32[10,10]\n    │   │   ├── bias=f32[1,10]\n    │   │   └── notes=*'string' \n    │   └── l3=Linear\n    │       ├── weight=f32[10,1]\n    │       ├── bias=f32[1,1]\n    │       └── notes=*'string'     \n    └── mdl2=StackedLinear\n        ├── l1=Linear\n        │   ├── weight=f32[1,10]\n        │   ├── bias=f32[1,10]\n        │   └── notes=*'string' \n        ├── l2=Linear\n        │   ├── weight=f32[10,10]\n        │   ├── bias=f32[1,10]\n        │   └── notes=*'string' \n        └── l3=Linear\n            ├── weight=f32[10,1]\n            ├── bias=f32[1,1]\n            └── notes=*'string'         "

    assert fmt == unfrozen_diagram

    @pytc.treeclass(field_only=True)
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

    x = jnp.linspace(0, 1, 100)[:, None]
    y = x**3 + jax.random.uniform(jax.random.PRNGKey(0), (100, 1)) * 0.01

    model = StackedLinear(in_dim=1, out_dim=1, hidden_dim=10, key=jax.random.PRNGKey(0))

    model = model.freeze()

    @jax.jit
    def update(model, x, y):
        value, grads = jax.value_and_grad(loss_func)(model, x, y)
        return value, model - 1e-3 * grads

    for _ in range(1, 10_001):
        value, model = update(model, x, y)

    np.testing.assert_allclose(value, jnp.array(3.9368), atol=1e-3)

    @pytc.treeclass(field_only=True)
    class Stacked:
        mdl1: StackedLinear
        mdl2: StackedLinear

        def __init__(self):
            self.mdl1 = StackedLinear(
                in_dim=1, out_dim=1, hidden_dim=10, key=jax.random.PRNGKey(0)
            )
            self.mdl2 = StackedLinear(
                in_dim=1, out_dim=1, hidden_dim=10, key=jax.random.PRNGKey(0)
            )

    mdl = Stacked()

    frozen_diagram = pytc.tree_viz.tree_diagram((_freeze_nodes(mdl)))
    # trunk-ignore(flake8/E501)
    fmt = "Stacked\n    ├#─ mdl1=StackedLinear\n    │   ├#─ l1=Linear\n    │   │   ├#─ weight=f32[1,10]\n    │   │   ├#─ bias=f32[1,10]\n    │   │   └#─ notes=*'string' \n    │   ├#─ l2=Linear\n    │   │   ├#─ weight=f32[10,10]\n    │   │   ├#─ bias=f32[1,10]\n    │   │   └#─ notes=*'string' \n    │   └#─ l3=Linear\n    │       ├#─ weight=f32[10,1]\n    │       ├#─ bias=f32[1,1]\n    │       └#─ notes=*'string'     \n    └#─ mdl2=StackedLinear\n        ├#─ l1=Linear\n        │   ├#─ weight=f32[1,10]\n        │   ├#─ bias=f32[1,10]\n        │   └#─ notes=*'string' \n        ├#─ l2=Linear\n        │   ├#─ weight=f32[10,10]\n        │   ├#─ bias=f32[1,10]\n        │   └#─ notes=*'string' \n        └#─ l3=Linear\n            ├#─ weight=f32[10,1]\n            ├#─ bias=f32[1,1]\n            └#─ notes=*'string'         "
    assert fmt == frozen_diagram

    unfrozen_diagram = pytc.tree_viz.tree_diagram(_unfreeze_nodes(_freeze_nodes(mdl)))
    # trunk-ignore(flake8/E501)
    fmt = "Stacked\n    ├── mdl1=StackedLinear\n    │   ├── l1=Linear\n    │   │   ├── weight=f32[1,10]\n    │   │   ├── bias=f32[1,10]\n    │   │   └── notes=*'string' \n    │   ├── l2=Linear\n    │   │   ├── weight=f32[10,10]\n    │   │   ├── bias=f32[1,10]\n    │   │   └── notes=*'string' \n    │   └── l3=Linear\n    │       ├── weight=f32[10,1]\n    │       ├── bias=f32[1,1]\n    │       └── notes=*'string'     \n    └── mdl2=StackedLinear\n        ├── l1=Linear\n        │   ├── weight=f32[1,10]\n        │   ├── bias=f32[1,10]\n        │   └── notes=*'string' \n        ├── l2=Linear\n        │   ├── weight=f32[10,10]\n        │   ├── bias=f32[1,10]\n        │   └── notes=*'string' \n        └── l3=Linear\n            ├── weight=f32[10,1]\n            ├── bias=f32[1,1]\n            └── notes=*'string'         "
    assert fmt == unfrozen_diagram

    @pytc.treeclass(field_only=False)
    class Test:
        a: int = 1
        b: float = field(default=pytc.static_value(1.0))
        c: str = "test"

    t = Test()

    with pytest.raises(ValueError):
        t.freeze().a = 1

    @pytc.treeclass(field_only=True)
    class Test:
        a: int = 1
        b: float = field(default=static_value(1.0))
        c: str = field(default=static_value("test"))

    t = Test()
    assert jax.tree_util.tree_leaves(t) == [1]

    with pytest.raises(ValueError):
        t.freeze().a = 1

    t.unfreeze().a = 1

    hash(t)

    t = Test()
    t.unfreeze()
    t.freeze()
    assert t.frozen is False

    @pytc.treeclass
    class Test:
        a: int

    t = Test(100).freeze()

    with pytest.raises(ValueError):
        t.at[...].set(0)

    with pytest.raises(ValueError):
        t.at[...].apply(lambda x: x + 1)

    with pytest.raises(ValueError):
        t.at[...].reduce(jnp.sin)

    with pytest.raises(ValueError):
        t.at[...].static()

    class T:
        pass

    t = Test(T())

    with pytest.raises(NotImplementedError):
        t.at[...].set(0)

    with pytest.raises(NotImplementedError):
        t.at[...].apply(jnp.sin)

    with pytest.raises(NotImplementedError):
        t.at[...].reduce(jnp.sin)

    with pytest.raises(NotImplementedError):
        t.at[...].static()

    @pytc.treeclass
    class Test:
        x: jnp.ndarray

        def __init__(self, x):
            self.x = x

    t = Test(jnp.array([1, 2, 3]))
    assert is_treeclass_equal(t.at[...].set(None), Test(x=None))
