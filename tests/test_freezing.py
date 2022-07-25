from dataclasses import field

import jax
import jax.numpy as jnp

from pytreeclass import tree_viz, treeclass
from pytreeclass.src.tree_util import freeze_nodes, unfreeze_nodes


def test_freezing_unfreezing():
    @treeclass
    class A:
        a: int
        b: int

    a = A(1, 2)
    b = a.freeze()
    c = a.unfreeze()

    assert jax.tree_leaves(a) == [1, 2]
    assert jax.tree_leaves(b) == []
    assert jax.tree_leaves(c) == [1, 2]

    @treeclass
    class A:
        a: int
        b: int

    @treeclass
    class B:
        c: int = 3
        d: A = A(1, 2)

    a = B()
    a.d = a.d.freeze()
    assert a.d.frozen is True
    assert (
        tree_viz.tree_diagram(a)
    ) == "B\n    ├── c=3\n    └── d=A\n        ├#─ a=1\n        └#─ b=2     "

    @treeclass
    class Linear:

        weight: jnp.ndarray
        bias: jnp.ndarray
        notes: str = field(default="string")

        def __init__(self, key, in_dim, out_dim):
            self.weight = jax.random.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(
                2 / in_dim
            )
            self.bias = jnp.ones((1, out_dim))

        def __call__(self, x):
            return x @ self.weight + self.bias

    @treeclass
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

    assert value == 3.9368904

    @treeclass
    class Stacked:
        def __init__(self):
            self.mdl1 = StackedLinear(
                in_dim=1, out_dim=1, hidden_dim=10, key=jax.random.PRNGKey(0)
            )
            self.mdl2 = StackedLinear(
                in_dim=1, out_dim=1, hidden_dim=10, key=jax.random.PRNGKey(0)
            )

    mdl = Stacked()

    frozen_diagram = tree_viz.tree_diagram((freeze_nodes(mdl)))
    # trunk-ignore(flake8/E501)
    fmt = "Stacked\n    ├#─ mdl1=StackedLinear\n    │   ├#─ l1=Linear\n    │   │   ├#─ weight=f32[1,10]\n    │   │   ├#─ bias=f32[1,10]\n    │   │   └x─ notes='string'  \n    │   ├#─ l2=Linear\n    │   │   ├#─ weight=f32[10,10]\n    │   │   ├#─ bias=f32[1,10]\n    │   │   └x─ notes='string'  \n    │   └#─ l3=Linear\n    │       ├#─ weight=f32[10,1]\n    │       ├#─ bias=f32[1,1]\n    │       └x─ notes='string'      \n    └#─ mdl2=StackedLinear\n        ├#─ l1=Linear\n        │   ├#─ weight=f32[1,10]\n        │   ├#─ bias=f32[1,10]\n        │   └x─ notes='string'  \n        ├#─ l2=Linear\n        │   ├#─ weight=f32[10,10]\n        │   ├#─ bias=f32[1,10]\n        │   └x─ notes='string'  \n        └#─ l3=Linear\n            ├#─ weight=f32[10,1]\n            ├#─ bias=f32[1,1]\n            └x─ notes='string'          "
    assert fmt == frozen_diagram

    unfrozen_diagram = tree_viz.tree_diagram(unfreeze_nodes(freeze_nodes(mdl)))
    # trunk-ignore(flake8/E501)
    fmt = "Stacked\n    ├── mdl1=StackedLinear\n    │   ├── l1=Linear\n    │   │   ├── weight=f32[1,10]\n    │   │   ├── bias=f32[1,10]\n    │   │   └x─ notes='string'  \n    │   ├── l2=Linear\n    │   │   ├── weight=f32[10,10]\n    │   │   ├── bias=f32[1,10]\n    │   │   └x─ notes='string'  \n    │   └── l3=Linear\n    │       ├── weight=f32[10,1]\n    │       ├── bias=f32[1,1]\n    │       └x─ notes='string'      \n    └── mdl2=StackedLinear\n        ├── l1=Linear\n        │   ├── weight=f32[1,10]\n        │   ├── bias=f32[1,10]\n        │   └x─ notes='string'  \n        ├── l2=Linear\n        │   ├── weight=f32[10,10]\n        │   ├── bias=f32[1,10]\n        │   └x─ notes='string'  \n        └── l3=Linear\n            ├── weight=f32[10,1]\n            ├── bias=f32[1,1]\n            └x─ notes='string'          "
    assert fmt == unfrozen_diagram
