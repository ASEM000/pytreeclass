import jax
from jax import numpy as jnp

from pytreeclass import tree_viz, treeclass


def test_vbox():

    assert tree_viz.vbox("a", " a", "a ") == "┌──┐\n│a │\n├──┤\n│ a│\n├──┤\n│a │\n└──┘"


def test_hbox():
    assert tree_viz.hbox("a", "b", "c") == "┌─┬─┬─┐\n│a│b│c│\n└─┴─┴─┘\n"


def test_tree_box():
    @treeclass
    class test:
        a: int = 1

    correct = "┌──────────────┬────────┬──────┐\n│              │ Input  │ None │\n│ test(Parent) │────────┼──────┤\n│              │ Output │ None │\n└──────────────┴────────┴──────┘"  # noqa
    assert tree_viz.tree_box(test()) == correct


def test_tree_diagram():
    @treeclass
    class test:
        a: int = 1

    correct = "test\n  └── a=1 "
    assert tree_viz.tree_diagram(test()) == correct


@treeclass
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


@treeclass
class StackedLinear:
    l1: Linear
    l2: Linear
    l3: Linear

    def __init__(self, key, in_dim, out_dim):

        keys = jax.random.split(key, 3)

        self.l1 = Linear(key=keys[0], in_dim=in_dim, out_dim=128)
        self.l2 = Linear(key=keys[1], in_dim=128, out_dim=128)
        self.l3 = Linear(key=keys[2], in_dim=128, out_dim=out_dim)

    def __call__(self, x):
        x = self.l1(x)
        x = jax.nn.tanh(x)
        x = self.l2(x)
        x = jax.nn.tanh(x)
        x = self.l3(x)

        return x


x = jnp.linspace(0, 1, 100)[:, None]
y = x**3 + jax.random.uniform(jax.random.PRNGKey(0), (100, 1)) * 0.01

model = StackedLinear(in_dim=1, out_dim=1, key=jax.random.PRNGKey(0))


def test_model():
    assert (
        tree_viz.summary(model)
        # trunk-ignore(flake8/E501)
        == "┌──────┬───────┬─────────┬───────────────────┐\n│Type  │Param #│Size     │Config             │\n├──────┼───────┼─────────┼───────────────────┤\n│Linear│256    │1.000 KB │bias=f32[1,128]    │\n│      │       │         │weight=f32[1,128]  │\n├──────┼───────┼─────────┼───────────────────┤\n│Linear│16,512 │64.500 KB│bias=f32[1,128]    │\n│      │       │         │weight=f32[128,128]│\n├──────┼───────┼─────────┼───────────────────┤\n│Linear│129    │516.000 B│bias=f32[1,1]      │\n│      │       │         │weight=f32[128,1]  │\n└──────┴───────┴─────────┴───────────────────┘\nTotal params :\t16,897\nInexact params:\t16,897\nOther params:\t0\n----------------------------------------------\nTotal size :\t66.004 KB\nInexact size:\t66.004 KB\nOther size:\t0.000 B\n=============================================="
    )
    assert (
        tree_viz.tree_box(model, array=x)
        # trunk-ignore(flake8/E501)
        == "┌──────────────────────────────────────┐\n│StackedLinear(Parent)                 │\n├──────────────────────────────────────┤\n│┌────────────┬────────┬──────────────┐│\n││            │ Input  │ f32[100,1]   ││\n││ Linear(l1) │────────┼──────────────┤│\n││            │ Output │ f32[100,128] ││\n│└────────────┴────────┴──────────────┘│\n│┌────────────┬────────┬──────────────┐│\n││            │ Input  │ f32[100,128] ││\n││ Linear(l2) │────────┼──────────────┤│\n││            │ Output │ f32[100,128] ││\n│└────────────┴────────┴──────────────┘│\n│┌────────────┬────────┬──────────────┐│\n││            │ Input  │ f32[100,128] ││\n││ Linear(l3) │────────┼──────────────┤│\n││            │ Output │ f32[100,1]   ││\n│└────────────┴────────┴──────────────┘│\n└──────────────────────────────────────┘"
    )
    assert (
        tree_viz.tree_diagram(model)
        # trunk-ignore(flake8/E501)
        == "StackedLinear\n  ├── l1=Linear\n  │ ├── weight=f32[1,128]\n  │ └── bias=f32[1,128] \n  ├── l2=Linear\n  │ ├── weight=f32[128,128]\n  │ └── bias=f32[1,128] \n  └──l3=Linear\n    ├── weight=f32[128,1]\n    └── bias=f32[1,1]   "
    )


def test_repr_str():
    @treeclass
    class Test:
        a: float
        b: float
        c: float
        name: str

    A = Test(10, 20, jnp.array([1, 2, 3, 4, 5]), "A")
    str_string = f"{A!s}"
    repr_string = f"{A!r}"

    assert (
        str_string
        == "Test(\n  a=\n    10,\n  b=\n    20,\n  c=\n    [1 2 3 4 5],\n  name=\n    A)"
    )
    assert repr_string == "Test(\n  a=10,\n  b=20,\n  c=i32[5,],\n  name='A')"
