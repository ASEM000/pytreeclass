import os
from dataclasses import field

import jax
from jax import numpy as jnp

import pytreeclass as pytc
from pytreeclass import tree_viz


def test__vbox():

    assert tree_viz._vbox("a", " a", "a ") == "┌──┐\n│a │\n├──┤\n│ a│\n├──┤\n│a │\n└──┘"


def test__hbox():
    assert tree_viz._hbox("a", "b", "c") == "┌─┬─┬─┐\n│a│b│c│\n└─┴─┴─┘\n"


def test_tree_box():
    @pytc.treeclass
    class test:
        a: int = 1

    correct = "┌──────────────┬────────┬──────┐\n│              │ Input  │ None │\n│ test(Parent) │────────┼──────┤\n│              │ Output │ None │\n└──────────────┴────────┴──────┘"  # noqa
    assert tree_viz.tree_box(test()) == correct
    assert test().tree_box() == correct


def test_tree_diagram():
    @pytc.treeclass
    class test:
        a: int = 1

    correct = "test\n    └── a=1 "
    assert tree_viz.tree_diagram(test()) == correct
    assert test().tree_diagram() == correct


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
    l1: Linear
    l2: Linear
    l3: Linear

    def __init__(self, key, in_dim, out_dim):

        keys = jax.random.split(key, 3)

        self.l1 = Linear(key=keys[0], in_dim=in_dim, out_dim=128)
        self.l2 = Linear(key=keys[1], in_dim=128, out_dim=128)
        self.l3 = Linear(key=keys[2], in_dim=128, out_dim=out_dim)


x = jnp.linspace(0, 1, 100)[:, None]
y = x**3 + jax.random.uniform(jax.random.PRNGKey(0), (100, 1)) * 0.01

model = StackedLinear(in_dim=1, out_dim=1, key=jax.random.PRNGKey(0))


def test_model():
    assert (
        tree_viz.summary(model)
        # trunk-ignore(flake8/E501)
        == "┌──────┬───────┬───────┬───────────────────┐\n│Type  │Param #│Size   │Config             │\n├──────┼───────┼───────┼───────────────────┤\n│Linear│256    │1.00KB │weight=f32[1,128]  │\n│      │(0)    │(0.00B)│bias=f32[1,128]    │\n├──────┼───────┼───────┼───────────────────┤\n│Linear│16,512 │64.50KB│weight=f32[128,128]│\n│      │(0)    │(0.00B)│bias=f32[1,128]    │\n├──────┼───────┼───────┼───────────────────┤\n│Linear│129    │516.00B│weight=f32[128,1]  │\n│      │(0)    │(0.00B)│bias=f32[1,1]      │\n└──────┴───────┴───────┴───────────────────┘\nTotal # :\t\t16,897(0)\nDynamic #:\t\t16,897(0)\nStatic/Frozen #:\t0(0)\n--------------------------------------------\nTotal size :\t\t66.00KB(0.00B)\nDynamic size:\t\t66.00KB(0.00B)\nStatic/Frozen size:\t0.00B(0.00B)\n============================================"
    )

    assert (
        model.summary()
        # trunk-ignore(flake8/E501)
        == "┌──────┬───────┬───────┬───────────────────┐\n│Type  │Param #│Size   │Config             │\n├──────┼───────┼───────┼───────────────────┤\n│Linear│256    │1.00KB │weight=f32[1,128]  │\n│      │(0)    │(0.00B)│bias=f32[1,128]    │\n├──────┼───────┼───────┼───────────────────┤\n│Linear│16,512 │64.50KB│weight=f32[128,128]│\n│      │(0)    │(0.00B)│bias=f32[1,128]    │\n├──────┼───────┼───────┼───────────────────┤\n│Linear│129    │516.00B│weight=f32[128,1]  │\n│      │(0)    │(0.00B)│bias=f32[1,1]      │\n└──────┴───────┴───────┴───────────────────┘\nTotal # :\t\t16,897(0)\nDynamic #:\t\t16,897(0)\nStatic/Frozen #:\t0(0)\n--------------------------------------------\nTotal size :\t\t66.00KB(0.00B)\nDynamic size:\t\t66.00KB(0.00B)\nStatic/Frozen size:\t0.00B(0.00B)\n============================================"
    )

    assert (
        tree_viz.tree_box(model, array=x)
        # trunk-ignore(flake8/E501)
        == "┌──────────────────────────────────────┐\n│StackedLinear(Parent)                 │\n├──────────────────────────────────────┤\n│┌────────────┬────────┬──────────────┐│\n││            │ Input  │ f32[100,1]   ││\n││ Linear(l1) │────────┼──────────────┤│\n││            │ Output │ f32[100,128] ││\n│└────────────┴────────┴──────────────┘│\n│┌────────────┬────────┬──────────────┐│\n││            │ Input  │ f32[100,128] ││\n││ Linear(l2) │────────┼──────────────┤│\n││            │ Output │ f32[100,128] ││\n│└────────────┴────────┴──────────────┘│\n│┌────────────┬────────┬──────────────┐│\n││            │ Input  │ f32[100,128] ││\n││ Linear(l3) │────────┼──────────────┤│\n││            │ Output │ f32[100,1]   ││\n│└────────────┴────────┴──────────────┘│\n└──────────────────────────────────────┘"
    )

    assert (
        model.tree_box(array=x)
        # trunk-ignore(flake8/E501)
        == "┌──────────────────────────────────────┐\n│StackedLinear(Parent)                 │\n├──────────────────────────────────────┤\n│┌────────────┬────────┬──────────────┐│\n││            │ Input  │ f32[100,1]   ││\n││ Linear(l1) │────────┼──────────────┤│\n││            │ Output │ f32[100,128] ││\n│└────────────┴────────┴──────────────┘│\n│┌────────────┬────────┬──────────────┐│\n││            │ Input  │ f32[100,128] ││\n││ Linear(l2) │────────┼──────────────┤│\n││            │ Output │ f32[100,128] ││\n│└────────────┴────────┴──────────────┘│\n│┌────────────┬────────┬──────────────┐│\n││            │ Input  │ f32[100,128] ││\n││ Linear(l3) │────────┼──────────────┤│\n││            │ Output │ f32[100,1]   ││\n│└────────────┴────────┴──────────────┘│\n└──────────────────────────────────────┘"
    )

    assert (
        tree_viz.tree_diagram(model)
        # trunk-ignore(flake8/E501)
        == "StackedLinear\n    ├── l1=Linear\n    │   ├── weight=f32[1,128]\n    │   └── bias=f32[1,128] \n    ├── l2=Linear\n    │   ├── weight=f32[128,128]\n    │   └── bias=f32[1,128] \n    └── l3=Linear\n        ├── weight=f32[128,1]\n        └── bias=f32[1,1]       "
    )

    assert (
        model.tree_diagram()
        # trunk-ignore(flake8/E501)
        == "StackedLinear\n    ├── l1=Linear\n    │   ├── weight=f32[1,128]\n    │   └── bias=f32[1,128] \n    ├── l2=Linear\n    │   ├── weight=f32[128,128]\n    │   └── bias=f32[1,128] \n    └── l3=Linear\n        ├── weight=f32[128,1]\n        └── bias=f32[1,1]       "
    )


def test_repr_str():
    @pytc.treeclass
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


def test_save_viz():
    @pytc.treeclass
    class level0:
        a: int
        b: int = pytc.static_field()

    @pytc.treeclass
    class level1:
        d: level0 = level0(10, 20)

    @pytc.treeclass
    class level2:
        e: level1 = level1()
        f: level0 = level0(100, 200)

    model = level2()

    assert (
        tree_viz.save_viz(
            model, os.path.join("tests", "test_diagram"), method="tree_diagram"
        )
        is None
    )
    assert (
        tree_viz.save_viz(
            model, os.path.join("tests", "test_summary"), method="summary"
        )
        is None
    )
    assert (
        tree_viz.save_viz(model, os.path.join("tests", "test_box"), method="tree_box")
        is None
    )
    assert (
        tree_viz.save_viz(
            model,
            os.path.join("tests", "test_mermaid_html"),
            method="tree_mermaid_html",
        )
        is None
    )
    assert (
        tree_viz.save_viz(
            model, os.path.join("tests", "test_mermaid_md"), method="tree_mermaid_md"
        )
        is None
    )
    assert (
        tree_viz.save_viz(
            model, os.path.join("tests", "test_summary_md"), method="summary_md"
        )
        is None
    )


def test_summary_md():
    @pytc.treeclass
    class Linear:

        weight: jnp.ndarray
        bias: jnp.ndarray
        notes: str = field(default="string")

        def __init__(self, key, in_dim, out_dim):
            self.weight = jax.random.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(
                2 / in_dim
            )
            self.bias = jnp.ones((1, out_dim))

    @pytc.treeclass
    class StackedLinear:
        def __init__(self, key, in_dim, out_dim, hidden_dim):

            keys = jax.random.split(key, 3)

            self.l1 = Linear(key=keys[0], in_dim=in_dim, out_dim=hidden_dim)
            self.l2 = Linear(key=keys[1], in_dim=hidden_dim, out_dim=hidden_dim)
            self.l3 = Linear(key=keys[2], in_dim=hidden_dim, out_dim=out_dim)

    model = StackedLinear(in_dim=1, out_dim=1, hidden_dim=10, key=jax.random.PRNGKey(0))

    # trunk-ignore(flake8/E501)
    fmt = "<table>\n<tr>\n<td align = 'center'> Type </td>\n<td align = 'center'> Param #</td>\n<td align = 'center'> Size </td>\n<td align = 'center'> Config </td>\n<td align = 'center'> Output </td>\n</tr>\n<tr><td align = 'center'> Linear </td><td align = 'center'> 20\n(0) </td><td align = 'center'> 80.00B\n(0.00B) </td><td align = 'center'> weight=f32[1,10]<br>bias=f32[1,10] </td><td align = 'center'>  </td></tr><tr><td align = 'center'> Linear </td><td align = 'center'> 110\n(0) </td><td align = 'center'> 440.00B\n(0.00B) </td><td align = 'center'> weight=f32[10,10]<br>bias=f32[1,10] </td><td align = 'center'>  </td></tr><tr><td align = 'center'> Linear </td><td align = 'center'> 11\n(0) </td><td align = 'center'> 44.00B\n(0.00B) </td><td align = 'center'> weight=f32[10,1]<br>bias=f32[1,1] </td><td align = 'center'>  </td></tr></table>\n\n#### Summary\n<table><tr><td>Total #</td><td>141(0)</td></tr><tr><td>Dynamic #</td><td>141(0)</td></tr><tr><td>Static/Frozen #</td><td>0(0)</td></tr><tr><td>Total size</td><td>564.00B(0.00B)</td></tr><tr><td>Dynamic size</td><td>564.00B(0.00B)</td></tr><tr><td>Static/Frozen size</td><td>0.00B(0.00B)</td></tr></table>"
    pred = tree_viz.summary(model, render="md")
    assert pred == fmt

    @pytc.treeclass
    class StackedLinear:
        l1: Linear
        l2: Linear
        l3: Linear

        def __init__(self, key, in_dim, out_dim):

            keys = jax.random.split(key, 3)

            self.l1 = Linear(key=keys[0], in_dim=in_dim, out_dim=128)
            self.l2 = Linear(key=keys[1], in_dim=128, out_dim=128)
            self.l3 = Linear(key=keys[2], in_dim=128, out_dim=out_dim)

    model = StackedLinear(in_dim=1, out_dim=1, key=jax.random.PRNGKey(0))

    model = model.freeze()

    assert (
        model.summary()
        # trunk-ignore(flake8/E501)
        == "┌────────┬───────┬────────┬───────────────────┐\n│Type    │Param #│Size    │Config             │\n├────────┼───────┼────────┼───────────────────┤\n│Linear  │256    │1.03KB  │weight=f32[1,128]  │\n│(frozen)│(1)    │(55.00B)│bias=f32[1,128]    │\n│        │       │        │notes='string'     │\n├────────┼───────┼────────┼───────────────────┤\n│Linear  │16,512 │64.53KB │weight=f32[128,128]│\n│(frozen)│(1)    │(55.00B)│bias=f32[1,128]    │\n│        │       │        │notes='string'     │\n├────────┼───────┼────────┼───────────────────┤\n│Linear  │129    │544.00B │weight=f32[128,1]  │\n│(frozen)│(1)    │(55.00B)│bias=f32[1,1]      │\n│        │       │        │notes='string'     │\n└────────┴───────┴────────┴───────────────────┘\nTotal # :\t\t16,897(3)\nDynamic #:\t\t0(0)\nStatic/Frozen #:\t16,897(3)\n-----------------------------------------------\nTotal size :\t\t66.09KB(165.00B)\nDynamic size:\t\t0.00B(0.00B)\nStatic/Frozen size:\t66.09KB(165.00B)\n==============================================="
    )
