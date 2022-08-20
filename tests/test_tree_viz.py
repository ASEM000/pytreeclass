from __future__ import annotations

import os
from dataclasses import field
from typing import Any, Callable, Sequence

import jax
import jax.random as jr
from jax import numpy as jnp

import pytreeclass as pytc
from pytreeclass import tree_viz
from pytreeclass.src.tree_util import static_field, static_value


def test_tree_box():
    @pytc.treeclass
    class test:
        a: int = 1

    correct = "┌──────────────┬────────┬──────┐\n│              │ Input  │ None │\n│ test[Parent] │────────┼──────┤\n│              │ Output │ None │\n└──────────────┴────────┴──────┘"  # noqa
    assert tree_viz.tree_box(test()) == correct
    assert test().tree_box() == correct


def test_tree_diagram():
    @pytc.treeclass
    class test:
        a: int = 1

    correct = "test\n    └── a=1 "
    assert tree_viz.tree_diagram(test()) == correct
    assert test().tree_diagram() == correct


def test_model_viz_frozen_value():
    @pytc.treeclass
    class Linear:

        weight: jnp.ndarray
        bias: jnp.ndarray
        notes: str = field(default=static_value("string"))

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

        def __call__(self, x):
            x = self.l1(x)
            x = jax.nn.tanh(x)
            x = self.l2(x)
            x = jax.nn.tanh(x)
            x = self.l3(x)

            return x

    model = StackedLinear(in_dim=1, out_dim=1, key=jax.random.PRNGKey(0))
    x = jnp.linspace(0, 1, 100)[:, None]

    assert (
        (model.freeze().summary())
        # trunk-ignore(flake8/E501)
        == "┌────────┬──────┬───────┬───────┬───────────────────┐\n│Name    │Type  │Param #│Size   │Config             │\n├────────┼──────┼───────┼───────┼───────────────────┤\n│l1      │Linear│256    │1.00KB │weight=f32[1,128]  │\n│(frozen)│      │(0)    │(0.00B)│bias=f32[1,128]    │\n├────────┼──────┼───────┼───────┼───────────────────┤\n│l2      │Linear│16,512 │64.50KB│weight=f32[128,128]│\n│(frozen)│      │(0)    │(0.00B)│bias=f32[1,128]    │\n├────────┼──────┼───────┼───────┼───────────────────┤\n│l3      │Linear│129    │516.00B│weight=f32[128,1]  │\n│(frozen)│      │(0)    │(0.00B)│bias=f32[1,1]      │\n└────────┴──────┴───────┴───────┴───────────────────┘\nTotal # :\t\t16,897(0)\nDynamic #:\t\t0(0)\nStatic/Frozen #:\t16,897(0)\n-----------------------------------------------------\nTotal size :\t\t66.00KB(0.00B)\nDynamic size:\t\t0.00B(0.00B)\nStatic/Frozen size:\t66.00KB(0.00B)\n====================================================="
    )
    assert (
        (model.summary())
        # trunk-ignore(flake8/E501)
        == "┌────┬──────┬───────┬───────┬───────────────────┐\n│Name│Type  │Param #│Size   │Config             │\n├────┼──────┼───────┼───────┼───────────────────┤\n│l1  │Linear│256    │1.00KB │weight=f32[1,128]  │\n│    │      │(0)    │(0.00B)│bias=f32[1,128]    │\n├────┼──────┼───────┼───────┼───────────────────┤\n│l2  │Linear│16,512 │64.50KB│weight=f32[128,128]│\n│    │      │(0)    │(0.00B)│bias=f32[1,128]    │\n├────┼──────┼───────┼───────┼───────────────────┤\n│l3  │Linear│129    │516.00B│weight=f32[128,1]  │\n│    │      │(0)    │(0.00B)│bias=f32[1,1]      │\n└────┴──────┴───────┴───────┴───────────────────┘\nTotal # :\t\t16,897(0)\nDynamic #:\t\t16,897(0)\nStatic/Frozen #:\t0(0)\n-------------------------------------------------\nTotal size :\t\t66.00KB(0.00B)\nDynamic size:\t\t66.00KB(0.00B)\nStatic/Frozen size:\t0.00B(0.00B)\n================================================="
    )
    assert (
        (model.summary(array=x))
        # trunk-ignore(flake8/E501)
        == "┌────┬──────┬───────┬───────┬───────────────────┬────────────┐\n│Name│Type  │Param #│Size   │Config             │Input/Output│\n├────┼──────┼───────┼───────┼───────────────────┼────────────┤\n│l1  │Linear│256    │1.00KB │weight=f32[1,128]  │f32[100,1]  │\n│    │      │(0)    │(0.00B)│bias=f32[1,128]    │f32[100,128]│\n├────┼──────┼───────┼───────┼───────────────────┼────────────┤\n│l2  │Linear│16,512 │64.50KB│weight=f32[128,128]│f32[100,128]│\n│    │      │(0)    │(0.00B)│bias=f32[1,128]    │f32[100,128]│\n├────┼──────┼───────┼───────┼───────────────────┼────────────┤\n│l3  │Linear│129    │516.00B│weight=f32[128,1]  │f32[100,128]│\n│    │      │(0)    │(0.00B)│bias=f32[1,1]      │f32[100,1]  │\n└────┴──────┴───────┴───────┴───────────────────┴────────────┘\nTotal # :\t\t16,897(0)\nDynamic #:\t\t16,897(0)\nStatic/Frozen #:\t0(0)\n--------------------------------------------------------------\nTotal size :\t\t66.00KB(0.00B)\nDynamic size:\t\t66.00KB(0.00B)\nStatic/Frozen size:\t0.00B(0.00B)\n=============================================================="
    )
    assert (
        (model.freeze().summary(array=x))
        # trunk-ignore(flake8/E501)
        == "┌────────┬──────┬───────┬───────┬───────────────────┬────────────┐\n│Name    │Type  │Param #│Size   │Config             │Input/Output│\n├────────┼──────┼───────┼───────┼───────────────────┼────────────┤\n│l1      │Linear│256    │1.00KB │weight=f32[1,128]  │f32[100,1]  │\n│(frozen)│      │(0)    │(0.00B)│bias=f32[1,128]    │f32[100,128]│\n├────────┼──────┼───────┼───────┼───────────────────┼────────────┤\n│l2      │Linear│16,512 │64.50KB│weight=f32[128,128]│f32[100,128]│\n│(frozen)│      │(0)    │(0.00B)│bias=f32[1,128]    │f32[100,128]│\n├────────┼──────┼───────┼───────┼───────────────────┼────────────┤\n│l3      │Linear│129    │516.00B│weight=f32[128,1]  │f32[100,128]│\n│(frozen)│      │(0)    │(0.00B)│bias=f32[1,1]      │f32[100,1]  │\n└────────┴──────┴───────┴───────┴───────────────────┴────────────┘\nTotal # :\t\t16,897(0)\nDynamic #:\t\t0(0)\nStatic/Frozen #:\t16,897(0)\n------------------------------------------------------------------\nTotal size :\t\t66.00KB(0.00B)\nDynamic size:\t\t0.00B(0.00B)\nStatic/Frozen size:\t66.00KB(0.00B)\n=================================================================="
    )

    assert (
        (model.freeze().tree_diagram())
        # trunk-ignore(flake8/E501)
        == "StackedLinear\n    ├#─ l1=Linear\n    │   ├#─ weight=f32[1,128]\n    │   ├#─ bias=f32[1,128]\n    │   └#─ notes=*'string' \n    ├#─ l2=Linear\n    │   ├#─ weight=f32[128,128]\n    │   ├#─ bias=f32[1,128]\n    │   └#─ notes=*'string' \n    └#─ l3=Linear\n        ├#─ weight=f32[128,1]\n        ├#─ bias=f32[1,1]\n        └#─ notes=*'string'     "
    )
    assert (
        (model.tree_diagram())
        # trunk-ignore(flake8/E501)
        == "StackedLinear\n    ├── l1=Linear\n    │   ├── weight=f32[1,128]\n    │   ├── bias=f32[1,128]\n    │   └── notes=*'string' \n    ├── l2=Linear\n    │   ├── weight=f32[128,128]\n    │   ├── bias=f32[1,128]\n    │   └── notes=*'string' \n    └── l3=Linear\n        ├── weight=f32[128,1]\n        ├── bias=f32[1,1]\n        └── notes=*'string'     "
    )
    assert (
        (model.freeze().tree_diagram())
        # trunk-ignore(flake8/E501)
        == "StackedLinear\n    ├#─ l1=Linear\n    │   ├#─ weight=f32[1,128]\n    │   ├#─ bias=f32[1,128]\n    │   └#─ notes=*'string' \n    ├#─ l2=Linear\n    │   ├#─ weight=f32[128,128]\n    │   ├#─ bias=f32[1,128]\n    │   └#─ notes=*'string' \n    └#─ l3=Linear\n        ├#─ weight=f32[128,1]\n        ├#─ bias=f32[1,1]\n        └#─ notes=*'string'     "
    )

    assert (
        tree_viz.tree_summary_md(model)
        # trunk-ignore(flake8/E501)
        == "<table>\n<tr>\n<td align = 'center'> Name </td>\n<td align = 'center'> Type </td>\n<td align = 'center'> Param #</td>\n<td align = 'center'> Size </td>\n<td align = 'center'> Config </td>\n<td align = 'center'> Input/Output </td>\n</tr>\n<tr><td align = 'center'> l1 </td><td align = 'center'> Linear </td><td align = 'center'> 256\n(0) </td><td align = 'center'> 1.00KB\n(0.00B) </td><td align = 'center'> weight=f32[1,128]<br>bias=f32[1,128] </td><td align = 'center'>  </td></tr><tr><td align = 'center'> l2 </td><td align = 'center'> Linear </td><td align = 'center'> 16,512\n(0) </td><td align = 'center'> 64.50KB\n(0.00B) </td><td align = 'center'> weight=f32[128,128]<br>bias=f32[1,128] </td><td align = 'center'>  </td></tr><tr><td align = 'center'> l3 </td><td align = 'center'> Linear </td><td align = 'center'> 129\n(0) </td><td align = 'center'> 516.00B\n(0.00B) </td><td align = 'center'> weight=f32[128,1]<br>bias=f32[1,1] </td><td align = 'center'>  </td></tr></table>\n\n#### Summary\n<table><tr><td>Total #</td><td>16,897(0)</td></tr><tr><td>Dynamic #</td><td>16,897(0)</td></tr><tr><td>Static/Frozen #</td><td>0(0)</td></tr><tr><td>Total size</td><td>66.00KB(0.00B)</td></tr><tr><td>Dynamic size</td><td>66.00KB(0.00B)</td></tr><tr><td>Static/Frozen size</td><td>0.00B(0.00B)</td></tr></table>"
    )
    assert (
        tree_viz.tree_summary_md(model, array=x)
        # trunk-ignore(flake8/E501)
        == "<table>\n<tr>\n<td align = 'center'> Name </td>\n<td align = 'center'> Type </td>\n<td align = 'center'> Param #</td>\n<td align = 'center'> Size </td>\n<td align = 'center'> Config </td>\n<td align = 'center'> Input/Output </td>\n</tr>\n<tr><td align = 'center'> l1 </td><td align = 'center'> Linear </td><td align = 'center'> 256\n(0) </td><td align = 'center'> 1.00KB\n(0.00B) </td><td align = 'center'> weight=f32[1,128]<br>bias=f32[1,128] </td><td align = 'center'> f32[100,1]\nf32[100,128] </td></tr><tr><td align = 'center'> l2 </td><td align = 'center'> Linear </td><td align = 'center'> 16,512\n(0) </td><td align = 'center'> 64.50KB\n(0.00B) </td><td align = 'center'> weight=f32[128,128]<br>bias=f32[1,128] </td><td align = 'center'> f32[100,128]\nf32[100,128] </td></tr><tr><td align = 'center'> l3 </td><td align = 'center'> Linear </td><td align = 'center'> 129\n(0) </td><td align = 'center'> 516.00B\n(0.00B) </td><td align = 'center'> weight=f32[128,1]<br>bias=f32[1,1] </td><td align = 'center'> f32[100,128]\nf32[100,1] </td></tr></table>\n\n#### Summary\n<table><tr><td>Total #</td><td>16,897(0)</td></tr><tr><td>Dynamic #</td><td>16,897(0)</td></tr><tr><td>Static/Frozen #</td><td>0(0)</td></tr><tr><td>Total size</td><td>66.00KB(0.00B)</td></tr><tr><td>Dynamic size</td><td>66.00KB(0.00B)</td></tr><tr><td>Static/Frozen size</td><td>0.00B(0.00B)</td></tr></table>"
    )
    assert (
        tree_viz.tree_summary_md(model.freeze())
        # trunk-ignore(flake8/E501)
        == "<table>\n<tr>\n<td align = 'center'> Name </td>\n<td align = 'center'> Type </td>\n<td align = 'center'> Param #</td>\n<td align = 'center'> Size </td>\n<td align = 'center'> Config </td>\n<td align = 'center'> Input/Output </td>\n</tr>\n<tr><td align = 'center'> l1<br>(frozen) </td><td align = 'center'> Linear </td><td align = 'center'> 256\n(0) </td><td align = 'center'> 1.00KB\n(0.00B) </td><td align = 'center'> weight=f32[1,128]<br>bias=f32[1,128] </td><td align = 'center'>  </td></tr><tr><td align = 'center'> l2<br>(frozen) </td><td align = 'center'> Linear </td><td align = 'center'> 16,512\n(0) </td><td align = 'center'> 64.50KB\n(0.00B) </td><td align = 'center'> weight=f32[128,128]<br>bias=f32[1,128] </td><td align = 'center'>  </td></tr><tr><td align = 'center'> l3<br>(frozen) </td><td align = 'center'> Linear </td><td align = 'center'> 129\n(0) </td><td align = 'center'> 516.00B\n(0.00B) </td><td align = 'center'> weight=f32[128,1]<br>bias=f32[1,1] </td><td align = 'center'>  </td></tr></table>\n\n#### Summary\n<table><tr><td>Total #</td><td>16,897(0)</td></tr><tr><td>Dynamic #</td><td>0(0)</td></tr><tr><td>Static/Frozen #</td><td>16,897(0)</td></tr><tr><td>Total size</td><td>66.00KB(0.00B)</td></tr><tr><td>Dynamic size</td><td>0.00B(0.00B)</td></tr><tr><td>Static/Frozen size</td><td>66.00KB(0.00B)</td></tr></table>"
    )
    assert (
        tree_viz.tree_summary_md(model.freeze(), array=x)
        # trunk-ignore(flake8/E501)
        == "<table>\n<tr>\n<td align = 'center'> Name </td>\n<td align = 'center'> Type </td>\n<td align = 'center'> Param #</td>\n<td align = 'center'> Size </td>\n<td align = 'center'> Config </td>\n<td align = 'center'> Input/Output </td>\n</tr>\n<tr><td align = 'center'> l1<br>(frozen) </td><td align = 'center'> Linear </td><td align = 'center'> 256\n(0) </td><td align = 'center'> 1.00KB\n(0.00B) </td><td align = 'center'> weight=f32[1,128]<br>bias=f32[1,128] </td><td align = 'center'> f32[100,1]\nf32[100,128] </td></tr><tr><td align = 'center'> l2<br>(frozen) </td><td align = 'center'> Linear </td><td align = 'center'> 16,512\n(0) </td><td align = 'center'> 64.50KB\n(0.00B) </td><td align = 'center'> weight=f32[128,128]<br>bias=f32[1,128] </td><td align = 'center'> f32[100,128]\nf32[100,128] </td></tr><tr><td align = 'center'> l3<br>(frozen) </td><td align = 'center'> Linear </td><td align = 'center'> 129\n(0) </td><td align = 'center'> 516.00B\n(0.00B) </td><td align = 'center'> weight=f32[128,1]<br>bias=f32[1,1] </td><td align = 'center'> f32[100,128]\nf32[100,1] </td></tr></table>\n\n#### Summary\n<table><tr><td>Total #</td><td>16,897(0)</td></tr><tr><td>Dynamic #</td><td>0(0)</td></tr><tr><td>Static/Frozen #</td><td>16,897(0)</td></tr><tr><td>Total size</td><td>66.00KB(0.00B)</td></tr><tr><td>Dynamic size</td><td>0.00B(0.00B)</td></tr><tr><td>Static/Frozen size</td><td>66.00KB(0.00B)</td></tr></table>"
    )


def test_model_viz_frozen_field():
    @pytc.treeclass
    class Linear:

        weight: jnp.ndarray
        bias: jnp.ndarray
        notes: str = static_field(default=("string"))

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

        def __call__(self, x):
            x = self.l1(x)
            x = jax.nn.tanh(x)
            x = self.l2(x)
            x = jax.nn.tanh(x)
            x = self.l3(x)

            return x

    model = StackedLinear(in_dim=1, out_dim=1, key=jax.random.PRNGKey(0))
    x = jnp.linspace(0, 1, 100)[:, None]

    assert (
        (model.freeze().summary())
        # trunk-ignore(flake8/E501)
        == "┌────────┬──────┬───────┬───────┬───────────────────┐\n│Name    │Type  │Param #│Size   │Config             │\n├────────┼──────┼───────┼───────┼───────────────────┤\n│l1      │Linear│256    │1.00KB │weight=f32[1,128]  │\n│(frozen)│      │(0)    │(0.00B)│bias=f32[1,128]    │\n├────────┼──────┼───────┼───────┼───────────────────┤\n│l2      │Linear│16,512 │64.50KB│weight=f32[128,128]│\n│(frozen)│      │(0)    │(0.00B)│bias=f32[1,128]    │\n├────────┼──────┼───────┼───────┼───────────────────┤\n│l3      │Linear│129    │516.00B│weight=f32[128,1]  │\n│(frozen)│      │(0)    │(0.00B)│bias=f32[1,1]      │\n└────────┴──────┴───────┴───────┴───────────────────┘\nTotal # :\t\t16,897(0)\nDynamic #:\t\t0(0)\nStatic/Frozen #:\t16,897(0)\n-----------------------------------------------------\nTotal size :\t\t66.00KB(0.00B)\nDynamic size:\t\t0.00B(0.00B)\nStatic/Frozen size:\t66.00KB(0.00B)\n====================================================="
    )
    assert (
        (model.summary())
        # trunk-ignore(flake8/E501)
        == "┌────┬──────┬───────┬───────┬───────────────────┐\n│Name│Type  │Param #│Size   │Config             │\n├────┼──────┼───────┼───────┼───────────────────┤\n│l1  │Linear│256    │1.00KB │weight=f32[1,128]  │\n│    │      │(0)    │(0.00B)│bias=f32[1,128]    │\n├────┼──────┼───────┼───────┼───────────────────┤\n│l2  │Linear│16,512 │64.50KB│weight=f32[128,128]│\n│    │      │(0)    │(0.00B)│bias=f32[1,128]    │\n├────┼──────┼───────┼───────┼───────────────────┤\n│l3  │Linear│129    │516.00B│weight=f32[128,1]  │\n│    │      │(0)    │(0.00B)│bias=f32[1,1]      │\n└────┴──────┴───────┴───────┴───────────────────┘\nTotal # :\t\t16,897(0)\nDynamic #:\t\t16,897(0)\nStatic/Frozen #:\t0(0)\n-------------------------------------------------\nTotal size :\t\t66.00KB(0.00B)\nDynamic size:\t\t66.00KB(0.00B)\nStatic/Frozen size:\t0.00B(0.00B)\n================================================="
    )
    assert (
        (model.summary(array=x))
        # trunk-ignore(flake8/E501)
        == "┌────┬──────┬───────┬───────┬───────────────────┬────────────┐\n│Name│Type  │Param #│Size   │Config             │Input/Output│\n├────┼──────┼───────┼───────┼───────────────────┼────────────┤\n│l1  │Linear│256    │1.00KB │weight=f32[1,128]  │f32[100,1]  │\n│    │      │(0)    │(0.00B)│bias=f32[1,128]    │f32[100,128]│\n├────┼──────┼───────┼───────┼───────────────────┼────────────┤\n│l2  │Linear│16,512 │64.50KB│weight=f32[128,128]│f32[100,128]│\n│    │      │(0)    │(0.00B)│bias=f32[1,128]    │f32[100,128]│\n├────┼──────┼───────┼───────┼───────────────────┼────────────┤\n│l3  │Linear│129    │516.00B│weight=f32[128,1]  │f32[100,128]│\n│    │      │(0)    │(0.00B)│bias=f32[1,1]      │f32[100,1]  │\n└────┴──────┴───────┴───────┴───────────────────┴────────────┘\nTotal # :\t\t16,897(0)\nDynamic #:\t\t16,897(0)\nStatic/Frozen #:\t0(0)\n--------------------------------------------------------------\nTotal size :\t\t66.00KB(0.00B)\nDynamic size:\t\t66.00KB(0.00B)\nStatic/Frozen size:\t0.00B(0.00B)\n=============================================================="
    )

    assert (
        (model.freeze().tree_diagram())
        # trunk-ignore(flake8/E501)
        == "StackedLinear\n    ├#─ l1=Linear\n    │   ├#─ weight=f32[1,128]\n    │   ├#─ bias=f32[1,128]\n    │   └*─ notes='string'  \n    ├#─ l2=Linear\n    │   ├#─ weight=f32[128,128]\n    │   ├#─ bias=f32[1,128]\n    │   └*─ notes='string'  \n    └#─ l3=Linear\n        ├#─ weight=f32[128,1]\n        ├#─ bias=f32[1,1]\n        └*─ notes='string'      "
    )
    assert (
        (model.tree_diagram())
        # trunk-ignore(flake8/E501)
        == "StackedLinear\n    ├── l1=Linear\n    │   ├── weight=f32[1,128]\n    │   ├── bias=f32[1,128]\n    │   └*─ notes='string'  \n    ├── l2=Linear\n    │   ├── weight=f32[128,128]\n    │   ├── bias=f32[1,128]\n    │   └*─ notes='string'  \n    └── l3=Linear\n        ├── weight=f32[128,1]\n        ├── bias=f32[1,1]\n        └*─ notes='string'      "
    )

    assert (
        tree_viz.tree_summary_md(model)
        # trunk-ignore(flake8/E501)
        == "<table>\n<tr>\n<td align = 'center'> Name </td>\n<td align = 'center'> Type </td>\n<td align = 'center'> Param #</td>\n<td align = 'center'> Size </td>\n<td align = 'center'> Config </td>\n<td align = 'center'> Input/Output </td>\n</tr>\n<tr><td align = 'center'> l1 </td><td align = 'center'> Linear </td><td align = 'center'> 256\n(0) </td><td align = 'center'> 1.00KB\n(0.00B) </td><td align = 'center'> weight=f32[1,128]<br>bias=f32[1,128] </td><td align = 'center'>  </td></tr><tr><td align = 'center'> l2 </td><td align = 'center'> Linear </td><td align = 'center'> 16,512\n(0) </td><td align = 'center'> 64.50KB\n(0.00B) </td><td align = 'center'> weight=f32[128,128]<br>bias=f32[1,128] </td><td align = 'center'>  </td></tr><tr><td align = 'center'> l3 </td><td align = 'center'> Linear </td><td align = 'center'> 129\n(0) </td><td align = 'center'> 516.00B\n(0.00B) </td><td align = 'center'> weight=f32[128,1]<br>bias=f32[1,1] </td><td align = 'center'>  </td></tr></table>\n\n#### Summary\n<table><tr><td>Total #</td><td>16,897(0)</td></tr><tr><td>Dynamic #</td><td>16,897(0)</td></tr><tr><td>Static/Frozen #</td><td>0(0)</td></tr><tr><td>Total size</td><td>66.00KB(0.00B)</td></tr><tr><td>Dynamic size</td><td>66.00KB(0.00B)</td></tr><tr><td>Static/Frozen size</td><td>0.00B(0.00B)</td></tr></table>"
    )

    assert (
        tree_viz.tree_summary_md(model, array=x)
        # trunk-ignore(flake8/E501)
        == "<table>\n<tr>\n<td align = 'center'> Name </td>\n<td align = 'center'> Type </td>\n<td align = 'center'> Param #</td>\n<td align = 'center'> Size </td>\n<td align = 'center'> Config </td>\n<td align = 'center'> Input/Output </td>\n</tr>\n<tr><td align = 'center'> l1 </td><td align = 'center'> Linear </td><td align = 'center'> 256\n(0) </td><td align = 'center'> 1.00KB\n(0.00B) </td><td align = 'center'> weight=f32[1,128]<br>bias=f32[1,128] </td><td align = 'center'> f32[100,1]\nf32[100,128] </td></tr><tr><td align = 'center'> l2 </td><td align = 'center'> Linear </td><td align = 'center'> 16,512\n(0) </td><td align = 'center'> 64.50KB\n(0.00B) </td><td align = 'center'> weight=f32[128,128]<br>bias=f32[1,128] </td><td align = 'center'> f32[100,128]\nf32[100,128] </td></tr><tr><td align = 'center'> l3 </td><td align = 'center'> Linear </td><td align = 'center'> 129\n(0) </td><td align = 'center'> 516.00B\n(0.00B) </td><td align = 'center'> weight=f32[128,1]<br>bias=f32[1,1] </td><td align = 'center'> f32[100,128]\nf32[100,1] </td></tr></table>\n\n#### Summary\n<table><tr><td>Total #</td><td>16,897(0)</td></tr><tr><td>Dynamic #</td><td>16,897(0)</td></tr><tr><td>Static/Frozen #</td><td>0(0)</td></tr><tr><td>Total size</td><td>66.00KB(0.00B)</td></tr><tr><td>Dynamic size</td><td>66.00KB(0.00B)</td></tr><tr><td>Static/Frozen size</td><td>0.00B(0.00B)</td></tr></table>"
    )
    assert (
        tree_viz.tree_summary_md(model.freeze())
        # trunk-ignore(flake8/E501)
        == "<table>\n<tr>\n<td align = 'center'> Name </td>\n<td align = 'center'> Type </td>\n<td align = 'center'> Param #</td>\n<td align = 'center'> Size </td>\n<td align = 'center'> Config </td>\n<td align = 'center'> Input/Output </td>\n</tr>\n<tr><td align = 'center'> l1<br>(frozen) </td><td align = 'center'> Linear </td><td align = 'center'> 256\n(0) </td><td align = 'center'> 1.00KB\n(0.00B) </td><td align = 'center'> weight=f32[1,128]<br>bias=f32[1,128] </td><td align = 'center'>  </td></tr><tr><td align = 'center'> l2<br>(frozen) </td><td align = 'center'> Linear </td><td align = 'center'> 16,512\n(0) </td><td align = 'center'> 64.50KB\n(0.00B) </td><td align = 'center'> weight=f32[128,128]<br>bias=f32[1,128] </td><td align = 'center'>  </td></tr><tr><td align = 'center'> l3<br>(frozen) </td><td align = 'center'> Linear </td><td align = 'center'> 129\n(0) </td><td align = 'center'> 516.00B\n(0.00B) </td><td align = 'center'> weight=f32[128,1]<br>bias=f32[1,1] </td><td align = 'center'>  </td></tr></table>\n\n#### Summary\n<table><tr><td>Total #</td><td>16,897(0)</td></tr><tr><td>Dynamic #</td><td>0(0)</td></tr><tr><td>Static/Frozen #</td><td>16,897(0)</td></tr><tr><td>Total size</td><td>66.00KB(0.00B)</td></tr><tr><td>Dynamic size</td><td>0.00B(0.00B)</td></tr><tr><td>Static/Frozen size</td><td>66.00KB(0.00B)</td></tr></table>"
    )
    assert (
        tree_viz.tree_summary_md(model.freeze(), array=x)
        # trunk-ignore(flake8/E501)
        == "<table>\n<tr>\n<td align = 'center'> Name </td>\n<td align = 'center'> Type </td>\n<td align = 'center'> Param #</td>\n<td align = 'center'> Size </td>\n<td align = 'center'> Config </td>\n<td align = 'center'> Input/Output </td>\n</tr>\n<tr><td align = 'center'> l1<br>(frozen) </td><td align = 'center'> Linear </td><td align = 'center'> 256\n(0) </td><td align = 'center'> 1.00KB\n(0.00B) </td><td align = 'center'> weight=f32[1,128]<br>bias=f32[1,128] </td><td align = 'center'> f32[100,1]\nf32[100,128] </td></tr><tr><td align = 'center'> l2<br>(frozen) </td><td align = 'center'> Linear </td><td align = 'center'> 16,512\n(0) </td><td align = 'center'> 64.50KB\n(0.00B) </td><td align = 'center'> weight=f32[128,128]<br>bias=f32[1,128] </td><td align = 'center'> f32[100,128]\nf32[100,128] </td></tr><tr><td align = 'center'> l3<br>(frozen) </td><td align = 'center'> Linear </td><td align = 'center'> 129\n(0) </td><td align = 'center'> 516.00B\n(0.00B) </td><td align = 'center'> weight=f32[128,1]<br>bias=f32[1,1] </td><td align = 'center'> f32[100,128]\nf32[100,1] </td></tr></table>\n\n#### Summary\n<table><tr><td>Total #</td><td>16,897(0)</td></tr><tr><td>Dynamic #</td><td>0(0)</td></tr><tr><td>Static/Frozen #</td><td>16,897(0)</td></tr><tr><td>Total size</td><td>66.00KB(0.00B)</td></tr><tr><td>Dynamic size</td><td>0.00B(0.00B)</td></tr><tr><td>Static/Frozen size</td><td>66.00KB(0.00B)</td></tr></table>"
    )


def test_repr_str():
    @pytc.treeclass
    class Test:
        a: float
        b: float
        c: float
        name: str

    A = Test(10, 20, jnp.array([1, 2, 3, 4, 5]), static_value("A"))
    str_string = f"{A!s}"
    repr_string = f"{A!r}"

    assert str_string == "Test(a=10,b=20,c=[1 2 3 4 5],name=*A)"
    assert repr_string == "Test(a=10,b=20,c=i32[5],name=*'A')"


def test_save_viz():
    @pytc.treeclass
    class level0:
        a: int
        b: int

    @pytc.treeclass
    class level1:
        d: level0 = level0(10, static_value(20))

    @pytc.treeclass
    class level2:
        e: level1 = level1()
        f: level0 = level0(100, static_value(200))

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


def test_tree_indent():
    @pytc.treeclass
    class level1:
        a: int
        b: int
        c: int

    @pytc.treeclass
    class level2:
        d: level1
        e: level1
        name: str = field(default=static_value("A"))

    A = (
        level2(
            d=level1(a=1, b=10, c=jnp.array([1, 2, 3, 4, 5])),
            e=level1(
                a="SomethingWrittenHereSomethingWrittenHere",
                b=20,
                c=jnp.array([-1, -2, -3, -4, -5]),
            ),
            name=pytc.static_value("SomethingWrittenHere"),
        ),
    )

    B = (
        level2(
            d=level1(a=1, b=10, c=jnp.array([1, 2, 3, 4, 5])),
            e=level1(a=1, b=20, c=jnp.array([-1, -2, -3, -4, -5])),
            name=pytc.static_value("SomethingWrittenHere"),
        ),
    )

    assert (
        f"{A!r}"
        # trunk-ignore(flake8/E501)
        == "(level2(\n  d=level1(a=1,b=10,c=i32[5]),\n  e=level1(\n    a='SomethingWrittenHereSomethingWrittenHere',\n    b=20,\n    c=i32[5]),\n  name=*'SomethingWrittenHere'),)"
    )
    assert (
        f"{B!r}"
        # trunk-ignore(flake8/E501)
        == "(level2(\n  d=level1(a=1,b=10,c=i32[5]),\n  e=level1(a=1,b=20,c=i32[5]),\n  name=*'SomethingWrittenHere'),)"
    )


def test_repr_true_false():
    @pytc.treeclass
    class Test:
        a: float = field(repr=False)
        b: float = field(repr=False)
        c: float = field(repr=False)
        name: str = field(repr=False)

    A = Test(10, 20, jnp.array([1, 2, 3, 4, 5]), static_value("A"))

    assert f"{A!r}" == "Test()"

    @pytc.treeclass
    class Test:
        a: float = field(repr=False)
        b: float
        c: float
        name: str

    A = Test(10, 20, jnp.ones([10]), static_value("Test"))

    assert A.__repr__() == "Test(b=20,c=f32[10],name=*'Test')"
    assert (
        A.__str__()
        == "Test(\n  b=20,\n  c=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.],\n  name=*Test)"
    )

    @pytc.treeclass
    class Test:
        a: float = field(repr=False)
        b: float
        c: float
        name: str = field(repr=False)

    A = Test(10, 20, jnp.ones([10]), "Test")

    assert A.__str__() == "Test(b=20,c=[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.],)"
    assert A.__repr__() == "Test(b=20,c=f32[10],)"

    @pytc.treeclass
    class Linear:

        weight: jnp.ndarray
        bias: jnp.ndarray
        notes: str = field(default=static_value("string"))

        def __init__(self, key, in_dim, out_dim):
            self.weight = jax.random.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(
                2 / in_dim
            )
            self.bias = jnp.ones((1, out_dim))

    @pytc.treeclass
    class StackedLinear:
        l1: Linear = field(repr=False)
        l2: Linear
        l3: Linear

        def __init__(self, key, in_dim, out_dim):

            keys = jax.random.split(key, 3)

            self.l1 = Linear(key=keys[0], in_dim=in_dim, out_dim=128)
            self.l2 = Linear(key=keys[1], in_dim=128, out_dim=128)
            self.l3 = Linear(key=keys[2], in_dim=128, out_dim=out_dim)

    model = StackedLinear(in_dim=1, out_dim=1, key=jax.random.PRNGKey(0))

    assert (
        model.tree_diagram()
        # trunk-ignore(flake8/E501)
        == "StackedLinear\n    ├── l2=Linear\n    │   ├── weight=f32[128,128]\n    │   ├── bias=f32[1,128]\n    │   └── notes=*'string' \n    └── l3=Linear\n        ├── weight=f32[128,1]\n        ├── bias=f32[1,1]\n        └── notes=*'string'     "
    )

    assert (
        f"{model!r}"
        # trunk-ignore(flake8/E501)
        == "StackedLinear(\n  l2=Linear(\n    weight=f32[128,128],\n    bias=f32[1,128],\n    notes=*'string'),\n  l3=Linear(\n    weight=f32[128,1],\n    bias=f32[1,1],\n    notes=*'string'))"
    )

    assert (
        f"{model!s}"
        # trunk-ignore(flake8/E501)
        == 'StackedLinear(\n  l2=Linear(\n    weight=\n      [[ 0.0144316  -0.02565258  0.1499472  ...  0.008577    0.03262375\n         0.01743125]\n       [-0.0139193   0.19198167  0.258941   ... -0.12346198  0.01294849\n        -0.2187072 ]\n       [-0.07239359 -0.18226019 -0.3028738  ... -0.14551452 -0.15422817\n        -0.10291965]\n       ...\n       [-0.01665265  0.01209195  0.00641495 ... -0.1831385   0.06862506\n        -0.04054948]\n       [ 0.06876494  0.1895817  -0.28528026 ... -0.01250978 -0.0017787\n        -0.00140986]\n       [-0.00827225 -0.01063784  0.07471714 ... -0.09154531  0.10096554\n         0.11608632]],\n    bias=\n      [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n        1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n        1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n        1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n        1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n        1. 1. 1. 1. 1. 1. 1. 1.]],\n    notes=*string),\n  l3=Linear(\n    weight=\n      [[-1.27816513e-01]\n       [ 1.32774442e-01]\n       [-6.23992570e-02]\n       [-7.49895275e-02]\n       [-3.26067805e-01]\n       [-6.05660751e-02]\n       [ 9.42161772e-03]\n       [ 2.32644677e-02]\n       [ 2.19230186e-02]\n       [ 5.65892598e-03]\n       [ 1.73938096e-01]\n       [-6.12434782e-02]\n       [ 1.57998070e-01]\n       [-4.63032909e-02]\n       [-2.07810000e-01]\n       [-1.11553043e-01]\n       [ 3.34456592e-04]\n       [ 3.62022198e-04]\n       [ 7.67446086e-02]\n       [ 9.32634622e-02]\n       [ 6.82672486e-02]\n       [ 3.77380736e-02]\n       [-3.92685691e-03]\n       [-2.28517000e-02]\n       [ 3.70539159e-01]\n       [-1.07250832e-01]\n       [ 2.90615618e-01]\n       [-4.94155660e-02]\n       [-1.58073619e-01]\n       [ 1.73059702e-02]\n       [ 1.12092093e-01]\n       [ 2.62781501e-01]\n       [ 1.21488310e-01]\n       [ 2.35134047e-02]\n       [ 1.78431377e-01]\n       [-8.47617313e-02]\n       [-2.28548661e-01]\n       [-1.80765197e-01]\n       [ 1.58395842e-01]\n       [-9.32636857e-02]\n       [ 8.50839838e-02]\n       [-2.39499062e-02]\n       [-1.41875058e-01]\n       [-2.35416722e-02]\n       [ 1.53016120e-01]\n       [-6.64851367e-02]\n       [-6.48237243e-02]\n       [-3.47195677e-02]\n       [-6.45313859e-02]\n       [-9.28561985e-02]\n       [ 8.82501807e-03]\n       [ 8.81578326e-02]\n       [-1.10307902e-01]\n       [ 3.76644917e-03]\n       [ 1.77450284e-01]\n       [-2.46017352e-01]\n       [ 1.82314426e-01]\n       [ 2.05066040e-01]\n       [ 1.51756600e-01]\n       [ 1.10387504e-01]\n       [ 1.24458313e-01]\n       [-2.43619345e-02]\n       [-3.66542675e-02]\n       [ 1.20256767e-02]\n       [ 1.15861200e-01]\n       [-6.81490526e-02]\n       [-6.63509741e-02]\n       [ 2.95970023e-01]\n       [-2.56108880e-01]\n       [ 2.41085105e-02]\n       [ 1.62178770e-01]\n       [-9.90648493e-02]\n       [-1.74334608e-02]\n       [ 2.28304099e-02]\n       [ 4.36865306e-03]\n       [ 5.85013963e-02]\n       [-9.30247605e-02]\n       [ 3.41776609e-02]\n       [-1.65961653e-01]\n       [ 2.18492467e-02]\n       [-8.39286372e-02]\n       [ 5.88469356e-02]\n       [-1.20792515e-03]\n       [-1.87232509e-01]\n       [ 3.28620017e-01]\n       [-2.93127503e-02]\n       [-5.68653969e-03]\n       [ 9.76577960e-03]\n       [ 1.66406319e-01]\n       [ 7.57132843e-02]\n       [ 7.52598047e-02]\n       [ 6.68830201e-02]\n       [-2.54682396e-02]\n       [ 1.07006788e-01]\n       [ 1.66288093e-02]\n       [-3.88930887e-02]\n       [-1.02033049e-01]\n       [-2.88383458e-02]\n       [ 1.38657033e-01]\n       [-2.65064180e-01]\n       [ 3.98033708e-02]\n       [ 1.01474397e-01]\n       [-1.57758023e-03]\n       [-7.05675259e-02]\n       [ 1.61069810e-01]\n       [-1.07353851e-01]\n       [ 2.08973810e-02]\n       [ 2.29642868e-01]\n       [ 4.62677144e-02]\n       [-3.10902178e-01]\n       [-2.61312351e-02]\n       [-2.12754250e-01]\n       [ 4.32350487e-03]\n       [ 2.38580763e-01]\n       [ 3.97011787e-02]\n       [-1.26753747e-01]\n       [-1.85061261e-01]\n       [-1.46927118e-01]\n       [ 7.63780961e-04]\n       [-4.70348075e-02]\n       [-1.28862098e-01]\n       [-2.61529356e-01]\n       [-5.51525690e-02]\n       [-4.07684296e-02]\n       [ 1.20138936e-01]\n       [-6.21847659e-02]\n       [ 5.08390032e-02]\n       [ 3.24640907e-02]],\n    bias=[[1.]],\n    notes=*string))'
    )

    assert pytc.tree_viz.tree_mermaid(model, link=True).startswith(
        "Open URL in browser: https://pytreeclass.herokuapp.com/temp/?id="
    )


def test_tree_with_containers():
    @pytc.treeclass
    class Linear:

        weight: jnp.ndarray
        bias: jnp.ndarray

        def __init__(self, in_dim, out_dim, key):
            self.weight = jax.random.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(
                2 / in_dim
            )
            self.bias = jnp.ones((out_dim,))

    @pytc.treeclass
    class MLP:
        layers: Any
        act_func: Callable = field(default=static_value(jax.nn.relu), repr=False)

        def __init__(
            self,
            layers: Sequence[int],
            *,
            act_func: Callable = jax.nn.relu,
            weight_init_func: Callable = jax.nn.initializers.he_normal(),
            bias_init_func: Callable | None = lambda key, shape: jnp.ones(shape),
            key: jr.PRNGKey = jr.PRNGKey(0),
        ):
            # self.act_func = jax.nn.relu
            keys = jr.split(key, len(layers))

            self.layers = [
                Linear(in_dim, out_dim, key=ki)
                for ki, in_dim, out_dim in zip(keys, layers[:-1], layers[1:])
            ]

    model = MLP((1, 2, 1))

    assert (
        f"{model!r}"
        == "MLP(\n  layers=[  \n    Linear(weight=f32[1,2],bias=f32[2]),\n    Linear(weight=f32[2,1],bias=f32[1])],)"
    )

    assert (
        f"{model!s}"
        # trunk-ignore(flake8/E501)
        == "MLP(\n  layers=[  \n    Linear(\n      weight=[[-0.13426289 -0.12849723]],\n      bias=[1. 1.]),\n    Linear(\n      weight=[[ 1.2636864] [-0.0423024]],\n      bias=[1.])],)"
    )

    @pytc.treeclass
    class test:
        a: Any

    assert f"{test({'a':1,'b':'s'*20})}" == "test(a={a:1,b:ssssssssssssssssssss})"
    assert f"{test({'a':1,'b':'s'*20})!r}" == "test(a={a:1,b:'ssssssssssssssssssss'})"

    assert f"{test((1,2,3))!s}" == "test(a=(1,2,3))"
    assert f"{test([1,2,3])!r}" == "test(a=[1,2,3])"
    assert (
        (f"{test([jnp.ones([4,4]),2,3])}")
        == "test(\n  a=[ \n    [[1. 1. 1. 1.]\n     [1. 1. 1. 1.]\n     [1. 1. 1. 1.]\n     [1. 1. 1. 1.]],2,3])"
    )
    assert (f"{test([jnp.ones([4,4]),2,3])!r}") == "test(a=[f32[4,4],2,3])"
    assert (f"{test((jnp.ones([4,4]),2,3))!r}") == "test(a=(f32[4,4],2,3))"
    assert f"{test({'a':1,'b':jnp.array([1,2,3])})!r}" == "test(a={a:1,b:i32[3]})"
    assert f"{test(lambda x:x)!s}" == "test(a=Lambda(x))"
    assert f"{test(lambda x:x)!r}" == "test(a=Lambda(x))"
    assert f"{test(jax.nn.relu)!s}" == "test(a=relu(*args,**kwargs))"
    assert f"{test(jax.nn.relu)!r}" == "test(a=relu(*args,**kwargs))"
    assert test(jax.nn.relu).tree_diagram() == "test\n    └── a=relu(*args,**kwargs)  "
