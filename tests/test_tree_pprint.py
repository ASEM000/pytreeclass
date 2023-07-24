# Copyright 2023 PyTreeClass authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import dataclasses as dc
import re
from collections import namedtuple

# import jax
import jax.tree_util as jtu
import pytest
from jax import numpy as jnp

import pytreeclass as pytc
from pytreeclass import (
    TreeClass,
    tree_diagram,
    tree_graph,
    tree_mermaid,
    tree_repr,
    tree_repr_with_trace,
    tree_str,
    tree_summary,
)


@pytc.leafwise
@pytc.autoinit
class Repr1(TreeClass):
    a: int = 1
    b: str = "string"
    c: float = 1.0
    d: tuple = "a" * 5
    e: list = None
    f: set = None
    g: dict = None
    h: jnp.ndarray = None
    i: jnp.ndarray = None
    j: jnp.ndarray = None
    k: tuple = pytc.field(repr=False, default=(1, 2, 3))
    l: namedtuple = namedtuple("a", ["b", "c"])(1, 2)
    m: jnp.array = jnp.ones((5, 5))
    n: jnp.array = jnp.array(True)
    o: jnp.array = jnp.array([1, 2.0], dtype=jnp.complex64)

    def __post_init__(self):
        self.h = jnp.ones((5, 1))
        self.i = jnp.ones((1, 6))
        self.j = jnp.ones((1, 1, 4, 5))

        self.e = [10] * 5
        self.f = {1, 2, 3}
        self.g = {"a": "a" * 50, "b": "b" * 50, "c": jnp.ones([5, 5])}


r1 = Repr1()


def test_repr():
    assert (
        tree_repr(r1)
        # trunk-ignore(flake8/E501)
        == "Repr1(\n  a=1, \n  b=string, \n  c=1.0, \n  d=aaaaa, \n  e=[10, 10, 10, 10, 10], \n  f={1, 2, 3}, \n  g={\n    a:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, \n    b:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb, \n    c:f32[5,5](μ=1.00, σ=0.00, ∈[1.00,1.00])\n  }, \n  h=f32[5,1](μ=1.00, σ=0.00, ∈[1.00,1.00]), \n  i=f32[1,6](μ=1.00, σ=0.00, ∈[1.00,1.00]), \n  j=f32[1,1,4,5](μ=1.00, σ=0.00, ∈[1.00,1.00]), \n  l=a(b=1, c=2), \n  m=f32[5,5](μ=1.00, σ=0.00, ∈[1.00,1.00]), \n  n=bool[], \n  o=c64[2]\n)"
    )

    assert (
        tree_repr(r1, depth=1)
        # trunk-ignore(flake8/E501)
        == "Repr1(\n  a=1, \n  b=string, \n  c=1.0, \n  d=aaaaa, \n  e=[...], \n  f={...}, \n  g={...}, \n  h=f32[5,1](μ=1.00, σ=0.00, ∈[1.00,1.00]), \n  i=f32[1,6](μ=1.00, σ=0.00, ∈[1.00,1.00]), \n  j=f32[1,1,4,5](μ=1.00, σ=0.00, ∈[1.00,1.00]), \n  l=a(...), \n  m=f32[5,5](μ=1.00, σ=0.00, ∈[1.00,1.00]), \n  n=bool[], \n  o=c64[2]\n)"
    )

    assert tree_repr(r1, depth=0) == "Repr1(...)"


def test_str():
    assert (
        tree_str(r1)
        # trunk-ignore(flake8/E501)
        == "Repr1(\n  a=1, \n  b=string, \n  c=1.0, \n  d=aaaaa, \n  e=[10, 10, 10, 10, 10], \n  f={1, 2, 3}, \n  g={\n    a:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, \n    b:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb, \n    c:[[1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]]\n  }, \n  h=[[1.] [1.] [1.] [1.] [1.]], \n  i=[[1. 1. 1. 1. 1. 1.]], \n  j=[[[[1. 1. 1. 1. 1.]   [1. 1. 1. 1. 1.]   [1. 1. 1. 1. 1.]   [1. 1. 1. 1. 1.]]]], \n  l=a(b=1, c=2), \n  m=[[1. 1. 1. 1. 1.]\n   [1. 1. 1. 1. 1.]\n   [1. 1. 1. 1. 1.]\n   [1. 1. 1. 1. 1.]\n   [1. 1. 1. 1. 1.]], \n  n=True, \n  o=[1.+0.j 2.+0.j]\n)"
    )

    assert (
        tree_str(r1, depth=1)
        # trunk-ignore(flake8/E501)
        == "Repr1(\n  a=1, \n  b=string, \n  c=1.0, \n  d=aaaaa, \n  e=[...], \n  f={...}, \n  g={...}, \n  h=[[1.] [1.] [1.] [1.] [1.]], \n  i=[[1. 1. 1. 1. 1. 1.]], \n  j=[[[[1. 1. 1. 1. 1.]   [1. 1. 1. 1. 1.]   [1. 1. 1. 1. 1.]   [1. 1. 1. 1. 1.]]]], \n  l=a(...), \n  m=[[1. 1. 1. 1. 1.]\n   [1. 1. 1. 1. 1.]\n   [1. 1. 1. 1. 1.]\n   [1. 1. 1. 1. 1.]\n   [1. 1. 1. 1. 1.]], \n  n=True, \n  o=[1.+0.j 2.+0.j]\n)"
    )


def test_tree_summary():
    assert (
        tree_summary(r1, depth=0)
        # trunk-ignore(flake8/E501)
        == "┌────┬─────┬─────┬───────┐\n│Name│Type │Count│Size   │\n├────┼─────┼─────┼───────┤\n│Σ   │Repr1│101  │341.00B│\n└────┴─────┴─────┴───────┘"
    )

    assert (
        tree_summary(r1, depth=1)
        # trunk-ignore(flake8/E501)
        == "┌────┬────────────┬─────┬───────┐\n│Name│Type        │Count│Size   │\n├────┼────────────┼─────┼───────┤\n│.a  │int         │1    │       │\n├────┼────────────┼─────┼───────┤\n│.b  │str         │1    │       │\n├────┼────────────┼─────┼───────┤\n│.c  │float       │1    │       │\n├────┼────────────┼─────┼───────┤\n│.d  │str         │1    │       │\n├────┼────────────┼─────┼───────┤\n│.e  │list        │5    │       │\n├────┼────────────┼─────┼───────┤\n│.f  │set         │1    │       │\n├────┼────────────┼─────┼───────┤\n│.g  │dict        │27   │100.00B│\n├────┼────────────┼─────┼───────┤\n│.h  │f32[5,1]    │5    │20.00B │\n├────┼────────────┼─────┼───────┤\n│.i  │f32[1,6]    │6    │24.00B │\n├────┼────────────┼─────┼───────┤\n│.j  │f32[1,1,4,5]│20   │80.00B │\n├────┼────────────┼─────┼───────┤\n│.k  │tuple       │3    │       │\n├────┼────────────┼─────┼───────┤\n│.l  │a           │2    │       │\n├────┼────────────┼─────┼───────┤\n│.m  │f32[5,5]    │25   │100.00B│\n├────┼────────────┼─────┼───────┤\n│.n  │bool[]      │1    │1.00B  │\n├────┼────────────┼─────┼───────┤\n│.o  │c64[2]      │2    │16.00B │\n├────┼────────────┼─────┼───────┤\n│Σ   │Repr1       │101  │341.00B│\n└────┴────────────┴─────┴───────┘"
    )

    assert (
        tree_summary(r1, depth=2)
        == tree_summary(r1)
        # trunk-ignore(flake8/E501)
        == "┌───────┬────────────┬─────┬───────┐\n│Name   │Type        │Count│Size   │\n├───────┼────────────┼─────┼───────┤\n│.a     │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.b     │str         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.c     │float       │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.d     │str         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.e[0]  │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.e[1]  │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.e[2]  │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.e[3]  │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.e[4]  │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.f     │set         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.g['a']│str         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.g['b']│str         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.g['c']│f32[5,5]    │25   │100.00B│\n├───────┼────────────┼─────┼───────┤\n│.h     │f32[5,1]    │5    │20.00B │\n├───────┼────────────┼─────┼───────┤\n│.i     │f32[1,6]    │6    │24.00B │\n├───────┼────────────┼─────┼───────┤\n│.j     │f32[1,1,4,5]│20   │80.00B │\n├───────┼────────────┼─────┼───────┤\n│.k[0]  │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.k[1]  │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.k[2]  │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.l.b   │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.l.c   │int         │1    │       │\n├───────┼────────────┼─────┼───────┤\n│.m     │f32[5,5]    │25   │100.00B│\n├───────┼────────────┼─────┼───────┤\n│.n     │bool[]      │1    │1.00B  │\n├───────┼────────────┼─────┼───────┤\n│.o     │c64[2]      │2    │16.00B │\n├───────┼────────────┼─────┼───────┤\n│Σ      │Repr1       │101  │341.00B│\n└───────┴────────────┴─────┴───────┘"
    )


def test_tree_diagram():
    assert tree_diagram(r1, depth=0) == "Repr1(...)"

    # trunk-ignore(flake8/E501)
    out = "Repr1\n├── .a=1\n├── .b=string\n├── .c=1.0\n├── .d=aaaaa\n├── .e=[...]\n├── .f={...}\n├── .g={...}\n├── .h=f32[5,1](μ=1.00, σ=0.00, ∈[1.00,1.00])\n├── .i=f32[1,6](μ=1.00, σ=0.00, ∈[1.00,1.00])\n├── .j=f32[1,1,4,5](μ=1.00, σ=0.00, ∈[1.00,1.00])\n├── .k=(...)\n├── .l=a(...)\n├── .m=f32[5,5](μ=1.00, σ=0.00, ∈[1.00,1.00])\n├── .n=bool[]\n└── .o=c64[2]"

    assert tree_diagram(r1, depth=1) == out


def test_custom_jax_class():
    @jtu.register_pytree_node_class
    class Test:
        def __init__(self):
            self.a = 1
            self.b = 2

        def tree_flatten(self):
            return (self.a, self.b), None

        @classmethod
        def tree_unflatten(cls, _, children):
            return cls(*children)

    t = Test()

    out = "Test\n├── .leaf_0=1\n└── .leaf_1=2"
    assert tree_diagram(t) == tree_diagram(t, depth=3) == out

    assert (
        tree_summary(t)
        == tree_summary(t, depth=4)
        # trunk-ignore(flake8/E501)
        == "┌───────┬────┬─────┬────┐\n│Name   │Type│Count│Size│\n├───────┼────┼─────┼────┤\n│.leaf_0│int │1    │    │\n├───────┼────┼─────┼────┤\n│.leaf_1│int │1    │    │\n├───────┼────┼─────┼────┤\n│Σ      │Test│2    │    │\n└───────┴────┴─────┴────┘"
    )

    assert tree_repr(Test) == repr(Test)
    assert tree_str(Test) == str(Test)


def test_tree_mermaid():
    assert (
        re.sub(r"id\d*", "***", tree_mermaid(r1, depth=1))
        # trunk-ignore(flake8/E501)
        == 'flowchart LR\n    ***("<b>Repr1</b>")\n    *** --- ***("<b>.a=1</b>")\n    *** --- ***("<b>.b=string</b>")\n    *** --- ***("<b>.c=1.0</b>")\n    *** --- ***("<b>.d=aaaaa</b>")\n    *** --- ***("<b>.e=[...]</b>")\n    *** --- ***("<b>.f={...}</b>")\n    *** --- ***("<b>.g={...}</b>")\n    *** --- ***("<b>.h=f32[5,1](μ=1.00, σ=0.00, ∈[1.00,1.00])</b>")\n    *** --- ***("<b>.i=f32[1,6](μ=1.00, σ=0.00, ∈[1.00,1.00])</b>")\n    *** --- ***("<b>.j=f32[1,1,4,5](μ=1.00, σ=0.00, ∈[1.00,1.00])</b>")\n    *** --- ***("<b>.k=(...)</b>")\n    *** --- ***("<b>.l=a(...)</b>")\n    *** --- ***("<b>.m=f32[5,5](μ=1.00, σ=0.00, ∈[1.00,1.00])</b>")\n    *** --- ***("<b>.n=bool[]</b>")\n    *** --- ***("<b>.o=c64[2]</b>")'
    )
    assert (
        re.sub(r"id\d*", "***", tree_mermaid(r1, depth=2))
        # trunk-ignore(flake8/E501)
        == 'flowchart LR\n    ***("<b>Repr1</b>")\n    *** --- ***("<b>.a=1</b>")\n    *** --- ***("<b>.b=string</b>")\n    *** --- ***("<b>.c=1.0</b>")\n    *** --- ***("<b>.d=aaaaa</b>")\n    *** --- ***("<b>.e:list</b>")\n    *** --- ***("<b>[0]=10</b>")\n    *** --- ***("<b>[1]=10</b>")\n    *** --- ***("<b>[2]=10</b>")\n    *** --- ***("<b>[3]=10</b>")\n    *** --- ***("<b>[4]=10</b>")\n    *** --- ***("<b>.f={...}</b>")\n    *** --- ***("<b>.g:dict</b>")\n    *** --- ***("<b>[\'a\']=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa</b>")\n    *** --- ***("<b>[\'b\']=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb</b>")\n    *** --- ***("<b>[\'c\']=f32[5,5](μ=1.00, σ=0.00, ∈[1.00,1.00])</b>")\n    *** --- ***("<b>.h=f32[5,1](μ=1.00, σ=0.00, ∈[1.00,1.00])</b>")\n    *** --- ***("<b>.i=f32[1,6](μ=1.00, σ=0.00, ∈[1.00,1.00])</b>")\n    *** --- ***("<b>.j=f32[1,1,4,5](μ=1.00, σ=0.00, ∈[1.00,1.00])</b>")\n    *** --- ***("<b>.k:tuple</b>")\n    *** --- ***("<b>[0]=1</b>")\n    *** --- ***("<b>[1]=2</b>")\n    *** --- ***("<b>[2]=3</b>")\n    *** --- ***("<b>.l:a</b>")\n    *** --- ***("<b>.b=1</b>")\n    *** --- ***("<b>.c=2</b>")\n    *** --- ***("<b>.m=f32[5,5](μ=1.00, σ=0.00, ∈[1.00,1.00])</b>")\n    *** --- ***("<b>.n=bool[]</b>")\n    *** --- ***("<b>.o=c64[2]</b>")'
    )


def test_misc():
    x = (1, 2, 3)
    assert tree_repr(x) == tree_str(x) == "(1, 2, 3)"

    def example(a: int, b=1, *c, d, e=2, **f):
        pass

    assert tree_repr(example) == tree_str(example) == "example(a, b, *c, d, e, **f)"
    assert tree_repr(example, depth=-1) == "..."

    # example = jax.jit(example)
    # assert (
    #     tree_repr(example) == tree_str(example) == "jit(example(a, b, *c, d, e, **f))"
    # )

    assert (
        tree_repr(jnp.ones([1, 2], dtype=jnp.uint16))
        == "ui16[1,2](μ=1.00, σ=0.00, ∈[1,1])"
    )

    @dc.dataclass
    class Test:
        a: int = 1

    assert pytc.tree_repr(Test()) == pytc.tree_str(Test()) == "Test(a=1)"
    assert pytc.tree_repr(Test(), depth=0) == "Test(...)"


def test_extra_tree_diagram():
    @pytc.autoinit
    class L0(TreeClass):
        a: int = 1
        b: int = 2

    @pytc.autoinit
    class L1(TreeClass):
        c: L0 = L0()
        d: int = 3

    @pytc.autoinit
    class L2(TreeClass):
        e: int = 4
        f: L1 = L1()
        g: L0 = L0()
        h: int = 5

    tree = L2()
    # trunk-ignore(flake8/E501)
    out = "L2\n├── .e=4\n├── .f:L1\n│   ├── .c:L0\n│   │   ├── .a=1\n│   │   └── .b=2\n│   └── .d=3\n├── .g:L0\n│   ├── .a=1\n│   └── .b=2\n└── .h=5"

    assert (tree_diagram(tree)) == out

    @pytc.autoinit
    class L0(TreeClass):
        a: int = 1

    @pytc.autoinit
    class L1(TreeClass):
        b: L0 = L0()

    tree = L1()

    assert tree_diagram(tree) == "L1\n└── .b:L0\n    └── .a=1"


def test_invalid_depth():
    with pytest.raises(TypeError):
        tree_diagram(1, depth="a")
    with pytest.raises(TypeError):
        tree_summary(1, depth="a")
    with pytest.raises(TypeError):
        tree_mermaid(1, depth="a")


def test_tree_repr_with_trace():
    @pytc.autoinit
    @pytc.leafwise
    class Test(TreeClass):
        a: int = 1
        b: float = 2.0

    tree = Test()

    assert (
        str(tree_repr_with_trace(tree))
        # trunk-ignore(flake8/E501)
        == "Test(\n  a=\n    ┌─────────┬───┐\n    │Value    │1  │\n    ├─────────┼───┤\n    │Name path│.a │\n    ├─────────┼───┤\n    │Type path│int│\n    └─────────┴───┘, \n  b=\n    ┌─────────┬─────┐\n    │Value    │2.0  │\n    ├─────────┼─────┤\n    │Name path│.b   │\n    ├─────────┼─────┤\n    │Type path│float│\n    └─────────┴─────┘\n)"
    )

    assert (
        str(tree_repr_with_trace(tree, transpose=True))
        # trunk-ignore(flake8/E501)
        == "Test(\n  a=\n    ┌─────┬─────────┬─────────┐\n    │Value│Name path│Type path│\n    ├─────┼─────────┼─────────┤\n    │1    │.a       │int      │\n    └─────┴─────────┴─────────┘, \n  b=\n    ┌─────┬─────────┬─────────┐\n    │Value│Name path│Type path│\n    ├─────┼─────────┼─────────┤\n    │2.0  │.b       │float    │\n    └─────┴─────────┴─────────┘\n)"
    )


def test_tree_graph():
    assert (
        re.sub(r"\b\d{10,}", "***", tree_graph(r1))
        == 'digraph G {\n    *** [label="Repr1", shape=box];\n    *** [label=".a=1", shape=box];\n    *** -> ***;\n    *** [label=".b=string", shape=box];\n    *** -> ***;\n    *** [label=".c=1.0", shape=box];\n    *** -> ***;\n    *** [label=".d=aaaaa", shape=box];\n    *** -> ***;\n    *** [label=".e:list", shape=box];\n    *** -> ***;\n    *** [label="[0]=10", shape=box];\n    *** -> ***;\n    *** [label="[1]=10", shape=box];\n    *** -> ***;\n    *** [label="[2]=10", shape=box];\n    *** -> ***;\n    *** [label="[3]=10", shape=box];\n    *** -> ***;\n    *** [label="[4]=10", shape=box];\n    *** -> ***;\n    *** [label=".f={...}", shape=box];\n    *** -> ***;\n    *** [label=".g:dict", shape=box];\n    *** -> ***;\n    *** [label="[\'a\']=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa", shape=box];\n    *** -> ***;\n    *** [label="[\'b\']=bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb", shape=box];\n    *** -> ***;\n    *** [label="[\'c\']=f32[5,5](μ=1.00, σ=0.00, ∈[1.00,1.00])", shape=box];\n    *** -> ***;\n    *** [label=".h=f32[5,1](μ=1.00, σ=0.00, ∈[1.00,1.00])", shape=box];\n    *** -> ***;\n    *** [label=".i=f32[1,6](μ=1.00, σ=0.00, ∈[1.00,1.00])", shape=box];\n    *** -> ***;\n    *** [label=".j=f32[1,1,4,5](μ=1.00, σ=0.00, ∈[1.00,1.00])", shape=box];\n    *** -> ***;\n    *** [label=".k:tuple", shape=box];\n    *** -> ***;\n    *** [label="[0]=1", shape=box];\n    *** -> ***;\n    *** [label="[1]=2", shape=box];\n    *** -> ***;\n    *** [label="[2]=3", shape=box];\n    *** -> ***;\n    *** [label=".l:a", shape=box];\n    *** -> ***;\n    *** [label=".b=1", shape=box];\n    *** -> ***;\n    *** [label=".c=2", shape=box];\n    *** -> ***;\n    *** [label=".m=f32[5,5](μ=1.00, σ=0.00, ∈[1.00,1.00])", shape=box];\n    *** -> ***;\n    *** [label=".n=bool[]", shape=box];\n    *** -> ***;\n    *** [label=".o=c64[2]", shape=box];\n    *** -> ***;\n}'
    )
