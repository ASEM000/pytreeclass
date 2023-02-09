from __future__ import annotations

import jax.tree_util as jtu
from jax import numpy as jnp

import pytreeclass as pytc
from pytreeclass.tree_viz import (  # tree_mermaid,
    tree_diagram,
    tree_repr,
    tree_str,
    tree_summary,
)


@pytc.treeclass
class Repr1:
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

    def __post_init__(self):
        self.h = jnp.ones((5, 1))
        self.i = jnp.ones((1, 6))
        self.j = jnp.ones((1, 1, 4, 5))

        self.e = [10] * 5
        self.f = {1, 2, 3}
        self.g = {"a": "a" * 50, "b": "b" * 50, "c": jnp.ones([5, 5])}


r1 = Repr1()
mask = jtu.tree_map(pytc.is_nondiff, r1)
r1f = r1.at[mask].apply(pytc.tree_freeze)


def test_repr():

    assert (
        tree_repr(r1)
        # trunk-ignore(flake8/E501)
        == "Repr1(\n  a=1, \n  b='string', \n  c=1.0, \n  d='aaaaa', \n  e=[10, 10, 10, 10, 10], \n  f={1, 2, 3}, \n  g={\n    a:'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', \n    b:'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', \n    c:f32[5,5] ∈[1.0,1.0] μ(σ)=1.0(0.0)\n  }, \n  h=f32[5,1] ∈[1.0,1.0] μ(σ)=1.0(0.0), \n  i=f32[1,6] ∈[1.0,1.0] μ(σ)=1.0(0.0), \n  j=f32[1,1,4,5] ∈[1.0,1.0] μ(σ)=1.0(0.0)\n)"
    )

    assert (
        tree_repr(r1f)
        # trunk-ignore(flake8/E501)
        == "Repr1(\n  a=#1, \n  b=#'string', \n  c=1.0, \n  d=#'aaaaa', \n  e=[#10, #10, #10, #10, #10], \n  f=#{1, 2, 3}, \n  g={\n    a:#'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', \n    b:#'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', \n    c:f32[5,5] ∈[1.0,1.0] μ(σ)=1.0(0.0)\n  }, \n  h=f32[5,1] ∈[1.0,1.0] μ(σ)=1.0(0.0), \n  i=f32[1,6] ∈[1.0,1.0] μ(σ)=1.0(0.0), \n  j=f32[1,1,4,5] ∈[1.0,1.0] μ(σ)=1.0(0.0)\n)"
    )


def test_str():

    assert (
        tree_str(r1)
        # trunk-ignore(flake8/E501)
        == "Repr1(\n  a=1, \n  b=string, \n  c=1.0, \n  d=aaaaa, \n  e=[10, 10, 10, 10, 10], \n  f={1, 2, 3}, \n  g={\n    a:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa, \n    b:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb, \n    c:\n    [[1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]]\n  }, \n  h=[[1.]\n   [1.]\n   [1.]\n   [1.]\n   [1.]], \n  i=[[1. 1. 1. 1. 1. 1.]], \n  j=[[[[1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]]]]\n)"
    )

    assert (
        tree_str(r1f)
        # trunk-ignore(flake8/E501)
        == "Repr1(\n  a=#1, \n  b=#string, \n  c=1.0, \n  d=#aaaaa, \n  e=[#10, #10, #10, #10, #10], \n  f=#{1, 2, 3}, \n  g={\n    a:#'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa', \n    b:#'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb', \n    c:\n    [[1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]]\n  }, \n  h=[[1.]\n   [1.]\n   [1.]\n   [1.]\n   [1.]], \n  i=[[1. 1. 1. 1. 1. 1.]], \n  j=[[[[1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]\n     [1. 1. 1. 1. 1.]]]]\n)"
    )


def test_tree_summary():

    assert (
        tree_summary(r1, depth=0)
        # trunk-ignore(flake8/E501)
        == "┌────┬─────┬────────────┬──────────────┬────────────────────────────────────┐\n│Name│Type │Leaf #(size)│Frozen #(size)│Type stats                          │\n├────┼─────┼────────────┼──────────────┼────────────────────────────────────┤\n│    │Repr1│68(939.00B) │0(0.00B)      │int:6, str:4, float:1, set:1, f32:56│\n└────┴─────┴────────────┴──────────────┴────────────────────────────────────┘\nTotal leaf count:       68\nNon-frozen leaf count:  68\nFrozen leaf count:      0\n-----------------------------------------------------------------------------\nTotal leaf size:        939.00B\nNon-frozen leaf size:   939.00B\nFrozen leaf size:       0.00B\n=============================================================================\n"
    )

    assert (
        tree_summary(r1, depth=1)
        # trunk-ignore(flake8/E501)
        == "┌────┬─────┬────────────┬──────────────┬─────────────┐\n│Name│Type │Leaf #(size)│Frozen #(size)│Type stats   │\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│a   │int  │1(28.00B)   │0(0.00B)      │int:1        │\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│b   │str  │1(55.00B)   │0(0.00B)      │str:1        │\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│c   │float│1(24.00B)   │0(0.00B)      │float:1      │\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│d   │str  │1(54.00B)   │0(0.00B)      │str:1        │\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│e   │list │5(140.00B)  │0(0.00B)      │int:5        │\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│f   │set  │1(216.00B)  │0(0.00B)      │set:1        │\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│g   │dict │27(298.00B) │0(0.00B)      │str:2, f32:25│\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│h   │Array│5(20.00B)   │0(0.00B)      │f32:5        │\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│i   │Array│6(24.00B)   │0(0.00B)      │f32:6        │\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│j   │Array│20(80.00B)  │0(0.00B)      │f32:20       │\n└────┴─────┴────────────┴──────────────┴─────────────┘\nTotal leaf count:       68\nNon-frozen leaf count:  68\nFrozen leaf count:      0\n------------------------------------------------------\nTotal leaf size:        939.00B\nNon-frozen leaf size:   939.00B\nFrozen leaf size:       0.00B\n======================================================\n"
    )

    assert (
        tree_summary(r1, depth=2)
        == tree_summary(r1)
        # trunk-ignore(flake8/E501)
        == "┌────┬─────┬────────────┬──────────────┬──────────┐\n│Name│Type │Leaf #(size)│Frozen #(size)│Type stats│\n├────┼─────┼────────────┼──────────────┼──────────┤\n│a   │int  │1(28.00B)   │0(0.00B)      │int:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│b   │str  │1(55.00B)   │0(0.00B)      │str:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│c   │float│1(24.00B)   │0(0.00B)      │float:1   │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│d   │str  │1(54.00B)   │0(0.00B)      │str:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│e[0]│int  │1(28.00B)   │0(0.00B)      │int:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│e[1]│int  │1(28.00B)   │0(0.00B)      │int:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│e[2]│int  │1(28.00B)   │0(0.00B)      │int:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│e[3]│int  │1(28.00B)   │0(0.00B)      │int:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│e[4]│int  │1(28.00B)   │0(0.00B)      │int:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│f   │set  │1(216.00B)  │0(0.00B)      │set:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│g.a │str  │1(99.00B)   │0(0.00B)      │str:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│g.b │str  │1(99.00B)   │0(0.00B)      │str:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│g.c │Array│25(100.00B) │0(0.00B)      │f32:25    │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│h   │Array│5(20.00B)   │0(0.00B)      │f32:5     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│i   │Array│6(24.00B)   │0(0.00B)      │f32:6     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│j   │Array│20(80.00B)  │0(0.00B)      │f32:20    │\n└────┴─────┴────────────┴──────────────┴──────────┘\nTotal leaf count:       68\nNon-frozen leaf count:  68\nFrozen leaf count:      0\n---------------------------------------------------\nTotal leaf size:        939.00B\nNon-frozen leaf size:   939.00B\nFrozen leaf size:       0.00B\n===================================================\n"
    )

    assert (
        tree_summary(r1f, depth=0)
        # trunk-ignore(flake8/E501)
        == "┌────┬─────┬────────────┬──────────────┬────────────────────────────────────┐\n│Name│Type │Leaf #(size)│Frozen #(size)│Type stats                          │\n├────┼─────┼────────────┼──────────────┼────────────────────────────────────┤\n│    │Repr1│57(248.00B) │11(691.00B)   │int:6, str:4, float:1, set:1, f32:56│\n└────┴─────┴────────────┴──────────────┴────────────────────────────────────┘\nTotal leaf count:       68\nNon-frozen leaf count:  57\nFrozen leaf count:      11\n-----------------------------------------------------------------------------\nTotal leaf size:        939.00B\nNon-frozen leaf size:   248.00B\nFrozen leaf size:       691.00B\n=============================================================================\n"
    )

    assert (
        tree_summary(r1f, depth=1)
        # trunk-ignore(flake8/E501)
        == "┌────┬─────┬────────────┬──────────────┬─────────────┐\n│Name│Type │Leaf #(size)│Frozen #(size)│Type stats   │\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│a   │int  │0(0.00B)    │1(28.00B)     │int:1        │\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│b   │str  │0(0.00B)    │1(55.00B)     │str:1        │\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│c   │float│1(24.00B)   │0(0.00B)      │float:1      │\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│d   │str  │0(0.00B)    │1(54.00B)     │str:1        │\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│e   │list │0(0.00B)    │5(140.00B)    │int:5        │\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│f   │set  │0(0.00B)    │1(216.00B)    │set:1        │\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│g   │dict │25(100.00B) │2(198.00B)    │str:2, f32:25│\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│h   │Array│5(20.00B)   │0(0.00B)      │f32:5        │\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│i   │Array│6(24.00B)   │0(0.00B)      │f32:6        │\n├────┼─────┼────────────┼──────────────┼─────────────┤\n│j   │Array│20(80.00B)  │0(0.00B)      │f32:20       │\n└────┴─────┴────────────┴──────────────┴─────────────┘\nTotal leaf count:       68\nNon-frozen leaf count:  57\nFrozen leaf count:      11\n------------------------------------------------------\nTotal leaf size:        939.00B\nNon-frozen leaf size:   248.00B\nFrozen leaf size:       691.00B\n======================================================\n"
    )

    assert (
        tree_summary(r1f, depth=2)
        == tree_summary(r1f)
        == "┌────┬─────┬────────────┬──────────────┬──────────┐\n│Name│Type │Leaf #(size)│Frozen #(size)│Type stats│\n├────┼─────┼────────────┼──────────────┼──────────┤\n│a   │int  │0(0.00B)    │1(28.00B)     │int:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│b   │str  │0(0.00B)    │1(55.00B)     │str:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│c   │float│1(24.00B)   │0(0.00B)      │float:1   │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│d   │str  │0(0.00B)    │1(54.00B)     │str:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│e[0]│int  │0(0.00B)    │1(28.00B)     │int:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│e[1]│int  │0(0.00B)    │1(28.00B)     │int:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│e[2]│int  │0(0.00B)    │1(28.00B)     │int:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│e[3]│int  │0(0.00B)    │1(28.00B)     │int:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│e[4]│int  │0(0.00B)    │1(28.00B)     │int:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│f   │set  │0(0.00B)    │1(216.00B)    │set:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│g.a │str  │0(0.00B)    │1(99.00B)     │str:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│g.b │str  │0(0.00B)    │1(99.00B)     │str:1     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│g.c │Array│25(100.00B) │0(0.00B)      │f32:25    │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│h   │Array│5(20.00B)   │0(0.00B)      │f32:5     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│i   │Array│6(24.00B)   │0(0.00B)      │f32:6     │\n├────┼─────┼────────────┼──────────────┼──────────┤\n│j   │Array│20(80.00B)  │0(0.00B)      │f32:20    │\n└────┴─────┴────────────┴──────────────┴──────────┘\nTotal leaf count:       68\nNon-frozen leaf count:  57\nFrozen leaf count:      11\n---------------------------------------------------\nTotal leaf size:        939.00B\nNon-frozen leaf size:   248.00B\nFrozen leaf size:       691.00B\n===================================================\n"
    )


def test_tree_diagram():
    assert (
        tree_diagram(r1)
        # trunk-ignore(flake8/E501)
        == "Repr1\n    ├── a:int=1\n    ├── b:str='string'\n    ├── c:float=1.0\n    ├── d:str='aaaaa'\n    ├── e:list=[10, 10, 10, 10, 10]\n    ├── f:set={1, 2, 3}\n    ├── g:dict\n    │   ├-─ a:str='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'\n    │   ├-─ b:str='bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'\n    │   └-─ c:Array=f32[5,5] ∈[1.0,1.0] μ(σ)=1.0(0.0)   \n    ├── h:Array=f32[5,1] ∈[1.0,1.0] μ(σ)=1.0(0.0)\n    ├── i:Array=f32[1,6] ∈[1.0,1.0] μ(σ)=1.0(0.0)\n    └── j:Array=f32[1,1,4,5] ∈[1.0,1.0] μ(σ)=1.0(0.0)   "
    )

    assert (
        tree_diagram(r1f)
        # trunk-ignore(flake8/E501)
        == "Repr1\n    ├#─ a:int=1\n    ├#─ b:str='string'\n    ├── c:float=1.0\n    ├#─ d:str='aaaaa'\n    ├── e:list=[#10, #10, #10, #10, #10]\n    ├#─ f:set={1, 2, 3}\n    ├── g:dict\n    │   ├-─ a:FrozenWrapper=#'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'\n    │   ├-─ b:FrozenWrapper=#'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'\n    │   └-─ c:Array=f32[5,5] ∈[1.0,1.0] μ(σ)=1.0(0.0)   \n    ├── h:Array=f32[5,1] ∈[1.0,1.0] μ(σ)=1.0(0.0)\n    ├── i:Array=f32[1,6] ∈[1.0,1.0] μ(σ)=1.0(0.0)\n    └── j:Array=f32[1,1,4,5] ∈[1.0,1.0] μ(σ)=1.0(0.0)   "
    )


# def test_tree_mermaid():
#     assert (
#         tree_mermaid(r1)
#         # trunk-ignore(flake8/E501)
#         == 'flowchart LR\n    id15696277213149321320(<b>Repr1</b>)\n    id15696277213149321320 ---> |"1 leaf<br>28.00B"| id159132120600507116["<b>a</b>:int=1"]\n    id15696277213149321320 ---> |"1 leaf<br>55.00B"| id10009280772564895168["<b>b</b>:str=\'string\'"]\n    id15696277213149321320 ---> |"1 leaf<br>24.00B"| id7572222925824649475["<b>c</b>:float=1.0"]\n    id15696277213149321320 ---> |"1 leaf<br>54.00B"| id10865740276892226484["<b>d</b>:str=\'aaaaa\'"]\n    id15696277213149321320 ---> |"5 leaves<br>140.00B"| id2269144855147062920["<b>e</b>:list=[10, 10, 10, 10, 10]"]\n    id15696277213149321320 ---> |"1 leaf<br>216.00B"| id18278831082116368843["<b>f</b>:set={1, 2, 3}"]\n    id15696277213149321320 ---> |"27 leaves<br>298.00B"| id9682235660371205279["<b>g</b>:dict={\n    a:\'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\', \n    b:\'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\', \n    c:f32[5,5] ∈[1.0,1.0] μ(σ)=1.0(0.0)\n}"]\n    id15696277213149321320 ---> |"5 leaves<br>20.00B"| id12975753011438782288["<b>h</b>:Array=f32[5,1] ∈[1.0,1.0] μ(σ)=1.0(0.0)"]\n    id15696277213149321320 ---> |"6 leaves<br>24.00B"| id10538695164698536595["<b>i</b>:Array=f32[1,6] ∈[1.0,1.0] μ(σ)=1.0(0.0)"]\n    id15696277213149321320 ---> |"20 leaves<br>80.00B"| id1942099742953373031["<b>j</b>:Array=f32[1,1,4,5] ∈[1.0,1.0] μ(σ)=1.0(0.0)"]'
#     )

#     assert (
#         tree_mermaid(r1f)
#         # trunk-ignore(flake8/E501)
#         == 'flowchart LR\n    id15696277213149321320(<b>Repr1</b>)\n    id15696277213149321320 --x id159132120600507116["<b>a</b>:int=1"]\n    id15696277213149321320 --x id10009280772564895168["<b>b</b>:str=\'string\'"]\n    id15696277213149321320 ---> |"1 leaf<br>24.00B"| id7572222925824649475["<b>c</b>:float=1.0"]\n    id15696277213149321320 --x id10865740276892226484["<b>d</b>:str=\'aaaaa\'"]\n    id15696277213149321320 ---> |"5 leaves<br>240.00B"| id2269144855147062920["<b>e</b>:list=[#10, #10, #10, #10, #10]"]\n    id15696277213149321320 --x id18278831082116368843["<b>f</b>:set={1, 2, 3}"]\n    id15696277213149321320 ---> |"27 leaves<br>196.00B"| id9682235660371205279["<b>g</b>:dict={\n    a:#\'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa\', \n    b:#\'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb\', \n    c:f32[5,5] ∈[1.0,1.0] μ(σ)=1.0(0.0)\n}"]\n    id15696277213149321320 ---> |"5 leaves<br>20.00B"| id12975753011438782288["<b>h</b>:Array=f32[5,1] ∈[1.0,1.0] μ(σ)=1.0(0.0)"]\n    id15696277213149321320 ---> |"6 leaves<br>24.00B"| id10538695164698536595["<b>i</b>:Array=f32[1,6] ∈[1.0,1.0] μ(σ)=1.0(0.0)"]\n    id15696277213149321320 ---> |"20 leaves<br>80.00B"| id1942099742953373031["<b>j</b>:Array=f32[1,1,4,5] ∈[1.0,1.0] μ(σ)=1.0(0.0)"]'
#     )
