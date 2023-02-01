from __future__ import annotations

import dataclasses

import jax.tree_util as jtu
from jax import numpy as jnp

import pytreeclass as pytc
from pytreeclass.tree_viz import tree_diagram, tree_repr, tree_str, tree_summary


@pytc.treeclass
class Repr1:
    a: int = 1
    b: str = "string"
    c: float = 1.0
    d: tuple = "a" * 50
    e: list = None
    f: set = None
    g: dict = None
    h: jnp.ndarray = jnp.ones((5, 1))
    i: jnp.ndarray = jnp.ones((1, 6))
    j: jnp.ndarray = jnp.ones((1, 1, 4, 5))

    def __post_init__(self):
        self.e = [10] * 25
        self.f = {1, 2, 3}
        self.g = {"a": "a" * 50, "b": "b" * 50, "c": jnp.ones([5, 5])}


@pytc.treeclass
class Repr2:
    a: jnp.ndarray = jnp.ones((5, 1))
    b: jnp.ndarray = jnp.ones((1, 1))
    c: jnp.ndarray = jnp.ones((1, 1, 4, 5))


@pytc.treeclass
class Linear:
    weight: jnp.ndarray
    bias: jnp.ndarray
    notes: str = pytc.field(nondiff=True, default=("string"))

    def __init__(self, in_dim, out_dim):
        self.weight = jnp.ones((in_dim, out_dim))
        self.bias = jnp.ones((1, out_dim))


@pytc.treeclass
class Repr3:
    l1: Linear = dataclasses.field(repr=False)

    def __init__(self, in_dim, out_dim):
        self.l1 = Linear(in_dim=in_dim, out_dim=128)
        self.l2 = Linear(in_dim=128, out_dim=128)
        self.l3 = Linear(in_dim=128, out_dim=out_dim)


r1 = Repr1()
r2 = Repr2()
r3 = Repr3(in_dim=128, out_dim=10)

mask = jtu.tree_map(pytc.is_nondiff, r1)
r1f = r1.at[mask].apply(pytc.tree_freeze)

mask = r2 == r2
r2f = r2.at[mask].apply(pytc.tree_freeze)


def test_tree_diagram():

    assert (
        tree_diagram(r1)
        == "Repr1\n    ├── a:int=1\n    ├── b:str='string'\n    ├── c:float=1.0\n    ├── d:str='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'\n    ├── e:list\n    │   ├── [0]:int=10\n    │   ├── [1]:int=10\n    │   ├── [2]:int=10\n    │   ├── [3]:int=10\n    │   ├── [4]:int=10\n    │   ├── [5]:int=10\n    │   ├── [6]:int=10\n    │   ├── [7]:int=10\n    │   ├── [8]:int=10\n    │   ├── [9]:int=10\n    │   ├── [10]:int=10\n    │   ├── [11]:int=10\n    │   ├── [12]:int=10\n    │   ├── [13]:int=10\n    │   ├── [14]:int=10\n    │   ├── [15]:int=10\n    │   ├── [16]:int=10\n    │   ├── [17]:int=10\n    │   ├── [18]:int=10\n    │   ├── [19]:int=10\n    │   ├── [20]:int=10\n    │   ├── [21]:int=10\n    │   ├── [22]:int=10\n    │   ├── [23]:int=10\n    │   └── [24]:int=10 \n    ├── f:set={1,2,3}\n    ├── g:dict\n    │   ├-─ a:str='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa'\n    │   ├-─ b:str='bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb'\n    │   └-─ c:Array=f32[5,5]∈[1.0,1.0]    \n    ├── h:Array=f32[5,1]∈[1.0,1.0]\n    ├── i:Array=f32[1,6]∈[1.0,1.0]\n    └── j:Array=f32[1,1,4,5]∈[1.0,1.0]    "
    )

    assert (
        tree_diagram(r2)
        == "Repr2\n    ├── a:Array=f32[5,1]∈[1.0,1.0]\n    ├── b:Array=f32[1,1]∈[1.0,1.0]\n    └── c:Array=f32[1,1,4,5]∈[1.0,1.0]    "
    )
