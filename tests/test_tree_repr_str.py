from __future__ import annotations

import dataclasses

from jax import numpy as jnp

import pytreeclass as pytc


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

r1f = pytc.tree_filter(r1)
r2f = pytc.tree_filter(r2, where=lambda _: True)


def test_repr():
    assert (
        pytc.tree_unfilter(pytc.tree_filter(r1)).__repr__()
        == r1.__repr__()
        # trunk-ignore(flake8/E501)
        == "Repr1(\n  a=1,\n  b='string',\n  c=1.0,\n  d='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',\n  e=[\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10\n  ],\n  f={1,2,3},\n  g={\n    a:'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',\n    b:'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',\n    c:f32[5,5]\n  },\n  h=f32[5,1],\n  i=f32[1,6],\n  j=f32[1,1,4,5]\n)"
    )
    assert r2.__repr__() == "Repr2(a=f32[5,1],b=f32[1,1],c=f32[1,1,4,5])"
    assert (
        r3.__repr__()
        # trunk-ignore(flake8/E501)
        == "Repr3(\n  l2=Linear(weight=f32[128,128],bias=f32[1,128],*notes='string'),\n  l3=Linear(weight=f32[128,10],bias=f32[1,10],*notes='string')\n)"
    )

    assert (
        r1f.__repr__()
        # trunk-ignore(flake8/E501)
        == "Repr1(\n  #a=1,\n  #b='string',\n  c=1.0,\n  #d='aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',\n  #e=[\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10\n  ],\n  #f={1,2,3},\n  #g={\n    a:'aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa',\n    b:'bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb',\n    c:f32[5,5]\n  },\n  h=f32[5,1],\n  i=f32[1,6],\n  j=f32[1,1,4,5]\n)"
    )


def test_str():
    assert (
        r1.__str__()
        # trunk-ignore(flake8/E501)
        == "Repr1(\n  a=1,\n  b=string,\n  c=1.0,\n  d=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n  e=[\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10,\n    10\n  ],\n  f={1,2,3},\n  g=\n    {\n      a:aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa,\n      b:bbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbbb,\n      c:\n      [[1. 1. 1. 1. 1.]\n       [1. 1. 1. 1. 1.]\n       [1. 1. 1. 1. 1.]\n       [1. 1. 1. 1. 1.]\n       [1. 1. 1. 1. 1.]]\n    },\n  h=\n    [[1.]\n     [1.]\n     [1.]\n     [1.]\n     [1.]],\n  i=[[1. 1. 1. 1. 1. 1.]],\n  j=\n    [[[[1. 1. 1. 1. 1.]\n       [1. 1. 1. 1. 1.]\n       [1. 1. 1. 1. 1.]\n       [1. 1. 1. 1. 1.]]]]\n)"
    )
    assert (
        r2.__str__()
        # trunk-ignore(flake8/E501)
        == "Repr2(\n  a=\n    [[1.]\n     [1.]\n     [1.]\n     [1.]\n     [1.]],\n  b=[[1.]],\n  c=\n    [[[[1. 1. 1. 1. 1.]\n       [1. 1. 1. 1. 1.]\n       [1. 1. 1. 1. 1.]\n       [1. 1. 1. 1. 1.]]]]\n)"
    )
    assert (
        r3.__str__()
        # trunk-ignore(flake8/E501)
        == "Repr3(\n  l2=Linear(\n    weight=\n      [[1. 1. 1. ... 1. 1. 1.]\n       [1. 1. 1. ... 1. 1. 1.]\n       [1. 1. 1. ... 1. 1. 1.]\n       ...\n       [1. 1. 1. ... 1. 1. 1.]\n       [1. 1. 1. ... 1. 1. 1.]\n       [1. 1. 1. ... 1. 1. 1.]],\n    bias=\n      [[1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n        1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n        1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n        1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n        1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1. 1.\n        1. 1. 1. 1. 1. 1. 1. 1.]],\n    *notes=string\n  ),\n  l3=Linear(\n    weight=\n      [[1. 1. 1. ... 1. 1. 1.]\n       [1. 1. 1. ... 1. 1. 1.]\n       [1. 1. 1. ... 1. 1. 1.]\n       ...\n       [1. 1. 1. ... 1. 1. 1.]\n       [1. 1. 1. ... 1. 1. 1.]\n       [1. 1. 1. ... 1. 1. 1.]],\n    bias=[[1. 1. 1. 1. 1. 1. 1. 1. 1. 1.]],\n    *notes=string\n  )\n)"
    )
