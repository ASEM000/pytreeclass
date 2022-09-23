import copy

import jax.tree_util as jtu
import pytest
from jax import numpy as jnp

import pytreeclass as pytc
from pytreeclass._src.tree_util import tree_copy


def test_hash():
    @pytc.treeclass
    class T:
        a: jnp.ndarray

    # with pytest.raises(TypeError):
    hash(T(jnp.array([1, 2, 3])))


def test_post_init():
    @pytc.treeclass
    class Test:
        a: int = 1

        def __post_init__(self):
            self.a = 2

    t = Test()

    assert t.a == 2


def test_subclassing():
    @pytc.treeclass
    class L0:
        a: int = 1
        b: int = 3
        c: int = 5

        def inc(self, x):
            return x

        def sub(self, x):
            return x - 10

    @pytc.treeclass
    class L1(L0):
        a: int = 2
        b: int = 4

        def inc(self, x):
            return x + 10

    l1 = L1()

    assert jtu.tree_leaves(l1) == [2, 4, 5]
    assert l1.inc(10) == 20
    assert l1.sub(10) == 0


def test_overriding_setattr():

    with pytest.raises(AttributeError):

        @pytc.treeclass
        class Test:
            a: int = 1

            def __setattr__(self, name, value):
                super().__setattr__(name, value)


def test_nonclass_input():

    with pytest.raises(TypeError):

        @pytc.treeclass
        def f(x):
            return x


def test_registering_state():
    @pytc.treeclass
    class L0:
        def __init__(self):
            self.a = 10
            self.b = 20

    t = L0()
    tt = tree_copy(t)

    assert tt.a == 10
    assert tt.b == 20


def test_copy():
    @pytc.treeclass
    class L0:
        a: int = 1
        b: int = 3
        c: int = 5

    t = L0()

    assert copy.copy(t).a == 1
    assert copy.copy(t).b == 3
    assert copy.copy(t).c == 5
