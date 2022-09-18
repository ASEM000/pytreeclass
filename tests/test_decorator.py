import jax.tree_util as jtu
import pytest
from jax import numpy as jnp

import pytreeclass as pytc


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
