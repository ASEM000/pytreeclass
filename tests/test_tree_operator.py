from __future__ import annotations

import jax.numpy as jnp
import pytest

import pytreeclass as pytc
from pytreeclass._src.tree_freeze import _hash_node


def test_bmap():
    @pytc.treeclass
    class Test:
        a: tuple[int]
        b: tuple[int]
        c: jnp.ndarray
        d: int

        def __init__(self, a=(1, 2, 3), b=(4, 5, 6), c=jnp.array([1, 2, 3]), d=1):
            self.a = a
            self.b = b
            self.c = c
            self.d = d

    tree = Test()
    rhs = Test(a=(1, 0, 0), b=(0, 0, 0), c=jnp.array([1, 0, 0]), d=1)

    lhs = pytc.bmap(jnp.where)(tree > 1, 0, tree)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = pytc.bmap(jnp.where)(tree > 1, 0, y=tree)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = pytc.bmap(jnp.where)(tree > 1, x=0, y=tree)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = pytc.bmap(jnp.where)(tree > 1, x=0, y=tree)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = pytc.bmap(jnp.where)(condition=tree > 1, x=0, y=tree)
    assert pytc.is_tree_equal(lhs, rhs)

    with pytest.raises(ValueError):
        pytc.bmap(lambda *x: x)(tree)

    with pytest.raises(ValueError):

        @pytc.bmap
        def func(**k):
            return 0


def test_math_operations():
    @pytc.treeclass
    class Test:
        a: float
        b: float
        c: float
        name: str = pytc.field(nondiff=True)

    A = Test(10, 20, 30, ("A"))
    # binary operations

    assert (A + A) == Test(20, 40, 60, ("A"))
    assert (A - A) == Test(0, 0, 0, ("A"))
    # assert ((A["a"] + A) | A) == Test(20, 20, 30, ("A"))
    assert A.at[...].reduce(lambda x, y: x + jnp.sum(y)) == jnp.array(60)
    assert abs(A) == A

    @pytc.treeclass
    class Test:
        a: float
        b: float
        name: str = pytc.field(nondiff=True)

    A = Test(-10, 20, ("A"))

    # magic ops
    assert abs(A) == Test(10, 20, ("A"))
    assert A + A == Test(-20, 40, ("A"))
    assert A == A
    assert A // 2 == Test(-5, 10, ("A"))
    assert A / 2 == Test(-5.0, 10.0, ("A"))
    assert (A > A) == Test(False, False, ("A"))
    assert (A >= A) == Test(True, True, ("A"))
    assert (A <= A) == Test(True, True, ("A"))
    assert -A == Test(10, -20, ("A"))
    assert A * A == Test(100, 400, ("A"))
    assert A**A == Test((-10) ** (-10), 20**20, ("A"))
    assert A - A == Test(0, 0, ("A"))

    # unary operations
    assert abs(A) == Test(10, 20, ("A"))
    assert -A == Test(-10, -20, ("A"))
    assert +A == Test(10, 20, ("A"))
    assert ~A == Test(~10, ~20, ("A"))


def test_math_operations_errors():
    @pytc.treeclass
    class Test:
        a: float
        b: float
        c: float
        name: str = pytc.field(nondiff=True)
        d: jnp.ndarray = None

        def __post_init__(self):
            self.d = jnp.array([1, 2, 3])

    A = Test(10, 20, 30, ("A"))

    with pytest.raises(TypeError):
        A + "s"

    with pytest.raises(TypeError):
        A == (1,)


def test_hash_node():
    assert _hash_node([1, 2, 3])
    assert _hash_node(jnp.array([1, 2, 3]))
    assert _hash_node({1, 2, 3})
    assert _hash_node({"a": 1})
