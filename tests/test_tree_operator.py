from __future__ import annotations

import functools as ft
import math

import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

import pytreeclass as pytc
from pytreeclass._src.tree_freeze import _hash_node


def test_bcmap():
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
    rhs = jtu.tree_map(lambda x: jnp.array(x), rhs)
    # test auto broadcasting
    lhs = pytc.bcmap(jnp.where)(tree > 1, 0, tree)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = pytc.bcmap(jnp.where)(tree > 1, 0, y=tree)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = pytc.bcmap(jnp.where)(tree > 1, x=0, y=tree)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = pytc.bcmap(jnp.where)(tree > 1, x=0, y=tree)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = pytc.bcmap(jnp.where)(condition=tree > 1, x=0, y=tree)
    assert pytc.is_tree_equal(lhs, rhs)

    with pytest.raises(ValueError):
        pytc.bcmap(lambda *x: x)(tree)

    with pytest.raises(ValueError):

        @pytc.bcmap
        def func(**k):
            return 0

    # test broadcasting with selected argnums/argnames
    lhs = ft.partial(pytc.bcmap, broadcast_argnums=(1,))(jnp.where)(tree > 1, 0, tree)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = ft.partial(pytc.bcmap, broadcast_argnames=("x",))(jnp.where)(
        tree > 1, x=0, y=tree
    )
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = ft.partial(pytc.bcmap, broadcast_argnums=(1,), broadcast_argnames=("x",))(
        jnp.where
    )(tree > 1, x=0, y=tree)
    assert pytc.is_tree_equal(lhs, rhs)


def test_math_operations():
    @pytc.treeclass
    class Test:
        a: float
        b: float
        c: float
        name: str

        def __post_init__(self):
            self.name = pytc.freeze(self.name)

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
        name: str

        def __post_init__(self):
            self.name = pytc.freeze(self.name)

    A = Test(-10, 20, ("A"))
    B = Test(10, 20, ("B"))
    C = Test(jnp.array([10]), jnp.array([20]), ("C"))

    # magic ops
    assert abs(A) == Test(10, 20, ("A"))
    assert A + A == Test(-20, 40, ("A"))
    assert A & A == Test(-10 & -10, 20 & 20, ("A"))
    assert math.ceil(A) == Test(math.ceil(-10), math.ceil(20), ("A"))
    assert divmod(A, A) == Test(divmod(-10, -10), divmod(20, 20), ("A"))
    assert (A == A) == Test(-10 == -10, 20 == 20, ("A"))
    assert math.floor(A) == Test(math.floor(-10), math.floor(20), ("A"))
    assert A // A == Test(-10 // (-10), 20 // (20), ("A"))
    assert A >= A == Test(-10 > -10, 20 > 20, ("A"))
    assert A > A == Test(-10 >= -10, 20 >= 20, ("A"))
    assert ~A == Test(~-10, ~20, ("A"))
    assert A <= A == Test(-10 <= -10, 20 <= 20, ("A"))
    assert A < A == Test(-10 < -10, 20 < 20, ("A"))

    assert B << B == Test(10 << 10, 20 << 20, ("B"))
    assert C @ C == Test(
        jnp.array([10]) @ jnp.array([10]), jnp.array([20]) @ jnp.array([20]), ("C")
    )

    with pytest.raises(TypeError):
        assert A @ A == Test(-10 @ -10, 20 @ 20, ("A"))

    assert A % A == Test(-10 % -10, 20 % 20, ("A"))
    assert A * A == Test(-10 * -10, 20 * 20, ("A"))
    assert -A == Test(-(-10), -20, ("A"))
    assert A | A == Test(-10 | -10, 20 | 20, ("A"))
    assert +A == Test(+(-10), +20, ("A"))
    assert A**A == Test(-(10**-10), 20**20, ("A"))
    assert round(A) == Test(round(-10), round(20), ("A"))
    assert A - A == Test(-10 + 10, 20 - 20, ("A"))
    assert A ^ A == Test(-10 ^ -10, 20 ^ 20, ("A"))


def test_math_operations_errors():
    @pytc.treeclass
    class Test:
        a: float
        b: float
        c: float
        name: str
        d: jnp.ndarray = None

        def __post_init__(self):
            self.name = pytc.freeze(self.name)
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
