from __future__ import annotations

import jax.numpy as jnp
import pytest

from pytreeclass import treeclass
from pytreeclass.src.tree_util import is_treeclass_equal


@treeclass
class Test:
    a: float
    b: float
    c: float
    d: jnp.ndarray
    name: str


def test_getter_by_param():
    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), "A")

    B = A.at["a"].get()
    assert is_treeclass_equal(B, Test(10, None, None, None, "A"))

    B = A.at["a", "b"].get()
    assert is_treeclass_equal(B, Test(10, 20, None, None, "A"))

    B = A.at[""].get()
    assert is_treeclass_equal(B, Test(None, None, None, None, "A"))

    B = A.at["a", "b", "c"].get()
    assert is_treeclass_equal(B, Test(10, 20, 30, None, "A"))

    B = A.at["a", "b", "c", "d"].get()
    assert is_treeclass_equal(B, Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), "A"))


def test_getter_by_slice():
    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), "A")

    B = A.at[0:1].get()
    assert is_treeclass_equal(B, Test(10, None, None, jnp.array([]), "A"))

    B = A.at[0:2].get()
    assert is_treeclass_equal(B, Test(10, 20, None, jnp.array([]), "A"))

    B = A.at[:-1].get()
    assert is_treeclass_equal(B, Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), "A"))

    B = A.at[:].get()
    assert is_treeclass_equal(B, Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), "A"))


def test_getter_by_int():
    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), "A")

    B = A.at[0].get()
    assert is_treeclass_equal(B, Test(10, None, None, None, "A"))

    B = A.at[1].get()
    assert is_treeclass_equal(B, Test(None, 20, None, None, "A"))

    B = A.at[2].get()
    assert is_treeclass_equal(B, Test(None, None, 30, None, "A"))

    B = A.at[3].get()
    assert is_treeclass_equal(
        B, Test(None, None, None, jnp.array([1, 2, 3, 4, 5]), "A")
    )


def test_getter_by_pytree():
    @treeclass
    class level1:
        a: int
        b: int
        c: int

    @treeclass
    class level2:
        d: level1
        e: level1

    A = level2(
        d=level1(a=1, b=10, c=jnp.array([1, 2, 3, 4, 5])),
        e=level1(a=2, b=20, c=jnp.array([-1, -2, -3, -4, -5])),
    )

    B = A.at[A > 0].get()
    C = level2(
        d=level1(a=1, b=10, c=jnp.array([1, 2, 3, 4, 5])),
        e=level1(a=2, b=20, c=jnp.array([])),
    )

    assert is_treeclass_equal(B, C)

    B = A.at[A == 0].get()
    C = level2(
        d=level1(a=None, b=None, c=jnp.array([])),
        e=level1(a=None, b=None, c=jnp.array([])),
    )

    assert is_treeclass_equal(B, C)

    with pytest.raises(ValueError):
        B = A.at[A].get()

    # with pytest.raises(NotImplementedError):
    B = A.at[0].get()


def test_setter_by_param():
    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), "A")

    B = A.at["a"].set(0)
    assert is_treeclass_equal(B, Test(0, 20, 30, jnp.array([1, 2, 3, 4, 5]), "A"))

    B = A.at["a", "b"].set(0)
    assert is_treeclass_equal(B, Test(0, 0, 30, jnp.array([1, 2, 3, 4, 5]), "A"))

    B = A.at["a", "b", "c"].set(0)
    assert is_treeclass_equal(B, Test(0, 0, 0, jnp.array([1, 2, 3, 4, 5]), "A"))


def test_setter_by_slice():
    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), "A")

    B = A.at[0:1].set(0)
    assert is_treeclass_equal(B, Test(0, 20, 30, jnp.array([1, 2, 3, 4, 5]), "A"))

    B = A.at[0:2].set(0)
    assert is_treeclass_equal(B, Test(0, 0, 30, jnp.array([1, 2, 3, 4, 5]), "A"))

    B = A.at[0:3].set(0)
    assert is_treeclass_equal(B, Test(0, 0, 0, jnp.array([1, 2, 3, 4, 5]), "A"))


def test_setter_by_int():
    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), "A")

    B = A.at[0].set(0)
    assert is_treeclass_equal(B, Test(0, 20, 30, jnp.array([1, 2, 3, 4, 5]), "A"))

    B = A.at[1].set(0)
    assert is_treeclass_equal(B, Test(10, 0, 30, jnp.array([1, 2, 3, 4, 5]), "A"))

    B = A.at[2].set(0)
    assert is_treeclass_equal(B, Test(10, 20, 0, jnp.array([1, 2, 3, 4, 5]), "A"))


def test_setter_by_pytree():
    @treeclass
    class level1:
        a: int
        b: int
        c: int

    @treeclass
    class level2:
        d: level1
        e: level1

    A = level2(
        d=level1(a=1, b=10, c=jnp.array([1, 2, 3, 4, 5])),
        e=level1(a=2, b=20, c=jnp.array([-1, -2, -3, -4, -5])),
    )

    B = A.at[A < 0].set(100)
    C = level2(
        d=level1(a=1, b=10, c=jnp.array([1, 2, 3, 4, 5])),
        e=level1(a=2, b=20, c=jnp.array([100, 100, 100, 100, 100])),
    )

    assert is_treeclass_equal(B, C)

    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), "A")

    with pytest.raises(ValueError):
        B = A.at[A].set(0)

    # with pytest.raises(NotImplementedError):
    B = A.at[0].set(0)


def test_apply_and_its_derivatives():
    @treeclass
    class A:
        a: int
        b: int
        c: jnp.ndarray

    init = A(1, 2, jnp.array([1, 2, 3, 4, 5]))

    # By boolean pytree
    lhs = A(1, 4, jnp.array([1, 4, 9, 16, 25]))
    rhs = init.at[init == init].apply(lambda x: x**2)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[init == init].apply(lambda x: x + 1)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(20, 30, jnp.array([20, 30, 40, 50, 60]))
    rhs = init.at[init == init].apply(lambda x: (x + 1) * 10)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[init == init].add(1)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(0.5, 1.0, jnp.array([0.5, 1.0, 1.5, 2.0, 2.5]))
    rhs = init.at[init == init].divide(2.0)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(1, 1, jnp.array([1, 1, 1, 1, 1]))
    rhs = init.at[init == init].min(1)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(4, 4, jnp.array([4, 4, 4, 4, 5]))
    rhs = init.at[init == init].max(4)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(1, 4, jnp.array([1, 4, 9, 16, 25]))
    rhs = init.at[init == init].power(2)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(1, 2, jnp.array([1, 2, 3, 16, 25]))
    rhs = init.at[init > 3].apply(lambda x: x**2)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[init > 0].apply(lambda x: x + 1)
    assert is_treeclass_equal(lhs, rhs)

    rhs = init.at[init > 100].apply(lambda x: (x + 1) * 10)
    assert is_treeclass_equal(init, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[init > 0].add(1)
    assert is_treeclass_equal(lhs, rhs)

    # by param

    lhs = A(1, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at["a"].apply(lambda x: x**2)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(2, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at["a"].apply(lambda x: x + 1)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(20, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at["a"].apply(lambda x: (x + 1) * 10)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(2, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at["a"].add(1)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(0.5, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at["a"].divide(2.0)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(1, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at["a"].min(1)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(4, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at["a"].max(4)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(1, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at["a"].power(2)
    assert is_treeclass_equal(lhs, rhs)

    # by param
    with pytest.raises(ValueError):
        init.freeze().at["a"].apply(lambda x: x**2)

    with pytest.raises(ValueError):
        init.freeze().at["a", "b"].apply(lambda x: x**2)

    # by slice
    with pytest.raises(ValueError):
        init.freeze().at[:].apply(lambda x: x**2)

    with pytest.raises(ValueError):
        init.freeze().at[0].apply(lambda x: x**2)

    # by pytree
    with pytest.raises(ValueError):
        init.freeze().at[init > 1].apply(lambda x: x**2)

    with pytest.raises(ValueError):
        init.freeze().at[init == 1].apply(lambda x: x**2)
