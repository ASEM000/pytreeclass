from __future__ import annotations

from dataclasses import field

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import pytreeclass as pytc
from pytreeclass.src.tree_indexer import _at_apply, _at_get, _at_reduce, _at_set
from pytreeclass.src.tree_util import is_treeclass_equal, static_value


@pytc.treeclass
class Test:
    a: float
    b: float
    c: float
    d: jnp.ndarray
    name: str


def test_getter_by_val():
    @pytc.treeclass
    class level1:
        a: int
        b: int
        c: int

    @pytc.treeclass
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

    B = A.at[(A > 0) & (A < 5)].get()
    C = level2(
        d=level1(a=1, b=None, c=jnp.array([1, 2, 3, 4])),
        e=level1(a=2, b=None, c=jnp.array([])),
    )

    assert is_treeclass_equal(B, C)

    B = A.at[A == 0].get()
    C = level2(
        d=level1(a=None, b=None, c=jnp.array([])),
        e=level1(a=None, b=None, c=jnp.array([])),
    )

    assert is_treeclass_equal(B, C)

    with pytest.raises(AssertionError):
        B = A.at[A].get()

    with pytest.raises(NotImplementedError):
        B = A.at[0].get()


def test_getter_by_param():

    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), static_value("A"))

    B = A.at[A == "a"].get(array_as_leaves=False)
    assert is_treeclass_equal(B, Test(10, None, None, None, static_value("A")))

    B = A.at[(A == "a") | (A == "b")].get(array_as_leaves=False)
    assert is_treeclass_equal(B, Test(10, 20, None, None, static_value("A")))

    B = A.at[A == ""].get(array_as_leaves=False)
    assert is_treeclass_equal(B, Test(None, None, None, None, static_value("A")))

    B = A.at[(A == "a") | (A == "b") | (A == "c")].get(array_as_leaves=False)
    assert is_treeclass_equal(B, Test(10, 20, 30, None, static_value("A")))

    B = A.at[(A == "a") | (A == "b") | (A == "c") | (A == "d")].get(
        array_as_leaves=False
    )
    assert is_treeclass_equal(
        B, Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), static_value("A"))
    )

    @pytc.treeclass
    class L0:
        a: int = 1
        b: int = 2
        c: int = 3

    @pytc.treeclass
    class L1:
        a: int = 1
        b: int = 2
        c: int = 3
        d: L0 = L0()

    @pytc.treeclass
    class L2:
        a: int = 10
        b: int = 20
        c: int = 30
        d: L1 = L1()

    t = L2()
    lhs = t.at[t == "a"].get()
    rhs = L2(10, None, None, L1(1, None, None, L0(1, None, None)))
    assert is_treeclass_equal(lhs, rhs)

    with pytest.raises(NotImplementedError):
        t.at["s"].get()

    with pytest.raises(NotImplementedError):
        t.at["s"].set(100)

    with pytest.raises(NotImplementedError):
        t.at["s"].apply(lambda _: 100)


def test_getter_by_metadata():
    @pytc.treeclass
    class Test:
        a: float = field(metadata={"name": "a", "unit": "m"})
        b: float = field(metadata={"name": "b", "unit": "m"})
        c: float = field(metadata={"name": "c", "unit": "m"})
        d: jnp.ndarray = field(metadata={"name": "d", "unit": "m"})
        name: str

    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), static_value("A"))

    B = A.at[A == {"name": "a"}].get(array_as_leaves=False)
    assert is_treeclass_equal(B, Test(10, None, None, None, static_value("A")))

    B = A.at[(A == {"name": "a"}) | (A == {"name": "b"})].get(array_as_leaves=False)
    assert is_treeclass_equal(B, Test(10, 20, None, None, static_value("A")))

    B = A.at[A == {"": ""}].get(array_as_leaves=False)
    assert is_treeclass_equal(B, Test(None, None, None, None, static_value("A")))

    B = A.at[(A == {"name": "a"}) | (A == {"name": "b"}) | (A == {"name": "c"})].get(
        array_as_leaves=False
    )
    assert is_treeclass_equal(B, Test(10, 20, 30, None, static_value("A")))

    B = A.at[
        (A == {"name": "a"})
        | (A == {"name": "b"})
        | (A == {"name": "c"})
        | (A == {"name": "d"})
    ].get(array_as_leaves=False)
    assert is_treeclass_equal(
        B, Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), static_value("A"))
    )

    @pytc.treeclass
    class L0:
        a: int = field(default=1, metadata={"name": "a", "unit": "m"})
        b: int = 2
        c: int = 3

    @pytc.treeclass
    class L1:
        a: int = field(default=1, metadata={"name": "a", "unit": "m"})
        b: int = 2
        c: int = 3
        d: L0 = L0()

    @pytc.treeclass
    class L2:
        a: int = field(default=10, metadata={"name": "a", "unit": "m"})
        b: int = 20
        c: int = 30
        d: L1 = L1()

    t = L2()
    lhs = t.at[t == {"name": "a"}].get()
    rhs = L2(10, None, None, L1(1, None, None, L0(1, None, None)))
    assert is_treeclass_equal(lhs, rhs)


def test_setter_by_val():
    @pytc.treeclass
    class level1:
        a: int
        b: int
        c: int

    @pytc.treeclass
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

    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), static_value("A"))

    with pytest.raises(AssertionError):
        B = A.at[A].set(0)

    with pytest.raises(NotImplementedError):
        B = A.at[0].set(0)

    @pytc.treeclass
    class L0:
        a: int = 1
        b: int = 2
        c: int = 3

    @pytc.treeclass
    class L1:
        a: int = 1
        b: int = 2
        c: int = 3
        d: L0 = L0()

    @pytc.treeclass
    class L2:
        a: int = 10
        b: int = 20
        c: int = 30
        d: L1 = L1()

    t = L2()
    lhs = t.at[t == "a"].set(100)
    rhs = L2(100, 20, 30, L1(100, 2, 3, L0(100, 2, 3)))
    assert is_treeclass_equal(lhs, rhs)


def test_setter_by_param():
    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), static_value("A"))

    B = A.at[A == "a"].set(0)
    assert is_treeclass_equal(
        B, Test(0, 20, 30, jnp.array([1, 2, 3, 4, 5]), static_value("A"))
    )

    B = A.at[(A == "a") | (A == "b")].set(0)
    assert is_treeclass_equal(
        B, Test(0, 0, 30, jnp.array([1, 2, 3, 4, 5]), static_value("A"))
    )

    B = A.at[(A == "a") | (A == "b") | (A == "c")].set(0)
    assert is_treeclass_equal(
        B, Test(0, 0, 0, jnp.array([1, 2, 3, 4, 5]), static_value("A"))
    )


def test_setter_by_metadata():
    @pytc.treeclass
    class Test:
        a: float = field(metadata={"name": "a", "unit": "m"})
        b: float = field(metadata={"name": "b", "unit": "m"})
        c: float = field(metadata={"name": "c", "unit": "m"})
        d: jnp.ndarray = field(metadata={"name": "d", "unit": "m"})
        name: str

    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), static_value("A"))

    B = A.at[A == {"name": "a"}].set(100, array_as_leaves=False)
    assert is_treeclass_equal(
        B, Test(100, 20, 30, jnp.array([1, 2, 3, 4, 5]), static_value("A"))
    )

    B = A.at[(A == {"name": "a"}) | (A == {"name": "b"})].set(
        100, array_as_leaves=False
    )
    assert is_treeclass_equal(
        B, Test(100, 100, 30, jnp.array([1, 2, 3, 4, 5]), static_value("A"))
    )

    B = A.at[A == {"": ""}].set(100, array_as_leaves=False)
    assert is_treeclass_equal(
        B, Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), static_value("A"))
    )

    B = A.at[(A == {"name": "a"}) | (A == {"name": "b"}) | (A == {"name": "c"})].set(
        100, array_as_leaves=False
    )
    assert is_treeclass_equal(
        B, Test(100, 100, 100, jnp.array([1, 2, 3, 4, 5]), static_value("A"))
    )

    B = A.at[
        (A == {"name": "a"})
        | (A == {"name": "b"})
        | (A == {"name": "c"})
        | (A == {"name": "d"})
    ].set(100, array_as_leaves=False)
    assert is_treeclass_equal(B, Test(100, 100, 100, 100, static_value("A")))

    @pytc.treeclass
    class L0:
        a: int = field(default=1, metadata={"name": "a", "unit": "m"})
        b: int = 2
        c: int = 3

    @pytc.treeclass
    class L1:
        a: int = field(default=1, metadata={"name": "a", "unit": "m"})
        b: int = 2
        c: int = 3
        d: L0 = L0()

    @pytc.treeclass
    class L2:
        a: int = field(default=10, metadata={"name": "a", "unit": "m"})
        b: int = 20
        c: int = 30
        d: L1 = L1()

    t = L2()
    lhs = t.at[t == {"name": "a"}].set(100)
    rhs = L2(100, 20, 30, L1(100, 2, 3, L0(100, 2, 3)))
    assert is_treeclass_equal(lhs, rhs)


def test_apply_and_its_derivatives():
    @pytc.treeclass
    class A:
        a: int
        b: int
        c: jnp.ndarray

    init = A(1, 2, jnp.array([1, 2, 3, 4, 5]))

    # By boolean pytree
    lhs = A(1, 4, jnp.array([1, 4, 9, 16, 25]))
    rhs = init.at[init == init].apply(lambda x: x**2)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(1, 4, jnp.array([1, 4, 9, 16, 25]))
    rhs = init.at[...].apply(lambda x: x**2)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[init == init].apply(lambda x: x + 1)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[...].apply(lambda x: x + 1)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(20, 30, jnp.array([20, 30, 40, 50, 60]))
    rhs = init.at[init == init].apply(lambda x: (x + 1) * 10)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(20, 30, jnp.array([20, 30, 40, 50, 60]))
    rhs = init.at[...].apply(lambda x: (x + 1) * 10)
    assert is_treeclass_equal(lhs, rhs)

    @pytc.treeclass
    class L0:
        a: int = 1
        b: int = 2
        c: int = 3

    @pytc.treeclass
    class L1:
        a: int = 1
        b: int = 2
        c: int = 3
        d: L0 = L0()

    @pytc.treeclass
    class L2:
        a: int = 10
        b: int = 20
        c: int = 30
        d: L1 = L1()

    t = L2()
    lhs = t.at[t == "a"].apply(lambda _: 100)
    rhs = L2(100, 20, 30, L1(100, 2, 3, L0(100, 2, 3)))
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[init == init].add(1)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[...].add(1)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(0.5, 1.0, jnp.array([0.5, 1.0, 1.5, 2.0, 2.5]))
    rhs = init.at[init == init].divide(2.0)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(0.5, 1.0, jnp.array([0.5, 1.0, 1.5, 2.0, 2.5]))
    rhs = init.at[...].divide(2.0)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(1, 1, jnp.array([1, 1, 1, 1, 1]))
    rhs = init.at[init == init].min(1)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(4, 4, jnp.array([4, 4, 4, 4, 5]))
    rhs = init.at[init == init].max(4)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(4, 4, jnp.array([4, 4, 4, 4, 5]))
    rhs = init.at[...].max(4)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(1, 4, jnp.array([1, 4, 9, 16, 25]))
    rhs = init.at[init == init].power(2)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(1, 4, jnp.array([1, 4, 9, 16, 25]))
    rhs = init.at[...].power(2)
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

    lhs = A(1, 4, jnp.array([1, 4, 3, 4, 5]))
    rhs = init.at[init == 2].multiply(2)
    assert is_treeclass_equal(lhs, rhs)

    # by param

    lhs = A(1, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at[init == "a"].apply(lambda x: x**2)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(2, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at[init == "a"].apply(lambda x: x + 1)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(20, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at[init == "a"].apply(lambda x: (x + 1) * 10)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(2, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at[init == "a"].add(1)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(0.5, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at[init == "a"].divide(2.0)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(1, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at[init == "a"].min(1)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(4, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at[init == "a"].max(4)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(1, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at[init == "a"].power(2)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(2, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at[init == "a"].multiply(2)
    assert is_treeclass_equal(lhs, rhs)

    #
    lhs = A(1, 4, jnp.array([1, 4, 9, 16, 25]))
    rhs = init.at[init != "a"].apply(lambda x: x**2)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(1, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[init != "a"].apply(lambda x: x + 1)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(1, 30, jnp.array([20, 30, 40, 50, 60]))
    rhs = init.at[init != "a"].apply(lambda x: (x + 1) * 10)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(1, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[init != "a"].add(1)
    assert is_treeclass_equal(lhs, rhs)

    lhs = A(1, 1, jnp.array([0.5, 1, 1.5, 2, 2.5]))
    rhs = init.at[init != "a"].divide(2.0)
    assert is_treeclass_equal(lhs, rhs)

    # raise
    with pytest.raises(ValueError):
        init.freeze().at[init == "a"].apply(lambda x: x**2)

    with pytest.raises(ValueError):
        init.freeze().at[(init == "a") | (A == "b")].apply(lambda x: x**2)

    with pytest.raises(ValueError):
        init.freeze().at[(init == "a") | (A == "b")].set(0)

    with pytest.raises(ValueError):
        init.freeze().at[(init == init) | (A == "b")].set(0)

    with pytest.raises(ValueError):
        init.freeze().at[(init == init) | (A == "b")].reduce_sum()

    @pytc.treeclass
    class Test:
        a: float = field(metadata={"name": "a", "unit": "m"})
        b: float = field(metadata={"name": "b", "unit": "m"})
        c: float = field(metadata={"name": "c", "unit": "m"})
        d: jnp.ndarray = field(metadata={"name": "d", "unit": "m"})
        name: str

    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), static_value("A"))

    B = A.at[A == {"name": "a"}].apply(lambda _: 100, array_as_leaves=False)
    assert is_treeclass_equal(
        B, Test(100, 20, 30, jnp.array([1, 2, 3, 4, 5]), static_value("A"))
    )

    B = A.at[(A == {"name": "a"}) | (A == {"name": "b"})].set(
        100, array_as_leaves=False
    )
    assert is_treeclass_equal(
        B, Test(100, 100, 30, jnp.array([1, 2, 3, 4, 5]), static_value("A"))
    )

    B = A.at[A == {"": ""}].apply(lambda _: 100, array_as_leaves=False)
    assert is_treeclass_equal(
        B, Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), static_value("A"))
    )

    B = A.at[(A == {"name": "a"}) | (A == {"name": "b"}) | (A == {"name": "c"})].set(
        100, array_as_leaves=False
    )
    assert is_treeclass_equal(
        B, Test(100, 100, 100, jnp.array([1, 2, 3, 4, 5]), static_value("A"))
    )

    B = A.at[
        (A == {"name": "a"})
        | (A == {"name": "b"})
        | (A == {"name": "c"})
        | (A == {"name": "d"})
    ].apply(lambda _: 100, array_as_leaves=False)
    assert is_treeclass_equal(B, Test(100, 100, 100, 100, static_value("A")))

    B = A.at[A != {"name": "a"}].apply(lambda _: 100)
    assert is_treeclass_equal(
        B, Test(10, 100, 100, jnp.array([100, 100, 100, 100, 100]), static_value("A"))
    )

    @pytc.treeclass
    class L0:
        a: int = field(default=1, metadata={"name": "a", "unit": "m"})
        b: int = 2
        c: int = 3

    @pytc.treeclass
    class L1:
        a: int = field(default=1, metadata={"name": "a", "unit": "m"})
        b: int = 2
        c: int = 3
        d: L0 = L0()

    @pytc.treeclass
    class L2:
        a: int = field(default=10, metadata={"name": "a", "unit": "m"})
        b: int = 20
        c: int = 30
        d: L1 = L1()

    t = L2()
    lhs = t.at[t == {"name": "a"}].apply(lambda _: 100)
    rhs = L2(100, 20, 30, L1(100, 2, 3, L0(100, 2, 3)))
    assert is_treeclass_equal(lhs, rhs)

    t = L2()
    lhs = t.at[t != {"name": "a"}].apply(lambda _: 100)
    rhs = L2(10, 100, 100, L1(1, 100, 100, L0(1, 100, 100)))

    assert is_treeclass_equal(lhs, rhs)

    @pytc.treeclass
    class L0:
        a: int = 1
        b: int = 2
        c: int = 3

    @pytc.treeclass
    class L1:
        a: int = 1
        b: int = 2
        c: int = 3
        d: L0 = L0()

    @pytc.treeclass
    class L2:
        a: int = 10
        b: int = 20
        c: int = 30
        d: L1 = field(default=L1(), metadata={"name": "d", "unit": "m"})

    t = L2()
    lhs = t.at[t == {"name": "d"}].apply(lambda _: 100)
    rhs = L2(10, 20, 30, L1(100, 100, 100, L0(100, 100, 100)))
    assert is_treeclass_equal(lhs, rhs)


def test_reduce():
    @pytc.treeclass
    class A:
        a: int
        b: int
        c: jnp.ndarray

    init = A(1, 2, jnp.array([1, 2, 3, 4, 5]))

    lhs = 2 + 2 + 3 + 4 + 5
    rhs = init.at[init > 1].reduce(lambda x, y: x + jnp.sum(y))
    assert lhs == rhs

    lhs = 3 + 4 + 5
    rhs = init.at[init > 2].reduce(lambda x, y: x + jnp.sum(y))
    assert lhs == rhs

    lhs = 0
    rhs = init.at[init > 100].reduce(lambda x, y: x + jnp.sum(y))
    assert lhs == rhs

    @pytc.treeclass
    class B:
        a: int
        b: int
        c: jnp.ndarray
        d: tuple

    init = B(1, 2, jnp.array([1, 2, 3, 4, 5]), (10, 20, 30))

    lhs = 2 + 2 + 3 + 4 + 5 + 10 + 20 + 30
    rhs = init.at[init > 1].reduce(lambda x, y: x + jnp.sum(y))
    assert lhs == rhs

    with pytest.raises(AssertionError):
        init.at[init].reduce(lambda x, y: x + jnp.sum(y))


def test_reduce_and_its_derivatives():
    @pytc.treeclass
    class Linear:
        weight: jnp.ndarray
        bias: jnp.ndarray

        def __init__(self, key, in_dim, out_dim):
            self.weight = jax.random.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(
                2 / in_dim
            )
            self.bias = jnp.ones((1, out_dim))

        # def __call__(self, x):
        #     return x @ self.weight + self.bias

    @pytc.treeclass
    class StackedLinear:
        l1: Linear = field(metadata={"description": "First layer"})
        l2: Linear = field(metadata={"description": "Second layer"})

        def __init__(self, key, in_dim, out_dim, hidden_dim):
            keys = jax.random.split(key, 3)

            self.l1 = Linear(key=keys[0], in_dim=in_dim, out_dim=hidden_dim)
            self.l2 = Linear(key=keys[2], in_dim=hidden_dim, out_dim=out_dim)

        # def __call__(self, x):
        #     x = self.l1(x)
        #     x = jax.nn.tanh(x)
        #     x = self.l2(x)

        #     return x

    model = StackedLinear(in_dim=1, out_dim=1, hidden_dim=5, key=jax.random.PRNGKey(0))

    assert (model.at[model > 0].reduce_min()) == 0.98507565
    assert (model.at[model > 0].reduce_max()) == 1.3969219
    assert (model.at[model > 0].reduce_sum()) == 10.6970625
    assert (model.at[model > 0].reduce_product()) == 1.8088213

    assert (model.at[model == "l1"].reduce_sum()) == 2.8428133
    assert (model.at[model == "l1"].reduce_product()) == -3.4602268

    assert (model.at[model == Linear].reduce_min()) == -2.8383057
    assert (model.at[model == Linear].reduce_max()) == 1.3969219
    assert (model.at[model == Linear].reduce_sum()) == 3.3538322
    assert (model.at[model == Linear].reduce_product()) == 0.84782064

    assert (model.at[model == jnp.ndarray].reduce_min()) == -2.8383057
    assert (model.at[model == jnp.ndarray].reduce_max()) == 1.3969219
    assert (model.at[model == jnp.ndarray].reduce_sum()) == 3.3538322
    assert (model.at[model == jnp.ndarray].reduce_product()) == 0.84782064


def test_not_implemented():
    @pytc.treeclass
    class Test:
        a: int = 1

    t = Test()

    with pytest.raises(NotImplementedError):
        _at_get(t == t, 3)

    with pytest.raises(NotImplementedError):
        _at_set(t == t, 3, 3)

    with pytest.raises(NotImplementedError):
        _at_reduce(t == t, lambda x: x, 3)

    with pytest.raises(NotImplementedError):
        _at_apply(t == t, lambda x: x, 3)

    with pytest.raises(NotImplementedError):
        _at_get(t == t, np.array([1, 2, 3]))

    with pytest.raises(NotImplementedError):
        _at_set(t == t, 3, np.array([1, 2, 3]))

    with pytest.raises(NotImplementedError):
        _at_reduce(t == t, lambda x: x, np.array([1, 2, 3]))

    with pytest.raises(NotImplementedError):
        _at_apply(t == t, lambda x: x, np.array([1, 2, 3]))
