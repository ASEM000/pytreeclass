from __future__ import annotations

from dataclasses import field

import jax
import jax.numpy as jnp
import jax.random as jr
import jax.tree_util as jtu
import numpy as np
import numpy.testing as npt
import pytest

import pytreeclass as pytc
from pytreeclass._src.tree_indexer import _at_apply, _at_get, _at_reduce, _at_set


@pytc.treeclass
class Test:
    a: float
    b: float
    c: float
    d: jnp.ndarray
    name: str = pytc.field(nondiff=True)


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

    assert pytc.is_treeclass_equal(B, C)

    B = A.at[(A > 0) & (A < 5)].get()
    C = level2(
        d=level1(a=1, b=None, c=jnp.array([1, 2, 3, 4])),
        e=level1(a=2, b=None, c=jnp.array([])),
    )

    assert pytc.is_treeclass_equal(B, C)

    B = A.at[A == 0].get()
    C = level2(
        d=level1(a=None, b=None, c=jnp.array([])),
        e=level1(a=None, b=None, c=jnp.array([])),
    )

    assert pytc.is_treeclass_equal(B, C)

    with pytest.raises(AssertionError):
        B = A.at[A].get()

    with pytest.raises(NotImplementedError):
        B = A.at[0].get()


def test_getter_by_param():
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
    assert pytc.is_treeclass_equal(lhs, rhs)

    with pytest.raises(AttributeError):
        t.at["s"].get()

    with pytest.raises(AttributeError):
        t.at["s"].apply(lambda _: 100)


def test_getter_by_metadata():
    @pytc.treeclass
    class Test:
        a: float = field(metadata={"name": "a"})
        b: float = field(metadata={"name": "b"})
        c: float = field(metadata={"name": "c"})
        d: jnp.ndarray = field(metadata={"name": "d"})
        name: str = pytc.field(nondiff=True)

    @pytc.treeclass
    class L0:
        a: int = field(default=1, metadata={"name": "a"})
        b: int = 2
        c: int = 3

    @pytc.treeclass
    class L1:
        a: int = field(default=1, metadata={"name": "a"})
        b: int = 2
        c: int = 3
        d: L0 = L0()

    @pytc.treeclass
    class L2:
        a: int = field(default=10, metadata={"name": "a"})
        b: int = 20
        c: int = 30
        d: L1 = L1()

    t = L2()
    lhs = t.at[t == {"name": "a"}].get()
    rhs = L2(10, None, None, L1(1, None, None, L0(1, None, None)))
    assert pytc.is_treeclass_equal(lhs, rhs)


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

    assert pytc.is_treeclass_equal(B, C)

    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), ("A"))

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
    assert pytc.is_treeclass_equal(lhs, rhs)


def test_setter_by_param():
    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), ("A"))

    B = A.at[A == "a"].set(0)
    assert pytc.is_treeclass_equal(
        B, Test(0, 20, 30, jnp.array([1, 2, 3, 4, 5]), ("A"))
    )

    B = A.at[(A == "a") | (A == "b")].set(0)
    assert pytc.is_treeclass_equal(B, Test(0, 0, 30, jnp.array([1, 2, 3, 4, 5]), ("A")))

    B = A.at[(A == "a") | (A == "b") | (A == "c")].set(0)
    assert pytc.is_treeclass_equal(B, Test(0, 0, 0, jnp.array([1, 2, 3, 4, 5]), ("A")))


def test_setter_by_metadata():
    @pytc.treeclass
    class Test:
        a: float = field(metadata={"name": "a"})
        b: float = field(metadata={"name": "b"})
        c: float = field(metadata={"name": "c"})
        d: jnp.ndarray = field(metadata={"name": "d"})
        name: str

    @pytc.treeclass
    class L0:
        a: int = field(default=1, metadata={"name": "a"})
        b: int = 2
        c: int = 3

    @pytc.treeclass
    class L1:
        a: int = field(default=1, metadata={"name": "a"})
        b: int = 2
        c: int = 3
        d: L0 = L0()

    @pytc.treeclass
    class L2:
        a: int = field(default=10, metadata={"name": "a"})
        b: int = 20
        c: int = 30
        d: L1 = L1()

    t = L2()
    lhs = t.at[t == {"name": "a"}].set(100)
    rhs = L2(100, 20, 30, L1(100, 2, 3, L0(100, 2, 3)))
    assert pytc.is_treeclass_equal(lhs, rhs)

    @pytc.treeclass
    class T:
        a: jnp.ndarray

    t = T(True)
    assert pytc.is_treeclass_equal(t.at[t == bool].set(False), T(a=False))


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
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(1, 4, jnp.array([1, 4, 9, 16, 25]))
    rhs = init.at[...].apply(lambda x: x**2)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[init == init].apply(lambda x: x + 1)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[...].apply(lambda x: x + 1)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(20, 30, jnp.array([20, 30, 40, 50, 60]))
    rhs = init.at[init == init].apply(lambda x: (x + 1) * 10)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(20, 30, jnp.array([20, 30, 40, 50, 60]))
    rhs = init.at[...].apply(lambda x: (x + 1) * 10)
    assert pytc.is_treeclass_equal(lhs, rhs)

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
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[init == init].apply(lambda x: x + 1)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[...].apply(lambda x: x + 1)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(0.5, 1.0, jnp.array([0.5, 1.0, 1.5, 2.0, 2.5]))
    rhs = init.at[init == init].apply(lambda x: x / 2.0)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(0.5, 1.0, jnp.array([0.5, 1.0, 1.5, 2.0, 2.5]))
    rhs = init.at[...].apply(lambda x: x / 2.0)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(1, 1, jnp.array([1, 1, 1, 1, 1]))
    rhs = init.at[init == init].apply(lambda x: jnp.minimum(x, 1))
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(4, 4, jnp.array([4, 4, 4, 4, 5]))
    rhs = init.at[init == init].apply(lambda x: jnp.maximum(x, 4))
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(4, 4, jnp.array([4, 4, 4, 4, 5]))
    rhs = init.at[...].apply(lambda x: jnp.maximum(x, 4))
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(1, 4, jnp.array([1, 4, 9, 16, 25]))
    rhs = init.at[init == init].apply(lambda x: x**2)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(1, 4, jnp.array([1, 4, 9, 16, 25]))
    rhs = init.at[...].apply(lambda x: x**2)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(1, 2, jnp.array([1, 2, 3, 16, 25]))
    rhs = init.at[init > 3].apply(lambda x: x**2)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[init > 0].apply(lambda x: x + 1)
    assert pytc.is_treeclass_equal(lhs, rhs)

    rhs = init.at[init > 100].apply(lambda x: (x + 1) * 10)
    assert pytc.is_treeclass_equal(init, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[init > 0].apply(lambda x: x + 1)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(1, 4, jnp.array([1, 4, 3, 4, 5]))
    rhs = init.at[init == 2].apply(lambda x: x * 2)
    assert pytc.is_treeclass_equal(lhs, rhs)

    # by param

    lhs = A(1, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at[init == "a"].apply(lambda x: x**2)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(2, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at[init == "a"].apply(lambda x: x + 1)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(20, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at[init == "a"].apply(lambda x: (x + 1) * 10)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(2, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at[init == "a"].apply(lambda x: x + 1)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(0.5, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at[init == "a"].apply(lambda x: x / 2.0)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(1, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at[init == "a"].apply(lambda x: jnp.minimum(x, 1))
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(4, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at[init == "a"].apply(lambda x: jnp.maximum(x, 4))
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(1, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at[init == "a"].apply(lambda x: x**2)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(2, 2, jnp.array([1, 2, 3, 4, 5]))
    rhs = init.at[init == "a"].apply(lambda x: x * 2)
    assert pytc.is_treeclass_equal(lhs, rhs)

    #
    lhs = A(1, 4, jnp.array([1, 4, 9, 16, 25]))
    rhs = init.at[init != "a"].apply(lambda x: x**2)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(1, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[init != "a"].apply(lambda x: x + 1)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(1, 30, jnp.array([20, 30, 40, 50, 60]))
    rhs = init.at[init != "a"].apply(lambda x: (x + 1) * 10)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(1, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[init != "a"].apply(lambda x: x + 1)
    assert pytc.is_treeclass_equal(lhs, rhs)

    lhs = A(1, 1, jnp.array([0.5, 1, 1.5, 2, 2.5]))
    rhs = init.at[init != "a"].apply(lambda x: x / 2.0)
    assert pytc.is_treeclass_equal(lhs, rhs)

    @pytc.treeclass
    class Test:
        a: float = field(metadata={"name": "a"})
        b: float = field(metadata={"name": "b"})
        c: float = field(metadata={"name": "c"})
        d: jnp.ndarray = field(metadata={"name": "d"})
        name: str = pytc.field(nondiff=True)

    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), ("A"))

    B = A.at[A != {"name": "a"}].apply(lambda _: 100)
    assert pytc.is_treeclass_equal(
        B, Test(10, 100, 100, jnp.array([100, 100, 100, 100, 100]), ("A"))
    )

    @pytc.treeclass
    class L0:
        a: int = field(default=1, metadata={"name": "a"})
        b: int = 2
        c: int = 3

    @pytc.treeclass
    class L1:
        a: int = field(default=1, metadata={"name": "a"})
        b: int = 2
        c: int = 3
        d: L0 = L0()

    @pytc.treeclass
    class L2:
        a: int = field(default=10, metadata={"name": "a"})
        b: int = 20
        c: int = 30
        d: L1 = L1()

    t = L2()
    lhs = t.at[t == {"name": "a"}].apply(lambda _: 100)
    rhs = L2(100, 20, 30, L1(100, 2, 3, L0(100, 2, 3)))
    assert pytc.is_treeclass_equal(lhs, rhs)

    t = L2()
    lhs = t.at[t != {"name": "a"}].apply(lambda _: 100)
    rhs = L2(10, 100, 100, L1(100, 100, 100, L0(100, 100, 100)))

    assert pytc.is_treeclass_equal(lhs, rhs)

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
        d: L1 = field(default=L1(), metadata={"name": "d"})

    t = L2()
    lhs = t.at[t == {"name": "d"}].apply(lambda _: 100)
    rhs = L2(10, 20, 30, L1(100, 100, 100, L0(100, 100, 100)))
    assert pytc.is_treeclass_equal(lhs, rhs)


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

    model = StackedLinear(in_dim=1, out_dim=1, hidden_dim=5, key=jax.random.PRNGKey(0))

    assert (
        model.at[model > 0].reduce(
            lambda x, y: jnp.minimum(x, jnp.min(y)), initializer=jnp.inf
        )
    ) == 0.98507565
    assert (
        model.at[model > 0].reduce(
            lambda x, y: jnp.maximum(x, jnp.max(y)), initializer=-jnp.inf
        )
    ) == 1.3969219
    assert (model.at[model > 0].reduce(lambda x, y: x + jnp.sum(y))) == 10.6970625
    assert (
        model.at[model > 0].reduce(lambda x, y: x * jnp.product(y), initializer=1)
    ) == 1.8088213

    assert (model.at[model == "l1"].reduce(lambda x, y: x + jnp.sum(y))) == 2.8428133
    assert (
        model.at[model == "l1"].reduce(lambda x, y: x * jnp.product(y), initializer=1)
    ) == -3.4602268

    assert (
        model.at[model == Linear].reduce(
            lambda x, y: jnp.minimum(x, jnp.min(y)), initializer=jnp.inf
        )
    ) == -2.8383057
    assert (
        model.at[model == Linear].reduce(
            lambda x, y: jnp.maximum(x, jnp.max(y)), initializer=-jnp.inf
        )
    ) == 1.3969219
    assert (model.at[model == Linear].reduce(lambda x, y: x + jnp.sum(y))) == 3.3538322
    assert (
        model.at[model == Linear].reduce(lambda x, y: x * jnp.product(y), initializer=1)
    ) == 0.84782064

    assert (
        model.at[model == jnp.ndarray].reduce(
            lambda x, y: jnp.minimum(x, jnp.min(y)), initializer=jnp.inf
        )
    ) == -2.8383057
    assert (
        model.at[model == jnp.ndarray].reduce(
            lambda x, y: jnp.maximum(x, jnp.max(y)), initializer=-jnp.inf
        )
    ) == 1.3969219
    assert (
        model.at[model == jnp.ndarray].reduce(lambda x, y: x + jnp.sum(y))
    ) == 3.3538322
    assert (
        model.at[model == jnp.ndarray].reduce(
            lambda x, y: x * jnp.product(y), initializer=1
        )
    ) == 0.84782064


def test_not_implemented():
    @pytc.treeclass
    class Test:
        a: int = 1

    t = Test()

    with pytest.raises(NotImplementedError):
        _at_get(t == t, 3, None)

    with pytest.raises(NotImplementedError):
        _at_set(t == t, 3, 3, None)

    with pytest.raises(NotImplementedError):
        _at_reduce(t == t, lambda x: x, 3, None, 0)

    with pytest.raises(NotImplementedError):
        _at_apply(t == t, lambda x: x, 3, None)

    with pytest.raises(NotImplementedError):
        _at_get(t == t, np.array([1, 2, 3]), None)

    with pytest.raises(NotImplementedError):
        _at_set(t == t, 3, np.array([1, 2, 3]), None)

    with pytest.raises(NotImplementedError):
        _at_reduce(t == t, lambda x: x, np.array([1, 2, 3]), None, 0)

    with pytest.raises(NotImplementedError):
        _at_apply(t == t, lambda x: x, np.array([1, 2, 3]), None)


def test_is_leaf():
    @pytc.treeclass
    class Test:
        a: int

    t = Test([1, 2, 3, None])

    assert pytc.is_treeclass_equal(
        t.at[...].set(10, is_leaf=lambda x: x is None), Test([10, 10, 10, 10])
    )

    assert pytc.is_treeclass_equal(
        t.at[...].apply(lambda x: 10, is_leaf=lambda x: x is None),
        Test([10, 10, 10, 10]),
    )


def test_masking():
    @pytc.treeclass
    class Linear:
        weight: jnp.ndarray
        bias: jnp.ndarray

        def __init__(self, key, in_dim, out_dim):
            self.weight = jax.random.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(
                2 / in_dim
            )
            self.bias = jnp.ones((1, out_dim))

        def __call__(self, x):
            return x @ self.weight + self.bias

    @pytc.treeclass
    class Dropout:
        p: float
        eval: bool | None

        def __init__(self, p: float = 0.5, eval: bool | None = None):
            """p : probability of an element to be zeroed out"""
            self.p = p
            self.eval = eval

        def __call__(self, x, *, key=jr.PRNGKey(0)):
            return (
                x
                if (self.eval is True)
                else jnp.where(
                    jr.bernoulli(key, (1 - self.p), x.shape), x / (1 - self.p), 0
                )
            )

    @pytc.treeclass
    class LinearWithDropout:
        def __init__(self):
            self.l1 = Linear(key=jr.PRNGKey(0), in_dim=1, out_dim=5)
            self.d1 = Dropout(p=1.0)  # zero out all elements

        def __call__(self, x):
            x = self.l1(x)
            x = self.d1(x)
            return x

    model = LinearWithDropout()
    npt.assert_allclose(model(jnp.ones((1, 1))), jnp.array([[0.0, 0.0, 0.0, 0.0, 0.0]]))

    mask = model == "eval"
    model_no_dropout = model.at[mask].set(True, is_leaf=lambda x: x is None)
    npt.assert_allclose(
        model_no_dropout(jnp.ones((1, 1))),
        jnp.array([[1.2656513, -0.8149204, 0.61661845, 2.7664368, 1.3457328]]),
    )


def test_attribute_get():
    @pytc.treeclass
    class l0:
        a: int = 2

    @pytc.treeclass
    class Test:
        a: int = 1
        b: l0 = l0()

    t = Test()
    assert t.at["a"].get() == 1
    assert t.at["b"].at["a"].get() == 2

    with pytest.raises(AttributeError):
        t.at["a"].at["c"].get()


def test_attribute_set():
    @pytc.treeclass
    class l0:
        a: int = 2

    @pytc.treeclass
    class Test:
        a: int = 1
        b: l0 = l0()

    t = Test()
    t.at["a"].set(10)

    assert pytc.is_treeclass_equal(t, Test())
    assert pytc.is_treeclass_equal(t.at["a"].set(10), Test(10, l0()))
    assert pytc.is_treeclass_equal(t.at["b"].at["a"].set(100), Test(1, l0(100)))

    with pytest.raises(AttributeError):
        t.at["c"].set(10)

    with pytest.raises(AttributeError):
        t.at["a"].at["c"].set(10)


def test_attribute_apply():
    @pytc.treeclass
    class l0:
        a: int = 2

    @pytc.treeclass
    class Test:
        a: int = 1
        b: l0 = l0()

    t = Test()
    t.at["a"].apply(lambda _: 10)

    assert pytc.is_treeclass_equal(t, Test())
    assert pytc.is_treeclass_equal(t.at["a"].apply(lambda _: 10), Test(10))
    assert pytc.is_treeclass_equal(
        t.at["b"].at["a"].apply(lambda _: 100), Test(1, l0(100))
    )

    with pytest.raises(AttributeError):
        t.at["c"].apply(lambda _: 10)


def test_method_call():
    @pytc.treeclass
    class Test:
        a: int = 1

        def increment(self):
            self.a += 1

    t = Test()

    assert pytc.is_treeclass_equal(t.at["increment"]()[1], Test(2))

    with pytest.raises(AttributeError):
        t.at["bla"]()

    with pytest.raises(TypeError):
        t.at["a"]()

    @pytc.treeclass
    class A:
        a: int

        def __call__(self, x):
            self.a += x
            return x

    a = A(1)
    _, b = a.at["__call__"](2)

    assert jtu.tree_leaves(a) == [1]
    assert jtu.tree_leaves(b) == [3]


def test_composed_at():
    @pytc.treeclass
    class Test:
        a: jnp.ndarray = jnp.array([1, 2, 3, 4, 5])

    t = Test()

    assert pytc.is_treeclass_equal(t.at[t > 0].at[t < 0].get(), Test(jnp.array([])))
    assert pytc.is_treeclass_equal(
        t.at[t > 0].at[t == "a"].get(), Test(jnp.array([1, 2, 3, 4, 5]))
    )

    with pytest.raises(AttributeError):
        t.at[t > 0].bet

    with pytest.raises(AttributeError):
        t.at["a"].bet


def test_repr_str():
    @pytc.treeclass
    class Test:
        a: jnp.ndarray = jnp.array([1, 2, 3, 4, 5])

    t = Test()

    assert f"{t.at[...]!r}" == "where=Test(a=bool[5])"
    assert f"{t.at[...]!s}" == "where=Test(a=[ True  True  True  True  True])"

    assert f"{t.at['a']!r}" == "where='a'"
    assert f"{t.at['a']!s}" == "where=a"


def test_not_equal():
    @pytc.treeclass
    class Test:
        a: int = 1
        b: float = 1.0

    t = Test()

    assert pytc.is_treeclass_equal(t.at[t == int].set(10), Test(a=10, b=1.0))

    assert pytc.is_treeclass_equal(t.at[t != int].set(10.0), Test(a=1, b=10.0))

    assert pytc.is_treeclass_equal(t.at[t != 10].set(10.0), Test(a=10.0, b=10.0))


def test_iterable_node():
    @pytc.treeclass
    class Test:
        a: int

    t = Test([1, 2, 3, 4])
    assert pytc.is_treeclass_equal(t.at[...].set(True), Test([True, True, True, True]))
    assert pytc.is_treeclass_equal(
        t.at[t == "a"].set(True), Test([True, True, True, True])
    )
    assert pytc.is_treeclass_equal(t.at[t != "a"].set(True), Test([1, 2, 3, 4]))
