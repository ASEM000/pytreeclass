from __future__ import annotations

from collections import namedtuple
from typing import Any

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy.testing as npt
import pytest

import pytreeclass as pytc
from pytreeclass import TreeClass
from pytreeclass._src.tree_indexer import _mutable_context


class Tree(TreeClass, leafwise=True):
    a: float
    b: float
    c: float
    d: jnp.ndarray
    name: str

    def __post_init__(self):
        self.name = pytc.freeze(self.name)


def test_getter_by_val():
    class level1(TreeClass, leafwise=True):
        a: int
        b: int
        c: int

    class level2(TreeClass, leafwise=True):
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

    assert pytc.is_tree_equal(B, C)

    B = A.at[(A > 0) & (A < 5)].get()
    C = level2(
        d=level1(a=1, b=None, c=jnp.array([1, 2, 3, 4])),
        e=level1(a=2, b=None, c=jnp.array([])),
    )

    assert pytc.is_tree_equal(B, C)

    B = A.at[A == 0].get()
    C = level2(
        d=level1(a=None, b=None, c=jnp.array([])),
        e=level1(a=None, b=None, c=jnp.array([])),
    )

    assert pytc.is_tree_equal(B, C)

    with pytest.raises(TypeError):
        B = A.at[A].get()

    # with pytest.raises(NotImplementedError):
    #     B = A.at[0].get()


def test_getter_by_param():
    class L0(TreeClass, leafwise=True):
        a: int = 1
        b: int = 2
        c: int = 3

    class L1(TreeClass, leafwise=True):
        a: int = 1
        b: int = 2
        c: int = 3
        d: L0 = L0()

    class L2(TreeClass, leafwise=True):
        a: int = 10
        b: int = 20
        c: int = 30
        d: L1 = L1()

    # t = L2()

    # with pytest.raises(AttributeError):
    #     t.at["s"].get()

    # with pytest.raises(AttributeError):
    #     t.at["s"].apply(lambda _: 100)


def test_setter_by_val():
    class level1(TreeClass, leafwise=True):
        a: int
        b: int
        c: int

    class level2(TreeClass, leafwise=True):
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

    assert pytc.is_tree_equal(B, C)

    A = Tree(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), ("A"))

    with pytest.raises(TypeError):
        B = A.at[A].set(0)

    # with pytest.raises(NotImplementedError):
    #     B = A.at[0].set(0)

    class L0(TreeClass, leafwise=True):
        a: int = 1
        b: int = 2
        c: int = 3

    class L1(TreeClass, leafwise=True):
        a: int = 1
        b: int = 2
        c: int = 3
        d: L0 = L0()

    class L2(TreeClass, leafwise=True):
        a: int = 10
        b: int = 20
        c: int = 30
        d: L1 = L1()

    t = L2()

    tt = L2(100, 200, 300, L1(10, 20, 30, L0(10, 20, 30)))
    lhs = t.at[t == t].set(tt)
    assert pytc.is_tree_equal(lhs, tt)


def test_apply_and_its_derivatives():
    class A(TreeClass, leafwise=True):
        a: int
        b: int
        c: jnp.ndarray

    init = A(1, 2, jnp.array([1, 2, 3, 4, 5]))

    # By boolean pytree
    lhs = A(1, 4, jnp.array([1, 4, 9, 16, 25]))
    rhs = init.at[init == init].apply(lambda x: x**2)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = A(1, 4, jnp.array([1, 4, 9, 16, 25]))
    rhs = init.at[...].apply(lambda x: x**2)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[init == init].apply(lambda x: x + 1)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[...].apply(lambda x: x + 1)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = A(20, 30, jnp.array([20, 30, 40, 50, 60]))
    rhs = init.at[init == init].apply(lambda x: (x + 1) * 10)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = A(20, 30, jnp.array([20, 30, 40, 50, 60]))
    rhs = init.at[...].apply(lambda x: (x + 1) * 10)
    assert pytc.is_tree_equal(lhs, rhs)

    with pytest.raises(TypeError):
        init.at[init].apply(lambda x: (x + 1) * 10)

    class L0(TreeClass, leafwise=True):
        a: int = 1
        b: int = 2
        c: int = 3

    class L1(TreeClass, leafwise=True):
        a: int = 1
        b: int = 2
        c: int = 3
        d: L0 = L0()

    class L2(TreeClass, leafwise=True):
        a: int = 10
        b: int = 20
        c: int = 30
        d: L1 = L1()

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[init == init].apply(lambda x: x + 1)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[...].apply(lambda x: x + 1)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = A(0.5, 1.0, jnp.array([0.5, 1.0, 1.5, 2.0, 2.5]))
    rhs = init.at[init == init].apply(lambda x: x / 2.0)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = A(0.5, 1.0, jnp.array([0.5, 1.0, 1.5, 2.0, 2.5]))
    rhs = init.at[...].apply(lambda x: x / 2.0)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = A(1, 1, jnp.array([1, 1, 1, 1, 1]))
    rhs = init.at[init == init].apply(lambda x: jnp.minimum(x, 1))
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = A(4, 4, jnp.array([4, 4, 4, 4, 5]))
    rhs = init.at[init == init].apply(lambda x: jnp.maximum(x, 4))
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = A(4, 4, jnp.array([4, 4, 4, 4, 5]))
    rhs = init.at[...].apply(lambda x: jnp.maximum(x, 4))
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = A(1, 4, jnp.array([1, 4, 9, 16, 25]))
    rhs = init.at[init == init].apply(lambda x: x**2)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = A(1, 4, jnp.array([1, 4, 9, 16, 25]))
    rhs = init.at[...].apply(lambda x: x**2)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = A(1, 2, jnp.array([1, 2, 3, 16, 25]))
    rhs = init.at[init > 3].apply(lambda x: x**2)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[init > 0].apply(lambda x: x + 1)
    assert pytc.is_tree_equal(lhs, rhs)

    rhs = init.at[init > 100].apply(lambda x: (x + 1) * 10)
    assert pytc.is_tree_equal(init, rhs)

    lhs = A(2, 3, jnp.array([2, 3, 4, 5, 6]))
    rhs = init.at[init > 0].apply(lambda x: x + 1)
    assert pytc.is_tree_equal(lhs, rhs)

    lhs = A(1, 4, jnp.array([1, 4, 3, 4, 5]))
    rhs = init.at[init == 2].apply(lambda x: x * 2)
    assert pytc.is_tree_equal(lhs, rhs)


def test_reduce():
    class A(TreeClass, leafwise=True):
        a: int
        b: int
        c: jnp.ndarray

    init = A(1, 2, jnp.array([1, 2, 3, 4, 5]))

    lhs = 2 + 2 + 3 + 4 + 5
    rhs = init.at[init > 1].reduce(lambda x, y: x + jnp.sum(y))
    assert lhs == rhs

    lhs = 3 + 4 + 5
    rhs = init.at[init > 2].reduce(lambda x, y: x + jnp.sum(y), initializer=0)
    assert lhs == rhs

    lhs = 0
    rhs = init.at[init > 100].reduce(lambda x, y: x + jnp.sum(y), initializer=0)
    assert lhs == rhs

    class B(TreeClass, leafwise=True):
        a: int
        b: int
        c: jnp.ndarray
        d: tuple

    init = B(1, 2, jnp.array([1, 2, 3, 4, 5]), (10, 20, 30))

    lhs = 2 + 2 + 3 + 4 + 5 + 10 + 20 + 30
    rhs = init.at[init > 1].reduce(lambda x, y: x + jnp.sum(y), initializer=0)
    assert lhs == rhs

    with pytest.raises(TypeError):
        init.at[init].reduce(lambda x, y: x + jnp.sum(y), initializer=0)

    class Tree(TreeClass, leafwise=True):
        a: tuple[int]

    lhs = Tree((1, 2, 3)).at["a"].reduce(lambda x, y: x + y, initializer=0)
    assert lhs == 6


def test_reduce_and_its_derivatives():
    class Linear(TreeClass, leafwise=True):
        weight: jnp.ndarray
        bias: jnp.ndarray

        def __init__(self, key, in_dim, out_dim):
            self.weight = jax.random.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(
                2 / in_dim
            )
            self.bias = jnp.ones((1, out_dim))

        # def __call__(self, x):
        #     return x @ self.weight + self.bias

    class StackedLinear(TreeClass, leafwise=True):
        l1: Linear
        l2: Linear

        def __init__(self, key, in_dim, out_dim, hidden_dim):
            keys = jax.random.split(key, 3)

            self.l1 = Linear(key=keys[0], in_dim=in_dim, out_dim=hidden_dim)
            self.l2 = Linear(key=keys[2], in_dim=hidden_dim, out_dim=out_dim)

    tree = StackedLinear(in_dim=1, out_dim=1, hidden_dim=5, key=jax.random.PRNGKey(0))

    assert (
        tree.at[tree > 0].reduce(
            lambda x, y: jnp.minimum(x, jnp.min(y)), initializer=jnp.inf
        )
    ) == 0.98507565
    npt.assert_allclose(
        tree.at[tree > 0].reduce(
            lambda x, y: jnp.maximum(x, jnp.max(y)), initializer=-jnp.inf
        ),
        1.3969219,
    )
    npt.assert_allclose(
        tree.at[tree > 0].reduce(lambda x, y: x + jnp.sum(y), initializer=0), 10.6970625
    )
    npt.assert_allclose(
        tree.at[tree > 0].reduce(lambda x, y: x * jnp.product(y), initializer=1),
        1.8088213,
    )


def test_is_leaf():
    class Tree(TreeClass, leafwise=True):
        a: int

    t = Tree([1, 2, 3, None])

    mask = jtu.tree_map(lambda x: True, t, is_leaf=lambda x: x is None)

    assert pytc.is_tree_equal(
        t.at[mask].set(10, is_leaf=lambda x: x is None), Tree([10, 10, 10, 10])
    )

    assert pytc.is_tree_equal(
        t.at[mask].apply(lambda x: 10, is_leaf=lambda x: x is None),
        Tree([10, 10, 10, 10]),
    )


def test_attribute_get():
    class l0(TreeClass, leafwise=True):
        a: int = 2

    class Tree(TreeClass, leafwise=True):
        a: int = 1
        b: l0 = l0()

    t = Tree()
    assert pytc.is_tree_equal(t.at["a"].get(), Tree(1, l0(None)))
    assert pytc.is_tree_equal(t.at["b"].at["a"].get(), Tree(None, l0(2)))


def test_attribute_set():
    class l0(TreeClass, leafwise=True):
        a: int = 2

    class Tree(TreeClass, leafwise=True):
        a: int = 1
        b: l0 = l0()

    t = Tree()
    t.at["a"].set(10)

    assert pytc.is_tree_equal(t, Tree())
    assert pytc.is_tree_equal(t.at["a"].set(10), Tree(10, l0()))
    assert pytc.is_tree_equal(t.at["b"].at["a"].set(100), Tree(1, l0(100)))


def test_attributre_apply():
    class l0(TreeClass, leafwise=True):
        a: int = 2

    class Tree(TreeClass, leafwise=True):
        a: int = 1
        b: l0 = l0()

    t = Tree()
    t.at["a"].apply(lambda _: 10)

    assert pytc.is_tree_equal(t, Tree())
    assert pytc.is_tree_equal(t.at["a"].apply(lambda _: 10), Tree(10))
    assert pytc.is_tree_equal(t.at["b"].at["a"].apply(lambda _: 100), Tree(1, l0(100)))


def test_trace_get():
    class l0(TreeClass, leafwise=True):
        a: int = 2

    class Tree(TreeClass, leafwise=True):
        a: int = 1
        b: l0 = l0()

    t = Tree()
    assert pytc.is_tree_equal(t.at[0].get(), Tree(1, l0(None)))
    assert pytc.is_tree_equal(t.at[1].at[0].get(), Tree(None, l0(2)))

    # with pytest.raises(IndexError):
    #     t.at[0].at[1].get()

    # with pytest.raises(IndexError):
    #     t.at[0].at[1].set(10)
    # with pytest.raises(IndexError):
    #     t.at[0].at[1].apply(lambda _: 10)
    # with pytest.raises(IndexError):
    #     t.at[0].at[1].reduce(lambda _, __: 10)


def test_trace_set():
    class l0(TreeClass, leafwise=True):
        a: int = 2

    class Tree(TreeClass, leafwise=True):
        a: int = 1
        b: l0 = l0()

    t = Tree()
    t.at["a"].set(10)

    assert pytc.is_tree_equal(t, Tree())
    assert pytc.is_tree_equal(t.at[0].set(10), Tree(10, l0()))
    assert pytc.is_tree_equal(t.at[1].at[0].set(100), Tree(1, l0(100)))


def test_trace_apply():
    class l0(TreeClass, leafwise=True):
        a: int = 2

    class Tree(TreeClass, leafwise=True):
        a: int = 1
        b: l0 = l0()

    t = Tree()
    t.at["a"].apply(lambda _: 10)

    assert pytc.is_tree_equal(t, Tree())
    assert pytc.is_tree_equal(t.at[0].apply(lambda _: 10), Tree(10))
    assert pytc.is_tree_equal(t.at[1].at[0].apply(lambda _: 100), Tree(1, l0(100)))


def test_trace_reduce():
    class A(TreeClass, leafwise=True):
        a: int
        b: int
        c: jnp.ndarray

    init = A(1, 2, jnp.array([1, 2, 3, 4, 5]))

    lhs = 1 + 2 + 3 + 4 + 5
    rhs = init.at[2].reduce(lambda x, y: x + jnp.sum(y), initializer=0)
    assert lhs == rhs


def test_mixed_get():
    class l0(TreeClass, leafwise=True):
        a: int = 2
        b: int = 1

    class Tree(TreeClass, leafwise=True):
        a: int = 1
        b: l0 = l0()

    t = Tree()
    assert pytc.is_tree_equal(t.at[1].at[t == 2].get(), Tree(None, l0(2, None)))
    assert pytc.is_tree_equal(t.at[t == 2].at[1].get(), Tree(None, l0(2, None)))

    # with pytest.raises(IndexError):
    #     t.at[0].at[2].get()


def test_mixed_set():
    class l0(TreeClass, leafwise=True):
        a: int = 2
        b: int = 1

    class Tree(TreeClass, leafwise=True):
        a: int = 1
        b: l0 = l0()

    t = Tree()

    assert pytc.is_tree_equal(t.at["b"].at[t == 2].set(100), Tree(1, l0(100)))
    assert pytc.is_tree_equal(t.at[t == 2].at["b"].set(100), Tree(1, l0(100)))
    assert pytc.is_tree_equal(t.at[1].at[t == 2].set(100), Tree(1, l0(100)))
    assert pytc.is_tree_equal(t.at[t == 2].at[1].set(100), Tree(1, l0(100)))
    assert pytc.is_tree_equal(t.at["b"].at[0].set(100), Tree(1, l0(100)))

    # with pytest.raises(IndexError):
    #     assert pytc.is_tree_equal(t.at[0].at["b"].set(100), Tree(1, l0(100, 2)))


def test_mixed_apply():
    class l0(TreeClass, leafwise=True):
        a: int = 2
        b: int = 1

    class Tree(TreeClass, leafwise=True):
        a: int = 1
        b: l0 = l0()

    t = Tree()

    assert pytc.is_tree_equal(t.at[1].at[t == 2].apply(lambda _: 100), Tree(1, l0(100)))
    assert pytc.is_tree_equal(t.at["b"].at[0].apply(lambda _: 100), Tree(1, l0(100)))

    # with pytest.raises(IndexError):
    #     t.at[0].at["a"].apply(lambda _: 100), Tree(1, l0(100))


def test_method_call():
    class Tree(TreeClass, leafwise=True):
        a: int = 1

        def increment(self):
            self.a += 1

        def show(self):
            return 1

    t = Tree()

    class Tree2(TreeClass):
        b: Tree = Tree()

    assert pytc.is_tree_equal(t.at["increment"]()[1], Tree(2))
    assert pytc.is_tree_equal(Tree2().at["b"].at["show"]()[0], 1)

    with pytest.raises(AttributeError):
        t.at["bla"]()

    with pytest.raises(TypeError):
        t.at["a"]()

    class A(TreeClass, leafwise=True):
        a: int

        def __call__(self, x):
            self.a += x
            return x

    a = A(1)
    _, b = a.at["__call__"](2)

    assert jtu.tree_leaves(a) == [1]
    assert jtu.tree_leaves(b) == [3]

    with pytest.raises(TypeError):
        a.at[0](1)


def test_composed_at():
    class Tree(TreeClass, leafwise=True):
        a: jnp.ndarray

        def __init__(self, a=jnp.array([1, 2, 3, 4, 5])) -> None:
            self.a = a

    t = Tree()

    assert pytc.is_tree_equal(t.at[t > 0].at[t < 0].get(), Tree(jnp.array([])))

    with pytest.raises(AttributeError):
        t.at[t > 0].bet

    with pytest.raises(AttributeError):
        t.at["a"].bet


def test_repr_str():
    class Tree(TreeClass, leafwise=True):
        a: int = 1
        b: int = 2

    t = Tree()

    assert repr(t.at["a"]) == "AtIndexer(tree=Tree(a=1, b=2), where=('a',))"
    assert str(t.at["a"]) == "AtIndexer(tree=Tree(a=1, b=2), where=('a',))"
    assert repr(t.at[...]) == "AtIndexer(tree=Tree(a=1, b=2), where=(Ellipsis,))"


def test_not_equal():
    class Tree(TreeClass, leafwise=True):
        a: int = 1
        b: float = 1.0

    t = Tree()

    assert pytc.is_tree_equal(t.at[t != 10].set(10.0), Tree(a=10.0, b=10.0))


def test_iterable_node():
    class Tree(TreeClass, leafwise=True):
        a: int

    t = Tree([1, 2, 3, 4])
    assert pytc.is_tree_equal(t.at[...].set(True), Tree([True, True, True, True]))


def test_call_context():
    class L2(TreeClass, leafwise=True):
        a: int = 1

        def delete(self, name):
            del self.a

    t = L2()

    with _mutable_context(t) as tx:
        tx.delete("a")

    with pytest.raises(AttributeError):
        t.delete("a")


def test_unsupported_indexing_type():
    class L2(TreeClass, leafwise=True):
        a: int = 1

        def delete(self, name):
            del self.a

    t = L2()

    with pytest.raises(NotImplementedError):
        t.at[None].set(1)


def test_mixed_not_implemented():
    class T(TreeClass):
        a: tuple[int, ...] = namedtuple("a", ["x", "y"])(1, 2)

    t = T()

    with pytest.raises(NotImplementedError):
        t.at["a"].at[[1]].get()

    with pytest.raises(NotImplementedError):
        t.at[0].at[[1]].get()


def test_nested_indexing():
    class Dict(dict):
        # test `FlattenedIndexKey`
        pass

    class Tree(pytc.TreeClass, leafwise=True):
        a: Any = (1, {"b": Dict({"c": 1})})

    tree = Tree()
    assert jtu.tree_leaves(tree.at["a"].at[1].at["b"].get())[0] == Dict({"c": 1})
    assert jtu.tree_leaves(tree.at[0].at[1].at["b"].get())[0] == Dict({"c": 1})
