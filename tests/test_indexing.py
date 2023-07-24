# Copyright 2023 PyTreeClass authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

import re
from collections import namedtuple
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy.testing as npt
import pytest

import pytreeclass as pytc
from pytreeclass import TreeClass
from pytreeclass._src.tree_base import _mutable_context
from pytreeclass._src.tree_util import construct_tree


@pytc.leafwise
@pytc.autoinit
class Tree(TreeClass):
    a: float
    b: float
    c: float
    d: jnp.ndarray
    name: str

    def __post_init__(self):
        self.name = pytc.freeze(self.name)


def test_getter_by_val():
    @pytc.leafwise
    @pytc.autoinit
    class level1(TreeClass):
        a: int
        b: int
        c: int

    @pytc.leafwise
    @pytc.autoinit
    class level2(TreeClass):
        d: level1
        e: level1

    A = level2(
        d=level1(a=1, b=10, c=jnp.array([1, 2, 3, 4, 5])),
        e=level1(a=2, b=20, c=jnp.array([-1, -2, -3, -4, -5])),
    )

    B = A.at[A > 0].get()
    C = level2(
        d=level1(a=1, b=10, c=jnp.array([1, 2, 3, 4, 5])),
        e=level1(a=2, b=20, c=jnp.array([], dtype=int)),
    )

    assert pytc.is_tree_equal(B, C)

    B = A.at[(A > 0) & (A < 5)].get()
    C = level2(
        d=level1(a=1, b=None, c=jnp.array([1, 2, 3, 4])),
        e=level1(a=2, b=None, c=jnp.array([], dtype=int)),
    )

    assert pytc.is_tree_equal(B, C)

    B = A.at[A == 0].get()
    C = level2(
        d=level1(a=None, b=None, c=jnp.array([], dtype=int)),
        e=level1(a=None, b=None, c=jnp.array([], dtype=int)),
    )

    assert pytc.is_tree_equal(B, C)

    with pytest.raises(NotImplementedError):
        B = A.at[A].get()

    # with pytest.raises(NotImplementedError):
    #     B = A.at[0].get()


def test_getter_by_param():
    @pytc.leafwise
    @pytc.autoinit
    class L0(TreeClass):
        a: int = 1
        b: int = 2
        c: int = 3

    @pytc.leafwise
    @pytc.autoinit
    class L1(TreeClass):
        a: int = 1
        b: int = 2
        c: int = 3
        d: L0 = L0()

    @pytc.leafwise
    @pytc.autoinit
    class L2(TreeClass):
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
    @pytc.leafwise
    @pytc.autoinit
    class level1(TreeClass):
        a: int
        b: int
        c: int

    @pytc.leafwise
    @pytc.autoinit
    class level2(TreeClass):
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

    with pytest.raises(NotImplementedError):
        B = A.at[A].set(0)

    # with pytest.raises(NotImplementedError):
    #     B = A.at[0].set(0)
    @pytc.leafwise
    @pytc.autoinit
    class L0(TreeClass):
        a: int = 1
        b: int = 2
        c: int = 3

    @pytc.leafwise
    @pytc.autoinit
    class L1(TreeClass):
        a: int = 1
        b: int = 2
        c: int = 3
        d: L0 = L0()

    @pytc.leafwise
    @pytc.autoinit
    class L2(TreeClass):
        a: int = 10
        b: int = 20
        c: int = 30
        d: L1 = L1()

    t = L2()

    tt = L2(100, 200, 300, L1(10, 20, 30, L0(10, 20, 30)))
    lhs = t.at[t == t].set(tt)
    assert pytc.is_tree_equal(lhs, tt)


def test_apply_and_its_derivatives():
    @pytc.leafwise
    @pytc.autoinit
    class A(TreeClass):
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

    with pytest.raises(NotImplementedError):
        init.at[init].apply(lambda x: (x + 1) * 10)

    @pytc.leafwise
    @pytc.autoinit
    class L0(TreeClass):
        a: int = 1
        b: int = 2
        c: int = 3

    @pytc.leafwise
    @pytc.autoinit
    class L1(TreeClass):
        a: int = 1
        b: int = 2
        c: int = 3
        d: L0 = L0()

    @pytc.leafwise
    @pytc.autoinit
    class L2(TreeClass):
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
    @pytc.leafwise
    @pytc.autoinit
    class A(TreeClass):
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

    @pytc.leafwise
    @pytc.autoinit
    class B(TreeClass):
        a: int
        b: int
        c: jnp.ndarray
        d: tuple

    init = B(1, 2, jnp.array([1, 2, 3, 4, 5]), (10, 20, 30))

    lhs = 2 + 2 + 3 + 4 + 5 + 10 + 20 + 30
    rhs = init.at[init > 1].reduce(lambda x, y: x + jnp.sum(y), initializer=0)
    assert lhs == rhs

    with pytest.raises(NotImplementedError):
        init.at[init].reduce(lambda x, y: x + jnp.sum(y), initializer=0)

    @pytc.leafwise
    @pytc.autoinit
    class Tree(TreeClass):
        a: tuple[int]

    lhs = Tree((1, 2, 3)).at["a"].reduce(lambda x, y: x + y, initializer=0)
    assert lhs == 6


def test_reduce_and_its_derivatives():
    @pytc.leafwise
    class Linear(TreeClass):
        def __init__(self, key, in_dim, out_dim):
            self.weight = jax.random.normal(key, shape=(in_dim, out_dim)) * jnp.sqrt(
                2 / in_dim
            )
            self.bias = jnp.ones((1, out_dim))

        # def __call__(self, x):
        #     return x @ self.weight + self.bias

    @pytc.leafwise
    class StackedLinear(TreeClass):
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
        tree.at[tree > 0].reduce(lambda x, y: x * jnp.prod(y), initializer=1),
        1.8088213,
    )


def test_is_leaf():
    @pytc.leafwise
    @pytc.autoinit
    class Tree(TreeClass):
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
    @pytc.leafwise
    @pytc.autoinit
    class l0(TreeClass):
        a: int = 2

    @pytc.leafwise
    @pytc.autoinit
    class Tree(TreeClass):
        a: int = 1
        b: l0 = l0()

    t = Tree()
    assert pytc.is_tree_equal(t.at["a"].get(), Tree(1, l0(None)))
    assert pytc.is_tree_equal(t.at["b"].at["a"].get(), Tree(None, l0(2)))


def test_attribute_set():
    @pytc.leafwise
    @pytc.autoinit
    class l0(TreeClass):
        a: int = 2

    @pytc.leafwise
    @pytc.autoinit
    class Tree(TreeClass):
        a: int = 1
        b: l0 = l0()

    t = Tree()
    t.at["a"].set(10)

    assert pytc.is_tree_equal(t, Tree())
    assert pytc.is_tree_equal(t.at["a"].set(10), Tree(10, l0()))
    assert pytc.is_tree_equal(t.at["b"].at["a"].set(100), Tree(1, l0(100)))


def test_attributre_apply():
    @pytc.leafwise
    @pytc.autoinit
    class l0(TreeClass):
        a: int = 2

    @pytc.leafwise
    @pytc.autoinit
    class Tree(TreeClass):
        a: int = 1
        b: l0 = l0()

    t = Tree()
    t.at["a"].apply(lambda _: 10)

    assert pytc.is_tree_equal(t, Tree())
    assert pytc.is_tree_equal(t.at["a"].apply(lambda _: 10), Tree(10))
    assert pytc.is_tree_equal(t.at["b"].at["a"].apply(lambda _: 100), Tree(1, l0(100)))


def test_trace_get():
    @pytc.leafwise
    @pytc.autoinit
    class l0(TreeClass):
        a: int = 2

    @pytc.leafwise
    @pytc.autoinit
    class Tree(TreeClass):
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
    @pytc.leafwise
    @pytc.autoinit
    class l0(TreeClass):
        a: int = 2

    @pytc.leafwise
    @pytc.autoinit
    class Tree(TreeClass):
        a: int = 1
        b: l0 = l0()

    t = Tree()
    t.at["a"].set(10)

    assert pytc.is_tree_equal(t, Tree())
    assert pytc.is_tree_equal(t.at[0].set(10), Tree(10, l0()))
    assert pytc.is_tree_equal(t.at[1].at[0].set(100), Tree(1, l0(100)))


def test_trace_apply():
    @pytc.leafwise
    @pytc.autoinit
    class l0(TreeClass):
        a: int = 2

    @pytc.leafwise
    @pytc.autoinit
    class Tree(TreeClass):
        a: int = 1
        b: l0 = l0()

    t = Tree()
    t.at["a"].apply(lambda _: 10)

    assert pytc.is_tree_equal(t, Tree())
    assert pytc.is_tree_equal(t.at[0].apply(lambda _: 10), Tree(10))
    assert pytc.is_tree_equal(t.at[1].at[0].apply(lambda _: 100), Tree(1, l0(100)))


def test_trace_reduce():
    @pytc.leafwise
    @pytc.autoinit
    class A(TreeClass):
        a: int
        b: int
        c: jnp.ndarray

    init = A(1, 2, jnp.array([1, 2, 3, 4, 5]))

    lhs = 1 + 2 + 3 + 4 + 5
    rhs = init.at[2].reduce(lambda x, y: x + jnp.sum(y), initializer=0)
    assert lhs == rhs


def test_mixed_get():
    @pytc.leafwise
    @pytc.autoinit
    class l0(TreeClass):
        a: int = 2
        b: int = 1

    @pytc.leafwise
    @pytc.autoinit
    class Tree(TreeClass):
        a: int = 1
        b: l0 = l0()

    t = Tree()
    assert pytc.is_tree_equal(t.at[1].at[t == 2].get(), Tree(None, l0(2, None)))
    assert pytc.is_tree_equal(t.at[t == 2].at[1].get(), Tree(None, l0(2, None)))

    # with pytest.raises(IndexError):
    #     t.at[0].at[2].get()


def test_mixed_set():
    @pytc.leafwise
    @pytc.autoinit
    class l0(TreeClass):
        a: int = 2
        b: int = 1

    @pytc.leafwise
    @pytc.autoinit
    class Tree(TreeClass):
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
    @pytc.autoinit
    class l0(TreeClass):
        a: int = 2
        b: int = 1

    @pytc.autoinit
    class Tree(TreeClass):
        a: int = 1
        b: l0 = l0()

    t = Tree()

    assert pytc.is_tree_equal(t.at[1].at[t == 2].apply(lambda _: 100), Tree(1, l0(100)))
    assert pytc.is_tree_equal(t.at["b"].at[0].apply(lambda _: 100), Tree(1, l0(100)))

    # with pytest.raises(IndexError):
    #     t.at[0].at["a"].apply(lambda _: 100), Tree(1, l0(100))


def test_method_call():
    @pytc.leafwise
    @pytc.autoinit
    class Tree(TreeClass):
        a: int = 1

        def increment(self):
            self.a += 1

        def show(self):
            return 1

    t = Tree()

    @pytc.autoinit
    class Tree2(TreeClass):
        b: Tree = Tree()

    assert pytc.is_tree_equal(t.at["increment"]()[1], Tree(2))
    assert pytc.is_tree_equal(Tree2().at["b"].at["show"]()[0], 1)

    with pytest.raises(AttributeError):
        t.at["bla"]()

    with pytest.raises(TypeError):
        t.at["a"]()

    @pytc.leafwise
    @pytc.autoinit
    class A(TreeClass):
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
    @pytc.leafwise
    class Tree(TreeClass):
        def __init__(self, a=jnp.array([1, 2, 3, 4, 5])) -> None:
            self.a = a

    t = Tree()

    assert pytc.is_tree_equal(
        t.at[t > 0].at[t < 0].get(), Tree(jnp.array([], dtype=int))
    )

    with pytest.raises(AttributeError):
        t.at[t > 0].bet

    with pytest.raises(AttributeError):
        t.at["a"].bet


def test_repr_str():
    @pytc.autoinit
    @pytc.leafwise
    class Tree(TreeClass):
        a: int = 1
        b: int = 2

    t = Tree()

    assert repr(t.at["a"]) == "TreeClassIndexer(tree=Tree(a=1, b=2), where=('a',))"
    assert str(t.at["a"]) == "TreeClassIndexer(tree=Tree(a=1, b=2), where=('a',))"
    assert repr(t.at[...]) == "TreeClassIndexer(tree=Tree(a=1, b=2), where=(Ellipsis,))"


def test_not_equal():
    @pytc.leafwise
    @pytc.autoinit
    class Tree(TreeClass):
        a: int = 1
        b: float = 1.0

    t = Tree()

    assert pytc.is_tree_equal(t.at[t != 10].set(10.0), Tree(a=10.0, b=10.0))


def test_iterable_node():
    @pytc.leafwise
    @pytc.autoinit
    class Tree(TreeClass):
        a: int

    t = Tree([1, 2, 3, 4])
    assert pytc.is_tree_equal(t.at[...].set(True), Tree([True, True, True, True]))


def test_call_context():
    @pytc.leafwise
    @pytc.autoinit
    class L2(TreeClass):
        a: int = 1

        def delete(self, name):
            del self.a

    t = L2()

    with _mutable_context(t) as tx:
        tx.delete("a")

    with pytest.raises(AttributeError):
        t.delete("a")


def test_unsupported_indexing_type():
    @pytc.leafwise
    @pytc.autoinit
    class L2(TreeClass):
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

    @pytc.autoinit
    class Tree(pytc.TreeClass):
        a: Any = (1, {"b": Dict({"c": 1})})

    tree = Tree()
    assert jtu.tree_leaves(tree.at["a"].at[1].at["b"].get())[0] == Dict({"c": 1})
    assert jtu.tree_leaves(tree.at[0].at[1].at["b"].get())[0] == Dict({"c": 1})


def test_construct_tree():
    tree = construct_tree([1, 2, [3, 4]])

    assert repr(tree) == "Node(data=((None, <class 'list'>), [1, 2, [3, 4]]))"

    with pytest.raises(TypeError):
        tree.add_child("a")


def test_regexkey():
    @pytc.autoinit
    class Tree(pytc.TreeClass):
        weight_1: float = 1.0
        weight_2: float = 2.0
        weight_3: float = 3.0
        bias: float = 0.0

    tree = Tree()

    tree = tree.at[re.compile(r"weight_.*")].set(100.0)
    # Tree(weight_1=100.0, weight_2=100.0, weight_3=100.0, bias=0.0)
    assert jtu.tree_leaves(tree) == [100.0, 100.0, 100.0, 0.0]
    tree = pytc.AtIndexer({"a": 1, "b": 2}).at[re.compile(r"a")].set(100.0)
    assert tree == {"a": 100.0, "b": 2}


def test_custom_key():
    class NameTypeContainer(NamedTuple):
        name: str
        type: type

    @jax.tree_util.register_pytree_with_keys_class
    class Tree:
        def __init__(self, a, b) -> None:
            self.a = a
            self.b = b

        def tree_flatten_with_keys(self):
            ak = (NameTypeContainer("a", type(self.a)), self.a)
            bk = (NameTypeContainer("b", type(self.b)), self.b)
            return (ak, bk), None

        @classmethod
        def tree_unflatten(cls, aux_data, children):
            return cls(*children)

        @property
        def at(self):
            return pytc.AtIndexer(self)

    tree = Tree(1, 2)

    class MatchNameType(pytc.BaseKey):
        def __init__(self, name, type):
            self.name = name
            self.type = type

        def __eq__(self, other):
            if isinstance(other, NameTypeContainer):
                return other == (self.name, self.type)
            return False

    assert jax.tree_util.tree_leaves(tree.at[MatchNameType("a", int)].get()) == [1]


def test_multi_key():
    @pytc.autoinit
    class Tree(pytc.TreeClass):
        a: int = 1
        b: int = 2
        c: int = 3

    tree = Tree()
    assert pytc.is_tree_equal(
        tree.at["a"].set(100).at["b"].set(100),
        tree.at["a", "b"].set(100),
    )


def test_scan():
    @pytc.autoinit
    class Tree(pytc.TreeClass):
        a: int = 1
        b: int = 2
        c: int = 3
        d: jax.Array = jnp.array([4, 5])

    tree = Tree()

    def func_with_state(x, state):
        return x + 1, state + 1

    tree, state = tree.at["a", "b", "d"].scan(func_with_state, state=1)

    assert pytc.is_tree_equal(tree, Tree(a=2, b=3, c=3, d=jnp.array([5, 6])))
    assert state == 4
