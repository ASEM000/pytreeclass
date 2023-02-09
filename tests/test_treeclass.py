import copy

import jax.tree_util as jtu
import numpy.testing as npt
import pytest
from jax import numpy as jnp

import pytreeclass as pytc
from pytreeclass._src.tree_base import ImmutableTreeError
from pytreeclass._src.tree_freeze import _MutableContext


def test_field():

    with pytest.raises(ValueError):
        pytc.field(default=1, default_factory=lambda: 1)

    assert pytc.field(default=1).default == 1


def test_field_nondiff():
    @pytc.treeclass
    class Test:
        a: int = 1
        b: int = 2
        c: int = 3

    test = Test()

    @pytc.treeclass
    class Test:
        a: jnp.ndarray
        b: jnp.ndarray

        def __init__(self, a=jnp.array([1, 2, 3]), b=jnp.array([4, 5, 6])):
            self.a = a
            self.b = b

    test = Test()

    @pytc.treeclass
    class Test:
        a: jnp.ndarray
        b: jnp.ndarray

        def __init__(
            self,
            a=pytc.FrozenWrapper(jnp.array([1, 2, 3])),
            b=pytc.FrozenWrapper(jnp.array([4, 5, 6])),
        ):

            self.a = a
            self.b = b

    test = Test()

    assert jtu.tree_leaves(test) == []

    @pytc.treeclass
    class Test:
        a: jnp.ndarray
        b: jnp.ndarray

        def __init__(self, a=jnp.array([1, 2, 3]), b=jnp.array([4, 5, 6])):
            self.a = pytc.FrozenWrapper(a)
            self.b = b

    test = Test()
    npt.assert_allclose(jtu.tree_leaves(test)[0], jnp.array([4, 5, 6]))


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


def test_registering_state():
    @pytc.treeclass
    class L0:
        def __init__(self):
            self.a = 10
            self.b = 20

    t = L0()
    tt = copy.copy(t)

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


def test_delattr():
    @pytc.treeclass
    class L0:
        a: int = 1
        b: int = 3
        c: int = 5

    t = L0()

    with pytest.raises(ImmutableTreeError):
        del t.a

    @pytc.treeclass
    class L2:
        a: int = 1

        def delete(self, name):
            del self.a

    t = L2()

    with _MutableContext(t, inplace=False) as tx:
        tx.delete("a")

    with _MutableContext(t, inplace=True) as tx:
        tx.delete("a")

    with pytest.raises(ImmutableTreeError):
        t.delete("a")


def test_treeclass_decorator_arguments():
    @pytc.treeclass(repr=False)
    class Test:
        a: int = 1
        b: int = 2
        c: int = 3

    assert "__repr__" not in Test.__dict__

    @pytc.treeclass(order=False)
    class Test:
        a: int = 1
        b: int = 2
        c: int = 3

    with pytest.raises(TypeError):
        Test() + 1


def test_is_tree_equal():

    assert pytc.is_tree_equal(1, 1)
    assert pytc.is_tree_equal(1, 2) is False
    assert pytc.is_tree_equal(1, 2.0) is False
    assert pytc.is_tree_equal([1, 2], [1, 2])

    @pytc.treeclass
    class Test1:
        a: int = 1

    @pytc.treeclass
    class Test2:
        a: jnp.ndarray

        def __init__(self) -> None:
            self.a = jnp.array([1, 2, 3])

    assert pytc.is_tree_equal(Test1(), Test2()) is False

    assert pytc.is_tree_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))
    assert pytc.is_tree_equal(jnp.array([1, 2, 3]), jnp.array([1, 3, 3])) is False

    @pytc.treeclass
    class Test3:
        a: int = 1
        b: int = 2

    assert pytc.is_tree_equal(Test1(), Test3()) is False

    assert pytc.is_tree_equal(jnp.array([1, 2, 3]), 1) is False


def test_params():
    @pytc.treeclass
    class l0:
        a: int = 2

    @pytc.treeclass
    class l1:
        a: int = 1
        b: l0 = l0()

    t1 = l1(1, l0(100))

    # t2 = copy.copy(t1)
    # t3 = l1(1, l0(100))

    with pytest.raises(AttributeError):
        t1.__FIELDS__["a"].default = 100


def test_mutable_field():

    with pytest.raises(TypeError):

        @pytc.treeclass
        class Test:
            a: list = [1, 2, 3]


def test_non_class_input():
    with pytest.raises(TypeError):

        @pytc.treeclass
        def f(x):
            return x
