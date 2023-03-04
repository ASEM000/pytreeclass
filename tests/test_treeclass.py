import copy
import dataclasses as dc

import jax.tree_util as jtu
import numpy.testing as npt
import pytest
from jax import numpy as jnp

import pytreeclass as pytc
from pytreeclass._src.tree_decorator import _FIELD_MAP, _dataclass_like_fields


def test_field():

    with pytest.raises(ValueError):
        pytc.field(default=1, default_factory=lambda: 1)

    assert pytc.field(default=1).default == 1

    with pytest.raises(TypeError):
        pytc.field(metadata=1)

    @pytc.treeclass
    class Test:
        a: int = pytc.field(default=1, metadata={"a": 1})

    assert getattr(Test(), _FIELD_MAP)["a"].metadata["a"] == 1


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
            a=pytc.freeze(jnp.array([1, 2, 3])),
            b=pytc.freeze(jnp.array([4, 5, 6])),
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
            self.a = pytc.freeze(a)
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

        def __post_init__(self):
            self.c = 5

    @pytc.treeclass
    class L1(L0):
        a: int = 2
        b: int = 4

        def __post_init__(self):
            self.d = 5

        def inc(self, x):
            return x + 10

    l1 = L1()

    assert jtu.tree_leaves(l1) == [2, 4, 5]
    assert l1.inc(10) == 20
    assert l1.sub(10) == 0
    assert l1.d == 5

    class L1(L0):
        a: int = 2
        b: int = 4

    l1 = L1()

    # leaves of L0 are only considered
    # as L1 is not decorated with @treeclass
    assert jtu.tree_leaves(l1) == [1, 3, 5]


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

    with pytest.raises(AttributeError):
        del t.a

    @pytc.treeclass
    class L2:
        a: int = 1

        def delete(self, name):
            del self.a

    t = L2()

    with pytest.raises(AttributeError):
        t.delete("a")


# def test_getattr():
#     with pytest.raises(AttributeError):

#         @pytc.treeclass
#         class L2:
#             a: int = 1

#             def __getattribute__(self, __name: str):
#                 pass

#     with pytest.raises(AttributeError):

#         @pytc.treeclass
#         class L3:
#             a: int = 1

#             def __getattribute__(self, __name: str):
#                 pass


# def test_treeclass_decorator_arguments():
#     @pytc.treeclass(order=False)
#     class Test:
#         a: int = 1
#         b: int = 2
#         c: int = 3

#     with pytest.raises(TypeError):
#         Test() + 1


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


def test_setattr_delattr():

    with pytest.raises(AttributeError):

        @pytc.treeclass
        class Test:
            def __setattr__(self, k, v):
                pass

    with pytest.raises(AttributeError):

        @pytc.treeclass
        class _:
            def __delattr__(self, k):
                pass


def test_callbacks():
    def instance_validator(types):
        def _instance_validator(x):
            if isinstance(x, types) is False:
                raise AssertionError
            return x

        return _instance_validator

    def range_validator(min, max):
        def _range_validator(x):
            if x < min or x > max:
                raise AssertionError
            return x

        return _range_validator

    @pytc.treeclass
    class Test:
        a: int = pytc.field(callbacks=[instance_validator(int)])

    with pytest.raises(AssertionError):
        Test(a="a")

    assert Test(a=1).a == 1

    @pytc.treeclass
    class Test:
        a: int = pytc.field(callbacks=[instance_validator((int, float))])

    assert Test(a=1).a == 1
    assert Test(a=1.0).a == 1.0

    with pytest.raises(AssertionError):
        Test(a="a")

    @pytc.treeclass
    class Test:
        a: int = pytc.field(callbacks=[range_validator(0, 10)])

    with pytest.raises(AssertionError):
        Test(a=-1)

    assert Test(a=0).a == 0

    with pytest.raises(AssertionError):
        Test(a=11)

    @pytc.treeclass
    class Test:
        a: int = pytc.field(callbacks=[range_validator(0, 10), instance_validator(int)])

    with pytest.raises(AssertionError):
        Test(a=-1)

    with pytest.raises(AssertionError):
        Test(a=11)

    with pytest.raises(TypeError):

        @pytc.treeclass
        class Test:
            a: int = pytc.field(callbacks=1)

    with pytest.raises(TypeError):

        @pytc.treeclass
        class Test:
            a: int = pytc.field(callbacks=[1])


def test_treeclass_frozen_field():
    @pytc.treeclass
    class Test:
        a: int = pytc.field(callbacks=[pytc.freeze])

    t = Test(1)

    assert t.a == 1
    assert jtu.tree_leaves(t) == []


def test_key_error():
    @pytc.treeclass
    class Test:
        a: int = pytc.field()

        def __init__(self) -> None:
            return

    with pytest.raises(AttributeError):
        Test()


def test_dataclass_fields_like():
    @dc.dataclass
    class Test:
        a: int = 1

    assert _dataclass_like_fields(Test) == dc.fields(Test)

    with pytest.raises(TypeError):
        _dataclass_like_fields(1)


def test_super():
    @pytc.treeclass
    class Test:
        # a:int
        def __init__(self) -> None:
            super().__init__()

    Test()


def test_optional_attrs():
    @pytc.treeclass
    class Test:
        def __repr__(self):
            return "a"

        def __or__(self, other):
            return 1

    tree = Test()

    assert tree.__repr__() == "a"
    assert tree | tree == 1
