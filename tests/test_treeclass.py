# Copyright 2023 pytreeclass authors
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

import copy
import dataclasses as dc
import inspect
from typing import Any

import jax
import jax.tree_util as jtu
import numpy.testing as npt
import pytest
from jax import numpy as jnp

import pytreeclass as tc
from pytreeclass._src.code_build import build_field_map, convert_hints_to_fields


def test_fields():
    @tc.autoinit
    class Test(tc.TreeClass):
        a: int = tc.field(default=1, metadata={"meta": 1})
        b: int = 2

    assert len(tc.fields(Test)) == 2
    assert tc.fields(Test)[0].metadata == {"meta": 1}

    with pytest.raises(ValueError):
        tc.field(kind="WRONG")

    assert (
        repr(tc.field(kind="KW_ONLY"))
        == "Field(name=None, type=None, default=NULL, init=True, repr=True, kind='KW_ONLY', metadata=None, on_setattr=(), on_getattr=(), alias=None)"
    )


def test_field():
    assert tc.field(default=1).default == 1

    with pytest.raises(TypeError):
        tc.field(metadata=1)

    @tc.autoinit
    class Test(tc.TreeClass):
        a: int = tc.field(default=1, kind="POS_ONLY")
        b: int = 2

    with pytest.raises(TypeError):
        # positonal only for a
        Test(a=1, b=2)

    assert Test(1, b=2).a == 1
    assert Test(1, 2).b == 2

    @tc.autoinit
    class Test(tc.TreeClass):
        a: int = tc.field(default=1, kind="POS_ONLY")
        b: int = tc.field(default=2, kind="POS_ONLY")

    assert Test(1, 2).a == 1
    assert Test(1, 2).b == 2

    # keyword only
    @tc.autoinit
    class Test(tc.TreeClass):
        a: int = tc.field(default=1, kind="KW_ONLY")
        b: int = 2

    with pytest.raises(TypeError):
        Test(1, 2)

    with pytest.raises(TypeError):
        Test(1, b=2)

    assert Test(a=1, b=2).a == 1

    @tc.autoinit
    class Test(tc.TreeClass):
        a: int = tc.field(default=1, kind="POS_ONLY")
        b: int = tc.field(default=2, kind="KW_ONLY")

    with pytest.raises(TypeError):
        Test(1, 2)

    assert Test(1, b=2).b == 2

    with pytest.raises(TypeError):
        Test(a=1, b=2)

    # test when init is False
    @tc.autoinit
    class Test(tc.TreeClass):
        a: int = tc.field(default=1, init=False, kind="KW_ONLY")
        b: int = 2

    with pytest.raises(TypeError):
        Test(1, 2)

    assert Test(b=2).a == 1


def test_field_alias():
    @tc.autoinit
    class Tree(tc.TreeClass):
        _name: str = tc.field(alias="name")

    tree = Tree(name="test")
    assert tree._name == "test"

    with pytest.raises(TypeError):
        tc.field(alias=1)


def test_field_nondiff():
    class Test(tc.TreeClass):
        def __init__(
            self,
            a=tc.freeze(jnp.array([1, 2, 3])),
            b=tc.freeze(jnp.array([4, 5, 6])),
        ):
            self.a = a
            self.b = b

    test = Test()

    assert jtu.tree_leaves(test) == []

    class Test(tc.TreeClass):
        def __init__(self, a=jnp.array([1, 2, 3]), b=jnp.array([4, 5, 6])):
            self.a = tc.freeze(a)
            self.b = b

    test = Test()
    npt.assert_allclose(jtu.tree_leaves(test)[0], jnp.array([4, 5, 6]))


# def test_hash():
#     class T(tc.TreeClass):
#         a: jax.Array

#     # with pytest.raises(TypeError):
#     hash(T(jnp.array([1, 2, 3])))


def test_post_init():
    @tc.autoinit
    class Test(tc.TreeClass):
        a: int = 1

        def __post_init__(self):
            self.a = 2

    t = Test()

    assert t.a == 2


def test_subclassing():
    @tc.autoinit
    class L0(tc.TreeClass):
        a: int = 1
        b: int = 3
        c: int = 5

        def inc(self, x):
            return x

        def sub(self, x):
            return x - 10

        def __post_init__(self):
            self.c = 5

    @tc.autoinit
    class L1(L0):
        a: int = 2
        b: int = 4

        def __post_init__(self):
            self.d = 5

        def inc(self, x):
            return x + 10

    l1 = L1()

    assert jtu.tree_leaves(l1) == [2, 4, 5, 5]
    assert l1.inc(10) == 20
    assert l1.sub(10) == 0
    assert l1.d == 5

    @tc.autoinit
    class L1(L0):
        a: int = 2
        b: int = 4

    l1 = L1()

    assert jtu.tree_leaves(l1) == [2, 4, 5]


def test_registering_state():
    class L0(tc.TreeClass):
        def __init__(self):
            self.a = 10
            self.b = 20

    t = L0()
    tt = copy.copy(t)

    assert tt.a == 10
    assert tt.b == 20


def test_copy():
    @tc.autoinit
    class L0(tc.TreeClass):
        a: int = 1
        b: int = 3
        c: int = 5

    t = L0()

    assert copy.copy(t).a == 1
    assert copy.copy(t).b == 3
    assert copy.copy(t).c == 5


def test_delattr():
    @tc.autoinit
    class L0(tc.TreeClass):
        a: int = 1
        b: int = 3
        c: int = 5

    t = L0()

    with pytest.raises(AttributeError):
        del t.a

    @tc.autoinit
    class L2(tc.TreeClass):
        a: int = 1

        def delete(self, name):
            del self.a

    t = L2()

    with pytest.raises(AttributeError):
        t.delete("a")


# def test_getattr():
#     with pytest.raises(AttributeError):

#
#         class L2:
#             a: int = 1

#             def __getattribute__(self, __name: str):
#                 pass

#     with pytest.raises(AttributeError):

#
#         class L3:
#             a: int = 1

#             def __getattribute__(self, __name: str):
#                 pass


# def test_treeclass_decorator_arguments():
#   (order=False)
#     class Test:
#         a: int = 1
#         b: int = 2
#         c: int = 3

#     with pytest.raises(TypeError):
#         Test() + 1


def test_is_tree_equal():
    assert tc.is_tree_equal(1, 1)
    assert tc.is_tree_equal(1, 2) is False
    assert tc.is_tree_equal(1, 2.0) is False
    assert tc.is_tree_equal([1, 2], [1, 2])

    @tc.autoinit
    class Test1(tc.TreeClass):
        a: int = 1

    class Test2(tc.TreeClass):
        a: jax.Array

        def __init__(self) -> None:
            self.a = jnp.array([1, 2, 3])

    assert tc.is_tree_equal(Test1(), Test2()) is False

    assert tc.is_tree_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))
    assert tc.is_tree_equal(jnp.array([1, 2, 3]), jnp.array([1, 3, 3])) is False

    @tc.autoinit
    class Test3(tc.TreeClass):
        a: int = 1
        b: int = 2

    assert tc.is_tree_equal(Test1(), Test3()) is False

    assert tc.is_tree_equal(jnp.array([1, 2, 3]), 1) is False


def test_mutable_field():
    with pytest.raises(TypeError):

        @tc.autoinit
        class Test(tc.TreeClass):
            a: list = [1, 2, 3]

    with pytest.raises(TypeError):

        @tc.autoinit
        class Test2(tc.TreeClass):
            a: list = tc.field(default=[1, 2, 3])


def test_setattr_delattr():
    with pytest.raises(TypeError):

        @tc.autoinit
        class Test(tc.TreeClass):
            def __setattr__(self, k, v):
                pass

    with pytest.raises(TypeError):

        @tc.autoinit
        class _(tc.TreeClass):
            def __delattr__(self, k):
                pass


def test_on_setattr():
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

    @tc.autoinit
    class Test(tc.TreeClass):
        a: int = tc.field(on_setattr=[instance_validator(int)])

    with pytest.raises(AssertionError):
        Test(a="a")

    assert Test(a=1).a == 1

    @tc.autoinit
    class Test(tc.TreeClass):
        a: int = tc.field(on_setattr=[instance_validator((int, float))])

    assert Test(a=1).a == 1
    assert Test(a=1.0).a == 1.0

    with pytest.raises(AssertionError):
        Test(a="a")

    @tc.autoinit
    class Test(tc.TreeClass):
        a: int = tc.field(on_setattr=[range_validator(0, 10)])

    with pytest.raises(AssertionError):
        Test(a=-1)

    assert Test(a=0).a == 0

    with pytest.raises(AssertionError):
        Test(a=11)

    @tc.autoinit
    class Test(tc.TreeClass):
        a: int = tc.field(on_setattr=[range_validator(0, 10), instance_validator(int)])

    with pytest.raises(AssertionError):
        Test(a=-1)

    with pytest.raises(AssertionError):
        Test(a=11)

    with pytest.raises(TypeError):

        @tc.autoinit
        class Test(tc.TreeClass):
            a: int = tc.field(on_setattr=1)

    with pytest.raises(TypeError):

        @tc.autoinit
        class Test(tc.TreeClass):
            a: int = tc.field(on_setattr=[1])

    with pytest.raises(TypeError):

        @tc.autoinit
        class Test(tc.TreeClass):
            a: int = tc.field(on_setattr=[lambda: True])

        Test(a=1)


def test_treeclass_frozen_field():
    @tc.autoinit
    class Test(tc.TreeClass):
        a: int = tc.field(on_setattr=[tc.freeze])

    t = Test(1)

    assert t.a == tc.freeze(1)
    assert jtu.tree_leaves(t) == []


def test_super():
    class Test(tc.TreeClass):
        # a:int
        def __init__(self) -> None:
            super().__init__()

    Test()


def test_optional_attrs():
    class Test(tc.TreeClass):
        def __repr__(self):
            return "a"

        def __or__(self, other):
            return 1

    tree = Test()

    assert tree.__repr__() == "a"
    assert tree | tree == 1


def test_override_repr():
    class Test(tc.TreeClass):
        def __repr__(self):
            return "a"

        def __add__(self, other):
            return 1

    assert Test().__repr__() == "a"
    assert Test() + Test() == 1


def test_optional_param():
    @tc.autoinit
    class Test(tc.TreeClass):
        a: int = tc.field(default=1)

    with pytest.raises(TypeError):
        Test() < Test()


def benchmark_dataclass_class():
    count = 10
    annot = {f"leaf_{i}": int for i in range(count)}
    leaves = {f"leaf_{i}": i for i in range(count)}
    Tree = type("Tree", (), {"__annotations__": annot, **leaves})
    Tree = dc.dataclass(Tree)
    return Tree()


def benchmark_treeclass_instance():
    count = 10
    annot = {f"leaf_{i}": int for i in range(count)}
    leaves = {f"leaf_{i}": i for i in range(count)}
    Tree = tc.autoinit(
        type("Tree", (tc.TreeClass,), {"__annotations__": annot, **leaves})
    )
    return Tree()


@pytest.mark.benchmark(group="treeclass")
def test_benchmark_treeclass_instance(benchmark):
    benchmark(benchmark_treeclass_instance)


@pytest.mark.benchmark(group="dataclass")
def test_benchmark_dataclass_class(benchmark):
    benchmark(benchmark_dataclass_class)


def test_self_field_name():
    with pytest.raises(ValueError):

        @tc.autoinit
        class Tree(tc.TreeClass):
            self: int = tc.field()


def test_instance_field_map():
    @tc.autoinit
    class Parameter(tc.TreeClass):
        value: Any

    @tc.autoinit
    class Tree(tc.TreeClass):
        bias: int = 0

        def add_param(self, name, param):
            return setattr(self, name, param)

    tree = Tree()

    _, tree_with_weight = tree.at["add_param"]("weight", Parameter(3))

    assert tree_with_weight.weight == Parameter(3)
    assert "weight" not in vars(tree)


def test_partial():
    def f(a, b, c):
        return a + b + c

    f_a = tc.Partial(f, ..., 2, 3)
    assert f_a(1) == 6

    f_b = tc.Partial(f, 1, ..., 3)
    assert f_b(2) == 6

    assert f_b == f_b
    assert hash(f_b) == hash(f_b)


def test_kind():
    @tc.autoinit
    class Tree(tc.TreeClass):
        a: int = tc.field(kind="VAR_POS")
        b: int = tc.field(kind="POS_ONLY")
        c: int = tc.field(kind="VAR_KW")
        d: int

    params = dict(inspect.signature(Tree.__init__).parameters)

    assert params["a"].kind is inspect.Parameter.VAR_POSITIONAL
    assert params["b"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert params["c"].kind is inspect.Parameter.VAR_KEYWORD
    assert params["d"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    @tc.autoinit
    class Tree(tc.TreeClass):
        a: int = tc.field(kind="VAR_POS")
        b: int = tc.field(kind="KW_ONLY")

    params = dict(inspect.signature(Tree.__init__).parameters)
    assert params["a"].kind is inspect.Parameter.VAR_POSITIONAL
    assert params["b"].kind is inspect.Parameter.KEYWORD_ONLY

    with pytest.raises(TypeError):

        @tc.autoinit
        class Tree(tc.TreeClass):
            a: int = tc.field(kind="VAR_POS")
            b: int = tc.field(kind="VAR_POS")

    with pytest.raises(TypeError):

        @tc.autoinit
        class Tree(tc.TreeClass):
            a: int = tc.field(kind="VAR_KW")
            b: int = tc.field(kind="VAR_KW")


def test_init_subclass():
    class Test:
        def __init_subclass__(cls, hello):
            cls.hello = hello

    class Test2(tc.TreeClass, Test, hello=1):
        ...

    assert Test2.hello == 1


def test_nested_mutation():
    @tc.autoinit
    class InnerModule(tc.TreeClass):
        a: int = 1

        def f(self):
            self.a += 1
            return self.a

    @tc.autoinit
    class OuterModule(tc.TreeClass):
        inner: InnerModule = InnerModule()

        def ff(self):
            return self.inner.f()

        def df(self):
            del self.inner.a

    _, v = OuterModule().at["ff"]()
    assert v.inner.a == 2

    _, v = OuterModule().at["df"]()
    assert "a" not in v.inner.__dict__


def test_autoinit_and_user_defined_init():
    @tc.autoinit
    class Tree(tc.TreeClass):
        b: int

        def __init__(self, a):
            self.a = a

    Tree(a=1)

    assert True


def test_nohints():
    assert convert_hints_to_fields(int) is int


def non_field_builder():
    class T:
        ...

    assert dict(build_field_map(T)) == {}


def test_on_getattr():
    @tc.autoinit
    class Tree(tc.TreeClass):
        a: int = tc.field(on_getattr=[lambda x: x + 1])

    assert Tree(a=1).a == 2

    # with subclassing

    @tc.autoinit
    class Parent:
        a: int = tc.field(on_getattr=[lambda x: x + 1])

    class Child(Parent):
        pass

    child = Child(a=1)

    assert child.a == 2
    assert vars(child)["a"] == 1
