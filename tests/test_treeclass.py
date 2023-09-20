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

import numpy.testing as npt
import pytest

from pytreeclass._src.backend import backend, treelib
from pytreeclass._src.code_build import (
    autoinit,
    build_field_map,
    convert_hints_to_fields,
    field,
    fields,
)
from pytreeclass._src.tree_base import TreeClass
from pytreeclass._src.tree_mask import freeze
from pytreeclass._src.tree_util import Partial, is_tree_equal

if backend == "jax":
    import jax.numpy as arraylib
elif backend in ["numpy", "default"]:
    import numpy as arraylib
elif backend == "torch":
    import torch as arraylib

    arraylib.array = arraylib.tensor
else:
    raise ImportError("no backend installed")


def test_fields():
    @autoinit
    class Test(TreeClass):
        a: int = field(default=1, metadata={"meta": 1})
        b: int = 2

    assert len(fields(Test)) == 2
    assert fields(Test)[0].metadata == {"meta": 1}

    with pytest.raises(ValueError):
        field(kind="WRONG")

    assert (
        repr(field(kind="KW_ONLY"))
        == "Field(name=NULL, type=NULL, default=NULL, init=True, repr=True, kind='KW_ONLY', metadata=None, on_setattr=(), on_getattr=(), alias=None)"
    )


def test_field():
    assert field(default=1).default == 1

    with pytest.raises(TypeError):
        field(metadata=1)

    @autoinit
    class Test(TreeClass):
        a: int = field(default=1, kind="POS_ONLY")
        b: int = 2

    with pytest.raises(TypeError):
        # positonal only for a
        Test(a=1, b=2)

    assert Test(1, b=2).a == 1
    assert Test(1, 2).b == 2

    @autoinit
    class Test(TreeClass):
        a: int = field(default=1, kind="POS_ONLY")
        b: int = field(default=2, kind="POS_ONLY")

    assert Test(1, 2).a == 1
    assert Test(1, 2).b == 2

    # keyword only
    @autoinit
    class Test(TreeClass):
        a: int = field(default=1, kind="KW_ONLY")
        b: int = 2

    with pytest.raises(TypeError):
        Test(1, 2)

    with pytest.raises(TypeError):
        Test(1, b=2)

    assert Test(a=1, b=2).a == 1

    @autoinit
    class Test(TreeClass):
        a: int = field(default=1, kind="POS_ONLY")
        b: int = field(default=2, kind="KW_ONLY")

    with pytest.raises(TypeError):
        Test(1, 2)

    assert Test(1, b=2).b == 2

    with pytest.raises(TypeError):
        Test(a=1, b=2)

    # test when init is False
    @autoinit
    class Test(TreeClass):
        a: int = field(default=1, init=False, kind="KW_ONLY")
        b: int = 2

    with pytest.raises(TypeError):
        Test(1, 2)

    assert Test(b=2).a == 1


def test_field_alias():
    @autoinit
    class Tree(TreeClass):
        _name: str = field(alias="name")

    tree = Tree(name="test")
    assert tree._name == "test"

    with pytest.raises(TypeError):
        field(alias=1)


def test_field_nondiff():
    class Test(TreeClass):
        def __init__(
            self,
            a=freeze(arraylib.array([1, 2, 3])),
            b=freeze(arraylib.array([4, 5, 6])),
        ):
            self.a = a
            self.b = b

    test = Test()

    assert treelib.tree_flatten(test)[0] == []

    class Test(TreeClass):
        def __init__(self, a=arraylib.array([1, 2, 3]), b=arraylib.array([4, 5, 6])):
            self.a = freeze(a)
            self.b = b

    test = Test()
    npt.assert_allclose(treelib.tree_flatten(test)[0][0], arraylib.array([4, 5, 6]))


def test_post_init():
    @autoinit
    class Test(TreeClass):
        a: int = 1

        def __post_init__(self):
            self.a = 2

    t = Test()

    assert t.a == 2


def test_subclassing():
    @autoinit
    class L0(TreeClass):
        a: int = 1
        b: int = 3
        c: int = 5

        def inc(self, x):
            return x

        def sub(self, x):
            return x - 10

        def __post_init__(self):
            self.c = 5

    @autoinit
    class L1(L0):
        a: int = 2
        b: int = 4

        def __post_init__(self):
            self.d = 5

        def inc(self, x):
            return x + 10

    l1 = L1()

    assert treelib.tree_flatten(l1)[0] == [2, 4, 5, 5]
    assert l1.inc(10) == 20
    assert l1.sub(10) == 0
    assert l1.d == 5

    @autoinit
    class L1(L0):
        a: int = 2
        b: int = 4

    l1 = L1()

    assert treelib.tree_flatten(l1)[0] == [2, 4, 5]


def test_registering_state():
    class L0(TreeClass):
        def __init__(self):
            self.a = 10
            self.b = 20

    t = L0()
    tt = copy.copy(t)

    assert tt.a == 10
    assert tt.b == 20


def test_copy():
    @autoinit
    class L0(TreeClass):
        a: int = 1
        b: int = 3
        c: int = 5

    t = L0()

    assert copy.copy(t).a == 1
    assert copy.copy(t).b == 3
    assert copy.copy(t).c == 5


def test_delattr():
    @autoinit
    class L0(TreeClass):
        a: int = 1
        b: int = 3
        c: int = 5

    t = L0()

    with pytest.raises(AttributeError):
        del t.a

    @autoinit
    class L2(TreeClass):
        a: int = 1

        def delete(self, name):
            del self.a

    t = L2()

    with pytest.raises(AttributeError):
        t.delete("a")


@pytest.mark.skipif(backend == "default", reason="no array")
def test_is_tree_equal():
    assert is_tree_equal(1, 1)
    assert is_tree_equal(1, 2) is False
    assert is_tree_equal(1, 2.0) is False
    assert is_tree_equal([1, 2], [1, 2])

    @autoinit
    class Test1(TreeClass):
        a: int = 1

    @autoinit
    class Test2(TreeClass):
        a: Any = arraylib.array([1, 2, 3])

    assert is_tree_equal(Test1(), Test2()) is False

    assert is_tree_equal(arraylib.array([1, 2, 3]), arraylib.array([1, 2, 3]))
    assert is_tree_equal(arraylib.array([1, 2, 3]), arraylib.array([1, 3, 3])) is False

    @autoinit
    class Test3(TreeClass):
        a: int = 1
        b: int = 2

    assert is_tree_equal(Test1(), Test3()) is False

    assert is_tree_equal(arraylib.array([1, 2, 3]), 1) is False


def test_mutable_field():
    with pytest.raises(TypeError):

        @autoinit
        class Test(TreeClass):
            a: list = [1, 2, 3]

    with pytest.raises(TypeError):

        @autoinit
        class Test2(TreeClass):
            a: list = field(default=[1, 2, 3])


def test_setattr_delattr():
    with pytest.raises(TypeError):

        @autoinit
        class Test(TreeClass):
            def __setattr__(self, k, v):
                pass

    with pytest.raises(TypeError):

        @autoinit
        class _(TreeClass):
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

    @autoinit
    class Test(TreeClass):
        a: int = field(on_setattr=[instance_validator(int)])

    with pytest.raises(AssertionError):
        Test(a="a")

    assert Test(a=1).a == 1

    @autoinit
    class Test(TreeClass):
        a: int = field(on_setattr=[instance_validator((int, float))])

    assert Test(a=1).a == 1
    assert Test(a=1.0).a == 1.0

    with pytest.raises(AssertionError):
        Test(a="a")

    @autoinit
    class Test(TreeClass):
        a: int = field(on_setattr=[range_validator(0, 10)])

    with pytest.raises(AssertionError):
        Test(a=-1)

    assert Test(a=0).a == 0

    with pytest.raises(AssertionError):
        Test(a=11)

    @autoinit
    class Test(TreeClass):
        a: int = field(on_setattr=[range_validator(0, 10), instance_validator(int)])

    with pytest.raises(AssertionError):
        Test(a=-1)

    with pytest.raises(AssertionError):
        Test(a=11)

    with pytest.raises(TypeError):

        @autoinit
        class Test(TreeClass):
            a: int = field(on_setattr=1)

    with pytest.raises(TypeError):

        @autoinit
        class Test(TreeClass):
            a: int = field(on_setattr=[1])

    with pytest.raises(TypeError):

        @autoinit
        class Test(TreeClass):
            a: int = field(on_setattr=[lambda: True])

        Test(a=1)


def test_treeclass_frozen_field():
    @autoinit
    class Test(TreeClass):
        a: int = field(on_setattr=[freeze])

    t = Test(1)

    assert t.a == freeze(1)
    assert treelib.tree_flatten(t)[0] == []


def test_super():
    class Test(TreeClass):
        # a:int
        def __init__(self) -> None:
            super().__init__()

    Test()


def test_optional_attrs():
    class Test(TreeClass):
        def __repr__(self):
            return "a"

        def __or__(self, other):
            return 1

    tree = Test()

    assert tree.__repr__() == "a"
    assert tree | tree == 1


def test_override_repr():
    class Test(TreeClass):
        def __repr__(self):
            return "a"

        def __add__(self, other):
            return 1

    assert Test().__repr__() == "a"
    assert Test() + Test() == 1


def test_optional_param():
    @autoinit
    class Test(TreeClass):
        a: int = field(default=1)

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
    Tree = autoinit(type("Tree", (TreeClass,), {"__annotations__": annot, **leaves}))
    return Tree()


@pytest.mark.benchmark(group="treeclass")
def test_benchmark_treeclass_instance(benchmark):
    benchmark(benchmark_treeclass_instance)


@pytest.mark.benchmark(group="dataclass")
def test_benchmark_dataclass_class(benchmark):
    benchmark(benchmark_dataclass_class)


def test_self_field_name():
    with pytest.raises(ValueError):

        @autoinit
        class Tree(TreeClass):
            self: int = field()


def test_instance_field_map():
    @autoinit
    class Parameter(TreeClass):
        value: Any

    @autoinit
    class Tree(TreeClass):
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

    f_a = Partial(f, ..., 2, 3)
    assert f_a(1) == 6

    f_b = Partial(f, 1, ..., 3)
    assert f_b(2) == 6

    assert f_b == f_b
    assert hash(f_b) == hash(f_b)


def test_kind():
    @autoinit
    class Tree(TreeClass):
        a: int = field(kind="VAR_POS")
        b: int = field(kind="POS_ONLY")
        c: int = field(kind="VAR_KW")
        d: int

    params = dict(inspect.signature(Tree.__init__).parameters)

    assert params["a"].kind is inspect.Parameter.VAR_POSITIONAL
    assert params["b"].kind is inspect.Parameter.POSITIONAL_ONLY
    assert params["c"].kind is inspect.Parameter.VAR_KEYWORD
    assert params["d"].kind is inspect.Parameter.POSITIONAL_OR_KEYWORD

    @autoinit
    class Tree(TreeClass):
        a: int = field(kind="VAR_POS")
        b: int = field(kind="KW_ONLY")

    params = dict(inspect.signature(Tree.__init__).parameters)
    assert params["a"].kind is inspect.Parameter.VAR_POSITIONAL
    assert params["b"].kind is inspect.Parameter.KEYWORD_ONLY

    with pytest.raises(TypeError):

        @autoinit
        class Tree(TreeClass):
            a: int = field(kind="VAR_POS")
            b: int = field(kind="VAR_POS")

    with pytest.raises(TypeError):

        @autoinit
        class Tree(TreeClass):
            a: int = field(kind="VAR_KW")
            b: int = field(kind="VAR_KW")


def test_init_subclass():
    class Test:
        def __init_subclass__(cls, hello):
            cls.hello = hello

    class Test2(TreeClass, Test, hello=1):
        ...

    assert Test2.hello == 1


def test_nested_mutation():
    @autoinit
    class InnerModule(TreeClass):
        a: int = 1

        def f(self):
            self.a += 1
            return self.a

    @autoinit
    class OuterModule(TreeClass):
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
    @autoinit
    class Tree(TreeClass):
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
    @autoinit
    class Tree(TreeClass):
        a: int = field(on_getattr=[lambda x: x + 1])

    assert Tree(a=1).a == 2

    # with subclassing

    @autoinit
    class Parent:
        a: int = field(on_getattr=[lambda x: x + 1])

    class Child(Parent):
        pass

    child = Child(a=1)

    assert child.a == 2
    assert vars(child)["a"] == 1


def test_unannotated_field():
    class T:
        a = field(default=1)

    assert str(T.a.type) == "NULL"
