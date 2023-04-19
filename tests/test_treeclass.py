import copy
import dataclasses as dc
from typing import Any

import jax
import jax.tree_util as jtu
import numpy.testing as npt
import pytest
from jax import numpy as jnp

import pytreeclass as pytc


def test_field_metadata():
    class Test(pytc.TreeClass, leafwise=True):
        a: int = pytc.field(default=1, metadata={"a": 1})

    assert Test()._fields["a"].metadata["a"] == 1


def test_field_mutually_exclusive():
    with pytest.raises(ValueError):
        pytc.field(default=1, factory=lambda: 1)

    with pytest.raises(ValueError):
        pytc.field(default=1, pos_only=True, kw_only=True)


def test_field():
    assert pytc.field(default=1).default == 1

    with pytest.raises(TypeError):
        pytc.field(metadata=1)

    class Test(pytc.TreeClass):
        a: int = pytc.field(default=1, pos_only=True)
        b: int = 2

    with pytest.raises(TypeError):
        # positonal only for a
        Test(a=1, b=2)

    assert Test(1, b=2).a == 1
    assert Test(1, 2).b == 2

    class Test(pytc.TreeClass):
        a: int = pytc.field(default=1, pos_only=True)
        b: int = pytc.field(default=2, pos_only=True)

    assert Test(1, 2).a == 1
    assert Test(1, 2).b == 2

    # keyword only
    class Test(pytc.TreeClass):
        a: int = pytc.field(default=1, kw_only=True)
        b: int = 2

    with pytest.raises(TypeError):
        Test(1, 2)

    with pytest.raises(TypeError):
        Test(1, b=2)

    assert Test(a=1, b=2).a == 1

    class Test(pytc.TreeClass):
        a: int = pytc.field(default=1, pos_only=True)
        b: int = pytc.field(default=2, kw_only=True)

    with pytest.raises(TypeError):
        Test(1, 2)

    assert Test(1, b=2).b == 2

    with pytest.raises(TypeError):
        Test(a=1, b=2)

    # test when init is False
    class Test(pytc.TreeClass):
        a: int = pytc.field(default=1, init=False, kw_only=True)
        b: int = 2

    with pytest.raises(TypeError):
        Test(1, 2)

    assert Test(b=2).a == 1

    class Test(pytc.TreeClass):
        a: int = pytc.field(default=1)
        b: int = pytc.field(factory=lambda: 1)

        def __init__(self) -> None:
            pass

    with pytest.raises(AttributeError):
        Test()


def test_field_alias():
    class Tree(pytc.TreeClass):
        _name: str = pytc.field(alias="name")

    tree = Tree(name="test")
    assert tree._name == "test"

    with pytest.raises(TypeError):
        pytc.field(alias=1)


def test_field_hash():
    f1 = pytc.field(default=1.0, callbacks=[lambda x: x])
    f2 = pytc.field(default=1.0)
    hash(f1) == hash(f2)


def test_field_nondiff():
    class Test(pytc.TreeClass):
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

    class Test(pytc.TreeClass):
        a: jnp.ndarray
        b: jnp.ndarray

        def __init__(self, a=jnp.array([1, 2, 3]), b=jnp.array([4, 5, 6])):
            self.a = pytc.freeze(a)
            self.b = b

    test = Test()
    npt.assert_allclose(jtu.tree_leaves(test)[0], jnp.array([4, 5, 6]))


def test_hash():
    class T(pytc.TreeClass):
        a: jnp.ndarray

    # with pytest.raises(TypeError):
    hash(T(jnp.array([1, 2, 3])))


def test_post_init():
    class Test(pytc.TreeClass):
        a: int = 1

        def __post_init__(self):
            self.a = 2

    t = Test()

    assert t.a == 2


def test_subclassing():
    class L0(pytc.TreeClass):
        a: int = 1
        b: int = 3
        c: int = 5

        def inc(self, x):
            return x

        def sub(self, x):
            return x - 10

        def __post_init__(self):
            self.c = 5

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

    assert jtu.tree_leaves(l1) == [2, 4, 5]


def test_registering_state():
    class L0(pytc.TreeClass):
        def __init__(self):
            self.a = 10
            self.b = 20

    t = L0()
    tt = copy.copy(t)

    assert tt.a == 10
    assert tt.b == 20


def test_copy():
    class L0(pytc.TreeClass):
        a: int = 1
        b: int = 3
        c: int = 5

    t = L0()

    assert copy.copy(t).a == 1
    assert copy.copy(t).b == 3
    assert copy.copy(t).c == 5


def test_delattr():
    class L0(pytc.TreeClass):
        a: int = 1
        b: int = 3
        c: int = 5

    t = L0()

    with pytest.raises(AttributeError):
        del t.a

    class L2(pytc.TreeClass):
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
    assert pytc.is_tree_equal(1, 1)
    assert pytc.is_tree_equal(1, 2) is False
    assert pytc.is_tree_equal(1, 2.0) is False
    assert pytc.is_tree_equal([1, 2], [1, 2])

    class Test1(pytc.TreeClass):
        a: int = 1

    class Test2(pytc.TreeClass):
        a: jnp.ndarray

        def __init__(self) -> None:
            self.a = jnp.array([1, 2, 3])

    assert pytc.is_tree_equal(Test1(), Test2()) is False

    assert pytc.is_tree_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))
    assert pytc.is_tree_equal(jnp.array([1, 2, 3]), jnp.array([1, 3, 3])) is False

    class Test3(pytc.TreeClass):
        a: int = 1
        b: int = 2

    assert pytc.is_tree_equal(Test1(), Test3()) is False

    assert pytc.is_tree_equal(jnp.array([1, 2, 3]), 1) is False


def test_params():
    class l0(pytc.TreeClass):
        a: int = 2

    class l1(pytc.TreeClass):
        a: int = 1
        b: l0 = l0()

    t1 = l1(1, l0(100))

    # t2 = copy.copy(t1)
    # t3 = l1(1, l0(100))

    with pytest.raises(AttributeError):
        t1.__FIELDS__["a"].default = 100


def test_mutable_field():
    with pytest.raises(TypeError):

        class Test(pytc.TreeClass):
            a: list = [1, 2, 3]

    with pytest.raises(TypeError):

        class Test2(pytc.TreeClass):
            a: list = pytc.field(default=[1, 2, 3])


def test_setattr_delattr():
    with pytest.raises(TypeError):

        class Test(pytc.TreeClass):
            def __setattr__(self, k, v):
                pass

    with pytest.raises(TypeError):

        class _(pytc.TreeClass):
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

    class Test(pytc.TreeClass):
        a: int = pytc.field(callbacks=[instance_validator(int)])

    with pytest.raises(AssertionError):
        Test(a="a")

    assert Test(a=1).a == 1

    class Test(pytc.TreeClass):
        a: int = pytc.field(callbacks=[instance_validator((int, float))])

    assert Test(a=1).a == 1
    assert Test(a=1.0).a == 1.0

    with pytest.raises(AssertionError):
        Test(a="a")

    class Test(pytc.TreeClass):
        a: int = pytc.field(callbacks=[range_validator(0, 10)])

    with pytest.raises(AssertionError):
        Test(a=-1)

    assert Test(a=0).a == 0

    with pytest.raises(AssertionError):
        Test(a=11)

    class Test(pytc.TreeClass):
        a: int = pytc.field(callbacks=[range_validator(0, 10), instance_validator(int)])

    with pytest.raises(AssertionError):
        Test(a=-1)

    with pytest.raises(AssertionError):
        Test(a=11)

    with pytest.raises(TypeError):

        class Test(pytc.TreeClass):
            a: int = pytc.field(callbacks=1)

    with pytest.raises(TypeError):

        class Test(pytc.TreeClass):
            a: int = pytc.field(callbacks=[1])

    with pytest.raises(TypeError):

        class Test(pytc.TreeClass):
            a: int = pytc.field(callbacks=[lambda: True])

        Test(a=1)


def test_treeclass_frozen_field():
    class Test(pytc.TreeClass):
        a: int = pytc.field(callbacks=[pytc.freeze])

    t = Test(1)

    assert t.a == pytc.freeze(1)
    assert jtu.tree_leaves(t) == []


def test_key_error():
    class Test(pytc.TreeClass):
        a: int = pytc.field()

        def __init__(self) -> None:
            return

    with pytest.raises(AttributeError):
        Test()


def test_super():
    class Test(pytc.TreeClass):
        # a:int
        def __init__(self) -> None:
            super().__init__()

    Test()


def test_optional_attrs():
    class Test(pytc.TreeClass):
        def __repr__(self):
            return "a"

        def __or__(self, other):
            return 1

    tree = Test()

    assert tree.__repr__() == "a"
    assert tree | tree == 1


def test_override_repr():
    class Test(pytc.TreeClass):
        def __repr__(self):
            return "a"

        def __add__(self, other):
            return 1

    assert Test().__repr__() == "a"
    assert Test() + Test() == 1


def test_optional_param():
    class Test(pytc.TreeClass):
        a: int = pytc.field(default=1)

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
    Tree = type("Tree", (pytc.TreeClass,), {"__annotations__": annot, **leaves})
    return Tree()


@pytest.mark.benchmark(group="treeclass")
def test_benchmark_treeclass_instance(benchmark):
    benchmark(benchmark_treeclass_instance)


@pytest.mark.benchmark(group="dataclass")
def test_benchmark_dataclass_class(benchmark):
    benchmark(benchmark_dataclass_class)


def test_incorrect_trace_func():
    class T:
        def __init__(self):
            self.a = 1

    def trace_func(tree):
        names = ("a",)
        types = (type(tree.a),)
        return [*zip(names, types)]

    flatten_func = lambda tree: ((tree.a,), None)
    unflatten_func = lambda _, x: T(x)

    jax.tree_util.register_pytree_node(T, flatten_func, unflatten_func)

    pytc.register_pytree_node_trace(T, trace_func)

    with pytest.raises(ValueError):
        # improper length
        pytc.tree_leaves_with_trace(T())

    class T:
        def __init__(self):
            self.a = 1

    def trace_func(tree):
        names = (1,)
        types = (type(tree.a),)
        indices = (0,)
        return [*zip(names, types, indices)]

    flatten_func = lambda tree: ((tree.a,), None)
    unflatten_func = lambda _, x: T(x)

    jax.tree_util.register_pytree_node(T, flatten_func, unflatten_func)

    pytc.register_pytree_node_trace(T, trace_func)

    with pytest.raises(TypeError):
        # improper name entry
        pytc.tree_leaves_with_trace(T())

    class T:
        def __init__(self):
            self.a = 1

    def trace_func(tree):
        names = ("a",)
        types = (1,)
        indices = (0,)
        return [*zip(names, types, indices)]

    flatten_func = lambda tree: ((tree.a,), None)
    unflatten_func = lambda _, x: T(x)

    jax.tree_util.register_pytree_node(T, flatten_func, unflatten_func)

    pytc.register_pytree_node_trace(T, trace_func)

    with pytest.raises(TypeError):
        # improper type entry
        pytc.tree_leaves_with_trace(T())

    class T:
        def __init__(self):
            self.a = 1

    def trace_func(tree):
        names = ("a",)
        types = (int,)
        indices = ("a",)
        return [*zip(names, types, indices)]

    flatten_func = lambda tree: ((tree.a,), None)
    unflatten_func = lambda _, x: T(x)

    jax.tree_util.register_pytree_node(T, flatten_func, unflatten_func)

    pytc.register_pytree_node_trace(T, trace_func)

    with pytest.raises(TypeError):
        # improper index entry
        pytc.tree_leaves_with_trace(T())

    class T:
        def __init__(self):
            self.a = 1

    def trace_func(tree):
        return [""]

    flatten_func = lambda tree: ((tree.a,), None)
    unflatten_func = lambda _, x: T(x)

    jax.tree_util.register_pytree_node(T, flatten_func, unflatten_func)

    pytc.register_pytree_node_trace(T, trace_func)

    with pytest.raises(TypeError):
        # improper return type
        pytc.tree_leaves_with_trace(T())

    # add valid trace func
    class T:
        def __init__(self):
            self.a = 1

    def trace_func(tree):
        names = ("a",)
        types = (int,)
        indices = (0,)
        return [*zip(names, types, indices)]

    flatten_func = lambda tree: ((tree.a,), None)
    unflatten_func = lambda _, x: T(x)

    jax.tree_util.register_pytree_node(T, flatten_func, unflatten_func)

    pytc.register_pytree_node_trace(T, trace_func)

    # first run should be validated
    pytc.tree_leaves_with_trace(T())
    # second run skip validation
    pytc.tree_leaves_with_trace(T())

    assert True


def test_self_field_name():
    with pytest.raises(ValueError):

        class Tree(pytc.TreeClass):
            self: int = pytc.field()


def test_instance_field_map():
    class Parameter(pytc.TreeClass):
        value: Any

    class Tree(pytc.TreeClass):
        bias: int = 0

        def add_param(self, name, param):
            return setattr(self, name, param)

    tree = Tree()

    _, tree_with_weight = tree.at["add_param"]("weight", Parameter(3))

    assert tree._fields != tree_with_weight._fields
    assert tree_with_weight.weight == Parameter(3)
    assert "weight" not in vars(tree)
    assert "weight" not in tree._fields


def test_field_factory():
    class Tree(pytc.TreeClass):
        a: int = pytc.field(factory=lambda: 1)

    assert Tree().a == 1
