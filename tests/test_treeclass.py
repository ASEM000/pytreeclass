import copy
import dataclasses as dc
import functools as ft
from typing import Tuple

import jax
import jax.tree_util as jtu
import numpy.testing as npt
import pytest
from jax import numpy as jnp

import pytreeclass as pytc
from pytreeclass._src.tree_decorator import _field_registry


def test_field():
    with pytest.raises(ValueError):
        pytc.field(default=1, factory=lambda: 1)

    assert pytc.field(default=1).default == 1

    with pytest.raises(TypeError):
        pytc.field(metadata=1)

    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        a: int = pytc.field(default=1, metadata={"a": 1})

    assert _field_registry[Test]["a"].metadata["a"] == 1

    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        a: int = pytc.field(default=1, pos_only=True)
        b: int = 2

    with pytest.raises(TypeError):
        # positonal only for a
        Test(a=1, b=2)

    assert Test(1, b=2).a == 1
    assert Test(1, 2).b == 2

    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        a: int = pytc.field(default=1, pos_only=True)
        b: int = pytc.field(default=2, pos_only=True)

    assert Test(1, 2).a == 1
    assert Test(1, 2).b == 2

    # keyword only
    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        a: int = pytc.field(default=1, kw_only=True)
        b: int = 2

    with pytest.raises(TypeError):
        Test(1, 2)

    with pytest.raises(TypeError):
        Test(1, b=2)

    assert Test(a=1, b=2).a == 1

    with pytest.raises(ValueError):
        # keyword only and pos_only are mutually exclusive
        @ft.partial(pytc.treeclass, leafwise=True)
        class Test:
            a: int = pytc.field(default=1, pos_only=True, kw_only=True)

    # pos_only, kw_only
    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        a: int = pytc.field(default=1, pos_only=True)
        b: int = pytc.field(default=2, kw_only=True)

    with pytest.raises(TypeError):
        Test(1, 2)

    assert Test(1, b=2).b == 2

    with pytest.raises(TypeError):
        Test(a=1, b=2)

    # test when init is False
    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        a: int = pytc.field(default=1, init=False, kw_only=True)
        b: int = 2

    with pytest.raises(TypeError):
        Test(1, 2)

    assert Test(b=2).a == 1

    with pytest.raises(TypeError):
        pytc.fields(1)

    assert len(pytc.fields(Test)) == 2

    @pytc.treeclass
    class Test:
        a: int = pytc.field(default=1)
        b: int = pytc.field(factory=lambda: 1)

        def __init__(self) -> None:
            pass

    with pytest.raises(AttributeError):
        Test()


def test_field_nondiff():
    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        a: int = 1
        b: int = 2
        c: int = 3

    test = Test()

    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        a: jnp.ndarray
        b: jnp.ndarray

        def __init__(self, a=jnp.array([1, 2, 3]), b=jnp.array([4, 5, 6])):
            self.a = a
            self.b = b

    test = Test()

    @ft.partial(pytc.treeclass, leafwise=True)
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

    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        a: jnp.ndarray
        b: jnp.ndarray

        def __init__(self, a=jnp.array([1, 2, 3]), b=jnp.array([4, 5, 6])):
            self.a = pytc.freeze(a)
            self.b = b

    test = Test()
    npt.assert_allclose(jtu.tree_leaves(test)[0], jnp.array([4, 5, 6]))


def test_hash():
    @ft.partial(pytc.treeclass, leafwise=True)
    class T:
        a: jnp.ndarray

    # with pytest.raises(TypeError):
    hash(T(jnp.array([1, 2, 3])))


def test_post_init():
    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        a: int = 1

        def __post_init__(self):
            self.a = 2

    t = Test()

    assert t.a == 2


def test_subclassing():
    @ft.partial(pytc.treeclass, leafwise=True)
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

    @ft.partial(pytc.treeclass, leafwise=True)
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
    @ft.partial(pytc.treeclass, leafwise=True)
    class L0:
        def __init__(self):
            self.a = 10
            self.b = 20

    t = L0()
    tt = copy.copy(t)

    assert tt.a == 10
    assert tt.b == 20


def test_copy():
    @ft.partial(pytc.treeclass, leafwise=True)
    class L0:
        a: int = 1
        b: int = 3
        c: int = 5

    t = L0()

    assert copy.copy(t).a == 1
    assert copy.copy(t).b == 3
    assert copy.copy(t).c == 5


def test_delattr():
    @ft.partial(pytc.treeclass, leafwise=True)
    class L0:
        a: int = 1
        b: int = 3
        c: int = 5

    t = L0()

    with pytest.raises(AttributeError):
        del t.a

    @ft.partial(pytc.treeclass, leafwise=True)
    class L2:
        a: int = 1

        def delete(self, name):
            del self.a

    t = L2()

    with pytest.raises(AttributeError):
        t.delete("a")


# def test_getattr():
#     with pytest.raises(AttributeError):

#         @ft.partial(pytc.treeclass, leafwise=True)
#         class L2:
#             a: int = 1

#             def __getattribute__(self, __name: str):
#                 pass

#     with pytest.raises(AttributeError):

#         @ft.partial(pytc.treeclass, leafwise=True)
#         class L3:
#             a: int = 1

#             def __getattribute__(self, __name: str):
#                 pass


# def test_treeclass_decorator_arguments():
#     @ft.partial(pytc.treeclass, leafwise=True)(order=False)
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

    @ft.partial(pytc.treeclass, leafwise=True)
    class Test1:
        a: int = 1

    @ft.partial(pytc.treeclass, leafwise=True)
    class Test2:
        a: jnp.ndarray

        def __init__(self) -> None:
            self.a = jnp.array([1, 2, 3])

    assert pytc.is_tree_equal(Test1(), Test2()) is False

    assert pytc.is_tree_equal(jnp.array([1, 2, 3]), jnp.array([1, 2, 3]))
    assert pytc.is_tree_equal(jnp.array([1, 2, 3]), jnp.array([1, 3, 3])) is False

    @ft.partial(pytc.treeclass, leafwise=True)
    class Test3:
        a: int = 1
        b: int = 2

    assert pytc.is_tree_equal(Test1(), Test3()) is False

    assert pytc.is_tree_equal(jnp.array([1, 2, 3]), 1) is False


def test_params():
    @ft.partial(pytc.treeclass, leafwise=True)
    class l0:
        a: int = 2

    @ft.partial(pytc.treeclass, leafwise=True)
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

        @ft.partial(pytc.treeclass, leafwise=True)
        class Test:
            a: list = [1, 2, 3]

    with pytest.raises(TypeError):

        @ft.partial(pytc.treeclass, leafwise=True)
        class Test:
            a: list = pytc.field(default=[1, 2, 3])


def test_non_class_input():
    with pytest.raises(TypeError):

        @ft.partial(pytc.treeclass, leafwise=True)
        def f(x):
            return x


def test_setattr_delattr():
    with pytest.raises(TypeError):

        @ft.partial(pytc.treeclass, leafwise=True)
        class Test:
            def __setattr__(self, k, v):
                pass

    with pytest.raises(TypeError):

        @ft.partial(pytc.treeclass, leafwise=True)
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

    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        a: int = pytc.field(callbacks=[instance_validator(int)])

    with pytest.raises(AssertionError):
        Test(a="a")

    assert Test(a=1).a == 1

    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        a: int = pytc.field(callbacks=[instance_validator((int, float))])

    assert Test(a=1).a == 1
    assert Test(a=1.0).a == 1.0

    with pytest.raises(AssertionError):
        Test(a="a")

    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        a: int = pytc.field(callbacks=[range_validator(0, 10)])

    with pytest.raises(AssertionError):
        Test(a=-1)

    assert Test(a=0).a == 0

    with pytest.raises(AssertionError):
        Test(a=11)

    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        a: int = pytc.field(callbacks=[range_validator(0, 10), instance_validator(int)])

    with pytest.raises(AssertionError):
        Test(a=-1)

    with pytest.raises(AssertionError):
        Test(a=11)

    with pytest.raises(TypeError):

        @ft.partial(pytc.treeclass, leafwise=True)
        class Test:
            a: int = pytc.field(callbacks=1)

    with pytest.raises(TypeError):

        @ft.partial(pytc.treeclass, leafwise=True)
        class Test:
            a: int = pytc.field(callbacks=[1])

    with pytest.raises(TypeError):

        @ft.partial(pytc.treeclass, leafwise=True)
        class Test:
            a: int = pytc.field(callbacks=[lambda: True])


def test_treeclass_frozen_field():
    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        a: int = pytc.field(callbacks=[pytc.freeze])

    t = Test(1)

    assert t.a == 1
    assert jtu.tree_leaves(t) == []


def test_key_error():
    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        a: int = pytc.field()

        def __init__(self) -> None:
            return

    with pytest.raises(AttributeError):
        Test()


def test_super():
    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        # a:int
        def __init__(self) -> None:
            super().__init__()

    Test()


def test_optional_attrs():
    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        def __repr__(self):
            return "a"

        def __or__(self, other):
            return 1

    tree = Test()

    assert tree.__repr__() == "a"
    assert tree | tree == 1


def test_override_repr():
    @ft.partial(pytc.treeclass, leafwise=True)
    class Test:
        def __repr__(self):
            return "a"

        def __add__(self, other):
            return 1

    assert Test().__repr__() == "a"
    assert Test() + Test() == 1


def test_optional_param():
    @ft.partial(pytc.treeclass, leafwise=False)
    class Test:
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
    Tree = type("Tree", (), {"__annotations__": annot, **leaves})
    Tree = pytc.treeclass(Tree)
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
        indices = (0,)
        return [*zip(names, types, indices)]

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
        metadatas = (None,)
        return [*zip(names, types, indices, metadatas)]

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
        metadatas = (None,)
        return [*zip(names, types, indices, metadatas)]

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
        metadatas = (None,)
        return [*zip(names, types, indices, metadatas)]

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
        metadatas = (None,)
        return [*zip(names, types, indices, metadatas)]

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

        @pytc.treeclass
        class Tree:
            self: int = pytc.field()
