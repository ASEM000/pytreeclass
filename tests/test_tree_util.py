from dataclasses import field

import jax.numpy as jnp
import pytest

import pytreeclass as pytc
from pytreeclass._src.tree_util import tree_freeze
from pytreeclass.tree_viz.utils import _node_count_and_size
from pytreeclass.treeclass import ImmutableInstanceError


def test_is_frozen():
    @pytc.treeclass
    class Test:
        a: jnp.ndarray = jnp.array([1.0, 2.0, 3.0])
        b: int = 1

    assert pytc.is_treeclass_frozen(tree_freeze(Test())) is True
    assert pytc.is_treeclass_frozen(1) is False


@pytc.treeclass
class Test:
    a: int = 10


a = Test()
b = (1, "s", 1.0, [2, 3])


@pytc.treeclass
class Test2:
    a: int = 1
    b: Test = Test()


def test_is_treeclass():
    assert pytc.is_treeclass(a) is True
    assert all(pytc.is_treeclass(bi) for bi in b) is False


def test_is_treeclass_leaf():
    assert pytc.is_treeclass_leaf(a) is True
    assert all(pytc.is_treeclass_leaf(bi) for bi in b) is False
    assert pytc.is_treeclass_leaf(Test2()) is False
    assert pytc.is_treeclass_leaf(Test2().b) is True


def test_is_treeclass_frozen():
    @pytc.treeclass
    class Test:
        a: jnp.ndarray = jnp.array([1.0, 2.0, 3.0])
        b: int = 1

    assert pytc.is_treeclass_frozen(Test()) is False
    assert pytc.is_treeclass_frozen(tree_freeze(Test())) is True
    assert pytc.is_treeclass_frozen([1]) is False


def test_is_treeclass_nondiff():
    @pytc.treeclass
    class Test:
        a: jnp.ndarray = jnp.array([1.0, 2.0, 3.0])
        b: int = 1

    assert pytc.is_treeclass_nondiff(Test()) is False
    assert pytc.is_treeclass_nondiff(1) is False


def test__node_count_and_size():
    @pytc.treeclass
    class Test:
        a: jnp.ndarray = jnp.array([1.0, 2.0, 3.0])
        b: int = 1

    t = Test()
    assert _node_count_and_size(t.b) == (complex(0, 1), complex(0, 28))
    assert _node_count_and_size(t.a) == (complex(3, 0), complex(12, 0))

    assert _node_count_and_size(jnp.array([1, 2, 3, 4, 5])) == (
        complex(0, 5),
        complex(0, 20),
    )
    assert _node_count_and_size(3.0) == (complex(1), complex(24))

    @pytc.treeclass
    class x:
        a: int
        b: float
        c: complex
        d: tuple
        e: list = field(default_factory=list)
        f: dict = field(default_factory=dict)
        g: set = field(default_factory=set)

    test = x(1, 1.0, complex(1, 1), (1, 2), [1, 2], {"a": 1}, {1})

    assert hash(test)

    xx = tree_freeze(test)

    with pytest.raises(ImmutableInstanceError):
        xx.a = 1

    assert _node_count_and_size("string") == (complex(0, 0), complex(0, 0))
