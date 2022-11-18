import dataclasses
from dataclasses import field

import jax.numpy as jnp
import pytest

import pytreeclass as pytc
import pytreeclass._src.dataclass_util as dcu
from pytreeclass.tree_viz.utils import _node_count_and_size


def test_is_frozen():
    @pytc.treeclass
    class Test:
        a: jnp.ndarray = jnp.array([1.0, 2.0, 3.0])
        b: int = 1

    frozen = dcu.is_dataclass_fields_frozen(
        pytc.tree_filter(Test(), where=lambda _: True)
    )
    assert frozen is True
    assert dcu.is_dataclass_fields_frozen(1) is False


@pytc.treeclass
class Test:
    a: int = 10


a = Test()
b = (1, "s", 1.0, [2, 3])


@pytc.treeclass
class Test2:
    a: int = 1
    b: Test = Test()


def test_is_dataclass_leaf():
    assert dcu.is_dataclass_leaf(a) is True
    assert all(dcu.is_dataclass_leaf(bi) for bi in b) is False
    assert dcu.is_dataclass_leaf(Test2()) is False
    assert dcu.is_dataclass_leaf(Test2().b) is True


def test_is_dataclass_fields_frozen():
    @pytc.treeclass
    class Test:
        a: jnp.ndarray = jnp.array([1.0, 2.0, 3.0])
        b: int = 1

    assert dcu.is_dataclass_fields_frozen(Test()) is False
    frozen = pytc.tree_filter(Test(), where=lambda _: True)
    assert dcu.is_dataclass_fields_frozen(frozen) is True
    assert dcu.is_dataclass_fields_frozen([1]) is False


def test_is_dataclass_fields_nondiff():
    @pytc.treeclass
    class Test:
        a: jnp.ndarray = jnp.array([1.0, 2.0, 3.0])
        b: int = 1

    assert dcu.is_dataclass_fields_nondiff(Test()) is False
    assert dcu.is_dataclass_fields_nondiff(1) is False


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

    xx = pytc.tree_filter(test, where=lambda _: True)

    with pytest.raises(dataclasses.FrozenInstanceError):
        xx.a = 1

    assert _node_count_and_size("string") == (complex(0, 0), complex(0, 0))
