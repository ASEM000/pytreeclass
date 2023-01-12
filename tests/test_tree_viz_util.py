from __future__ import annotations

import dataclasses as dc

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

import pytreeclass as pytc
from pytreeclass.tree_viz.box_drawing import _hbox, _vbox
from pytreeclass.tree_viz.node_pprint import (
    _format_node_repr,
    _format_node_str,
    _func_repr,
)
from pytreeclass.tree_viz.tree_summary import _format_count, _format_size
from pytreeclass.tree_viz.utils import _node_count_and_size


def test__vbox():

    assert _vbox("a", " a", "a ") == "┌──┐\n│a │\n├──┤\n│ a│\n├──┤\n│a │\n└──┘"


def test__hbox():
    assert _hbox("a", "b", "c") == "┌─┬─┬─┐\n│a│b│c│\n└─┴─┴─┘\n"


def test_func_repr():
    def example(a: int, b=1, *c, d, e=2, **f) -> str:
        ...  # fmt: skip

    assert _func_repr(example) == "example(a,b,*c,d,e,**f)"
    assert _func_repr(lambda x: x) == "Lambda(x)"
    assert _func_repr(jax.nn.relu) == "relu(*args,**kwargs)"
    assert (_format_node_repr(jtu.Partial(jax.nn.relu)) == "Partial(relu(*args,**kwargs))")  # fmt: skip
    assert _format_node_str(jtu.Partial(jax.nn.relu)) == "Partial(relu(*args,**kwargs))"
    assert (_func_repr(jax.nn.initializers.he_normal) == "he_normal(in_axis,out_axis,batch_axis,dtype)")  # fmt: skip


def test_format_count():
    assert _format_count(complex(1000, 2)) == "1,000(2)"
    assert _format_count(complex(1000, 0)) == "1,000(0)"
    assert _format_count((1000)) == "1,000"

    assert _format_size(1000) == "1000.00B"
    assert _format_size(complex(1000, 2)) == "1000.00B(2.00B)"


@pytc.treeclass
class Test:
    a: int = 10


a = Test()
b = (1, "s", 1.0, [2, 3])


@pytc.treeclass
class Test2:
    a: int = 1
    b: Test = Test()


def test_node_count_and_size():
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
        e: list = dc.field(default_factory=list)
        f: dict = dc.field(default_factory=dict)
        g: set = dc.field(default_factory=set)

    test = x(1, 1.0, complex(1, 1), (1, 2), [1, 2], {"a": 1}, {1})

    assert hash(test)

    xx = pytc.tree_filter(test, where=lambda _: True)

    with pytest.raises(dc.FrozenInstanceError):
        xx.a = 1

    assert _node_count_and_size("string") == (complex(0, 1), complex(0, 55))
