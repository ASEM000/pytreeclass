from __future__ import annotations

import jax
import jax.tree_util as jtu
import pytest

from pytreeclass._src.tree_pprint import (
    _format_count,
    _format_size,
    _func_pprint,
    _hbox,
    _hstack,
    _node_pprint,
    _slice_pprint,
    _table,
    _vbox,
)

# import pytest


def test_vbox():
    assert _vbox("a", " a", "a ") == "┌──┐\n│a │\n├──┤\n│ a│\n├──┤\n│a │\n└──┘"
    assert _vbox("a", "b") == "┌─┐\n│a│\n├─┤\n│b│\n└─┘"


def test_hbox():
    assert _hbox("a", "b", "c") == "┌─┬─┬─┐\n│a│b│c│\n└─┴─┴─┘"
    assert _hbox("a") == "┌─┐\n│a│\n└─┘"


def test_table():
    col1 = ["1\n", "2"]
    col2 = ["3", "4000"]
    assert (
        _table([col1, col2])
        == "┌─┬────┐\n│1│3   │\n│ │    │\n├─┼────┤\n│2│4000│\n└─┴────┘"
    )


def test_hstack():
    assert _hstack(_hbox("a"), _vbox("b", "c")) == "┌─┬─┐\n│a│b│\n└─┼─┤\n  │c│\n  └─┘"


def test_func_pprint():
    def example(a: int, b=1, *c, d, e=2, **f) -> str:
        ...  # fmt: skip

    assert _func_pprint(example, 0, "str", 60) == "example(a, b, *c, d, e, **f)"
    assert _func_pprint(lambda x: x, 0, "str", 60) == "Lambda(x)"
    assert _func_pprint(jax.nn.relu, 0, "str", 60) == "relu(*args, **kwargs)"
    assert (
        _node_pprint(jtu.Partial(jax.nn.relu), 0, "str", 60)
        == "Partial(relu(*args, **kwargs))"
    )
    assert (
        _node_pprint(jtu.Partial(jax.nn.relu), 0, "str", 60)
        == "Partial(relu(*args, **kwargs))"
    )
    assert (
        _func_pprint(jax.nn.initializers.he_normal, 0, "str", 60)
        == "he_normal(in_axis, out_axis, batch_axis, dtype)"
    )


def test_format_count():
    assert _format_count(complex(1000, 2)) == "1,000(2)"
    assert _format_count(complex(1000, 0)) == "1,000(0)"
    assert _format_count((1000)) == "1,000"

    assert _format_size(1000) == "1000.00B"
    assert _format_size(complex(1000, 2)) == "1000.00B(2.00B)"

    with pytest.raises(TypeError):
        _format_count("a")


def test_slice_pprint():
    assert _slice_pprint(slice(0, 1, 1), 0, "str", 60) == "[0]"
    assert _slice_pprint(slice(0, 2, 1), 0, "str", 60) == "[0:2]"
    assert _slice_pprint(slice(0, 2, 2), 0, "str", 60) == "[0:2:2]"
    assert _slice_pprint(slice(None, 1, 2), 0, "str", 60) == "[:1:2]"
    assert _slice_pprint(slice(1, None, 2), 0, "str", 60) == "[1::2]"
