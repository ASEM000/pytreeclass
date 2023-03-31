from __future__ import annotations

import jax
import jax.tree_util as jtu

from pytreeclass._src.tree_pprint import (
    _func_pprint,
    _hbox,
    _hstack,
    _node_pprint,
    _table,
    _vbox,
)


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
        _table([col1, col2], transpose=True)
        == "┌─┬────┐\n│1│3   │\n│ │    │\n├─┼────┤\n│2│4000│\n└─┴────┘"
    )


def test_hstack():
    assert _hstack(_hbox("a"), _vbox("b", "c")) == "┌─┬─┐\n│a│b│\n└─┼─┤\n  │c│\n  └─┘"


def test_func_pprint():
    def example(a: int, b=1, *c, d, e=2, **f) -> str:
        ...  # fmt: skip

    assert _func_pprint(example, 0, "str", 60, 0) == "example(a, b, *c, d, e, **f)"
    assert _func_pprint(lambda x: x, 0, "str", 60, 0) == "Lambda(x)"
    assert _func_pprint(jax.nn.relu, 0, "str", 60, 0) == "relu(*args, **kwargs)"
    assert (
        _node_pprint(jtu.Partial(jax.nn.relu), 0, "str", 60, 0)
        == "Partial(relu(*args, **kwargs))"
    )
    assert (
        _node_pprint(jtu.Partial(jax.nn.relu), 0, "str", 60, 0)
        == "Partial(relu(*args, **kwargs))"
    )
    assert (
        _func_pprint(jax.nn.initializers.he_normal, 0, "str", 60, 0)
        == "he_normal(in_axis, out_axis, batch_axis, dtype)"
    )

    assert _node_pprint(jax.jit(lambda x: x), 0, "repr", 60, 0) == "jit(Lambda(x))"
