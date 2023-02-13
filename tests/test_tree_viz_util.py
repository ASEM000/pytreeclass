from __future__ import annotations

import jax
import jax.tree_util as jtu

from pytreeclass.tree_viz.box_drawing import _hbox, _vbox
from pytreeclass.tree_viz.tree_pprint import _func_pprint, _node_pprint
from pytreeclass.tree_viz.tree_viz_util import _format_count, _format_size

# import pytest


def test__vbox():

    assert _vbox("a", " a", "a ") == "┌──┐\n│a │\n├──┤\n│ a│\n├──┤\n│a │\n└──┘"


def test__hbox():
    assert _hbox("a", "b", "c") == "┌─┬─┬─┐\n│a│b│c│\n└─┴─┴─┘\n"


def test_func_pprint():
    def example(a: int, b=1, *c, d, e=2, **f) -> str:
        ...  # fmt: skip

    assert _func_pprint(example) == "example(a, b, *c, d, e, **f)"
    assert _func_pprint(lambda x: x) == "Lambda(x)"
    assert _func_pprint(jax.nn.relu) == "relu(*args, **kwargs)"
    assert (_node_pprint(jtu.Partial(jax.nn.relu)) == "Partial(relu(*args, **kwargs))")  # fmt: skip
    assert (
        _node_pprint(jtu.Partial(jax.nn.relu), kind="str")
        == "Partial(relu(*args, **kwargs))"
    )
    assert (
        _func_pprint(jax.nn.initializers.he_normal)
        == "he_normal(in_axis, out_axis, batch_axis, dtype)"
    )


def test_format_count():
    assert _format_count(complex(1000, 2)) == "1,000(2)"
    assert _format_count(complex(1000, 0)) == "1,000(0)"
    assert _format_count((1000)) == "1,000"

    assert _format_size(1000) == "1000.00B"
    assert _format_size(complex(1000, 2)) == "1000.00B(2.00B)"
