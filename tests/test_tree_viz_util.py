from __future__ import annotations

import jax
import jax.tree_util as jtu

from pytreeclass.tree_viz.box_drawing import _hbox, _vbox
from pytreeclass.tree_viz.node_pprint import (
    _format_node_diagram,
    _format_node_repr,
    _format_node_str,
    _func_repr,
)


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
    assert (_format_node_diagram(jtu.Partial(jax.nn.relu)) == "Partial(relu(*args,**kwargs))")  # fmt: skip
    assert (_func_repr(jax.nn.initializers.he_normal) == "he_normal(in_axis,out_axis,batch_axis,dtype)")  # fmt: skip
