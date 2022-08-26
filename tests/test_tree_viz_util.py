from __future__ import annotations

import jax

from pytreeclass.src import tree_viz_util


def test__vbox():

    assert (
        tree_viz_util._vbox("a", " a", "a ")
        == "┌──┐\n│a │\n├──┤\n│ a│\n├──┤\n│a │\n└──┘"
    )


def test__hbox():
    assert tree_viz_util._hbox("a", "b", "c") == "┌─┬─┬─┐\n│a│b│c│\n└─┴─┴─┘\n"


def test_func_repr():
    def example(a: int, b=1, *c, d, e=2, **f) -> str:
        ...  # fmt: skip

    assert tree_viz_util._func_repr(example) == "example(a,b,*c,d,e,**f)"
    assert tree_viz_util._func_repr(lambda x: x) == "Lambda(x)"
    assert tree_viz_util._func_repr(jax.nn.relu) == "relu(*args,**kwargs)"
    assert (
        tree_viz_util._func_repr(jax.nn.initializers.he_normal)
        == "he_normal(in_axis,out_axis,batch_axis,dtype)"
    )
