import jax.tree_util as jtu

import pytreeclass as pytc
from pytreeclass.src.misc import filter_nondiff, unfilter_nondiff


def test_filter_nondiff():
    @pytc.treeclass
    class Test:
        a: int = pytc.static_field(default=1)
        b: str = "a"

    t = Test()

    assert jtu.tree_leaves(t) == ["a"]
    assert jtu.tree_leaves(filter_nondiff(t)) == []
    assert jtu.tree_leaves(unfilter_nondiff(filter_nondiff(t))) == ["a"]
