import jax.tree_util as jtu

import pytreeclass as pytc
from pytreeclass._src.misc import cached_method, filter_nondiff, unfilter_nondiff
from pytreeclass._src.tree_util import is_treeclass_equal


def test_filter_nondiff():
    @pytc.treeclass
    class Test:
        a: int = pytc.static_field(default=1)
        b: str = "a"

    t = Test()

    assert jtu.tree_leaves(t) == ["a"]
    assert jtu.tree_leaves(filter_nondiff(t)) == []
    assert jtu.tree_leaves(unfilter_nondiff(filter_nondiff(t))) == ["a"]
    assert is_treeclass_equal(t, unfilter_nondiff(filter_nondiff(t)))


def test_cached_method():
    class Test:
        @cached_method
        def a(self):
            return 1

    t = Test()
    assert t.a() == 1
    assert t.a() == 1

    class Test:
        @cached_method
        def a(self):
            return 2

    t = Test()
    assert t.a() == 2
    assert t.a() == 2
