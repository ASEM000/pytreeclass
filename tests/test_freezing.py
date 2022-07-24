import jax

from pytreeclass import tree_viz, treeclass


def test_freezing_unfreezing():
    @treeclass
    class A:
        a: int
        b: int

    a = A(1, 2)
    b = a.freeze()
    c = a.unfreeze()

    assert jax.tree_leaves(a) == [1, 2]
    assert jax.tree_leaves(b) == []
    assert jax.tree_leaves(c) == [1, 2]

    @treeclass
    class A:
        a: int
        b: int

    @treeclass
    class B:
        c: int = 3
        d: A = A(1, 2)

    a = B()
    a.d = a.d.freeze()
    assert a.d.frozen is True
    assert (
        tree_viz.tree_diagram(a)
    ) == "B\n    ├── c=3\n    └── d=A\n        ├#─ a=1\n        └#─ b=2     "
