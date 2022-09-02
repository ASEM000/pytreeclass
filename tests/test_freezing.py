from typing import Any

import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

import pytreeclass as pytc
from pytreeclass.src.tree_base import ImmutableInstanceError
from pytreeclass.src.tree_util import is_treeclass_equal


def test_freezing_unfreezing():
    @pytc.treeclass
    class A:
        a: int
        b: int

    a = A(1, 2)
    b = a.at[...].freeze()
    c = a.at[...].unfreeze()

    assert jtu.tree_leaves(a) == [1, 2]
    assert jtu.tree_leaves(b) == []
    assert jtu.tree_leaves(c) == [1, 2]

    @pytc.treeclass
    class A:
        a: int
        b: int

    @pytc.treeclass
    class B:
        c: int = 3
        d: A = A(1, 2)

    @pytc.treeclass(field_only=True)
    class A:
        a: int
        b: int

    a = A(1, 2)
    b = a.at[...].freeze()
    c = a.at[...].unfreeze()

    assert jtu.tree_leaves(a) == [1, 2]
    assert jtu.tree_leaves(b) == []
    assert jtu.tree_leaves(c) == [1, 2]

    @pytc.treeclass
    class l0:
        a: int = 0

    @pytc.treeclass
    class l1:
        b: l0 = l0()

    @pytc.treeclass
    class l2:
        c: l1 = l1()

    t = l2().at[...].freeze()

    assert jtu.tree_leaves(t) == []
    assert jtu.tree_leaves(t.c) == []
    assert jtu.tree_leaves(t.c.b) == []

    tt = t.at[...].unfreeze()
    assert jtu.tree_leaves(tt) != []
    assert jtu.tree_leaves(tt.c) != []
    assert jtu.tree_leaves(tt.c.b) != []

    @pytc.treeclass
    class l1:
        def __init__(self):
            self.b = l0()

    @pytc.treeclass
    class l2:
        def __init__(self):
            self.c = l1()

    t = l2().at[...].freeze()
    assert jtu.tree_leaves(t.c) == []
    assert jtu.tree_leaves(t.c.b) == []


def test_freezing_errors():
    class T:
        pass

    @pytc.treeclass(field_only=False)
    class Test:
        a: Any

    t = Test(T())

    with pytest.raises(NotImplementedError):
        t.at[...].set(0)

    with pytest.raises(NotImplementedError):
        t.at[...].apply(jnp.sin)

    with pytest.raises(NotImplementedError):
        t.at[...].reduce(jnp.sin)


def test_freezing_with_ops():
    @pytc.treeclass
    class A:
        a: int
        b: int

    @pytc.treeclass
    class B:
        c: int = 3
        d: A = A(1, 2)

    @pytc.treeclass(field_only=False)
    class Test:
        a: int = 1
        b: float = pytc.static_field(default=(1.0))
        c: str = "test"

    t = Test()

    with pytest.raises(ImmutableInstanceError):
        t.at[...].freeze().a = 1

    @pytc.treeclass(field_only=True)
    class Test:
        a: int = 1
        b: float = pytc.static_field(default=(1.0))
        c: str = pytc.static_field(default=("test"))

    t = Test()
    assert jtu.tree_leaves(t) == [1]

    with pytest.raises(ImmutableInstanceError):
        t.at[...].freeze().a = 1

    with pytest.raises(ImmutableInstanceError):
        t.at[...].unfreeze().a = 1

    hash(t)

    t = Test()
    t.at[...].unfreeze()
    t.at[...].freeze()
    assert t.frozen is False

    @pytc.treeclass
    class Test:
        a: int

    t = Test(100).at[...].freeze()

    assert is_treeclass_equal(t.at[...].set(0), t)
    assert is_treeclass_equal(t.at[...].apply(lambda x: x + 1), t)
    assert is_treeclass_equal(t.at[...].reduce(jnp.sin), 0.0)

    @pytc.treeclass
    class Test:
        x: jnp.ndarray

        def __init__(self, x):
            self.x = x

    t = Test(jnp.array([1, 2, 3]))
    assert is_treeclass_equal(t.at[...].set(None), Test(x=None))

    @pytc.treeclass
    class t0:
        a: int = 1

    @pytc.treeclass
    class t1:
        a: int = t0()

    t = t1()
    assert is_treeclass_equal(t.at["a"].freeze().at["a"].unfreeze(), t)


def test_freeze_diagram():
    @pytc.treeclass
    class A:
        a: int
        b: int

    @pytc.treeclass
    class B:
        c: int = 3
        d: A = A(1, 2)

    a = B()
    a = a.at["d"].freeze()
    assert a.d.frozen is True
    assert (
        pytc.tree_viz.tree_diagram(a)
    ) == "B\n    ├── c=3\n    └#─ d=A\n        ├#─ a=1\n        └#─ b=2     "
    assert (
        a.tree_diagram()
    ) == "B\n    ├── c=3\n    └#─ d=A\n        ├#─ a=1\n        └#─ b=2     "

    a = B()
    a = a.at["d"].freeze()  # = a.d.freeze()
    assert a.d.frozen is True
    assert (
        pytc.tree_viz.tree_diagram(a)
    ) == "B\n    ├── c=3\n    └#─ d=A\n        ├#─ a=1\n        └#─ b=2     "
