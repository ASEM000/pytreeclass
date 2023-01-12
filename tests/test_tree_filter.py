import dataclasses
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

import pytreeclass as pytc


def test_filter_unfilter():
    @pytc.treeclass
    class A:
        a: int
        b: int

    a = A(1, 2)
    b = pytc.tree_filter(a, where=a == a)
    c = pytc.tree_unfilter(a)
    d = pytc.tree_unfilter(a, where=a == a)

    assert jtu.tree_leaves(a) == [1, 2]
    assert jtu.tree_leaves(b) == []
    assert jtu.tree_leaves(c) == [1, 2]
    assert jtu.tree_leaves(d) == [1, 2]

    @pytc.treeclass
    class A:
        a: int
        b: int

    @pytc.treeclass
    class B:
        c: int = 3
        d: A = A(1, 2)

    @pytc.treeclass
    class A:
        a: int
        b: int

    a = A(1, 2)
    b = pytc.tree_filter(a, where=a == a)
    c = pytc.tree_unfilter(a)

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

    t = pytc.tree_filter(l2(), where=l2() == l2())

    assert jtu.tree_leaves(t) == []
    assert jtu.tree_leaves(t.c) == []
    assert jtu.tree_leaves(t.c.b) == []

    tt = pytc.tree_unfilter(t, where=lambda _: True)
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

    t = pytc.tree_filter(l2(), where=l2() == l2())
    assert jtu.tree_leaves(t.c) == []
    assert jtu.tree_leaves(t.c.b) == []


def test_filter_errors():
    class T:
        pass

    @pytc.treeclass
    class Test:
        a: Any

    t = Test(T())

    # with pytest.raises(Exception):
    t.at[...].set(0)

    with pytest.raises(TypeError):
        t.at[...].apply(jnp.sin)

    with pytest.raises(TypeError):
        t.at[...].reduce(jnp.sin)


def test_filter_with_ops():
    @pytc.treeclass
    class A:
        a: int
        b: int

    @pytc.treeclass
    class B:
        c: int = 3
        d: A = A(1, 2)

    @pytc.treeclass
    class Test:
        a: int = 1
        b: float = pytc.field(nondiff=True, default=(1.0))
        c: str = pytc.field(nondiff=True, default=("test"))

    t = Test()
    assert jtu.tree_leaves(t) == [1]

    with pytest.raises(dataclasses.FrozenInstanceError):
        pytc.tree_filter(t, where=t == t).a = 1

    with pytest.raises(dataclasses.FrozenInstanceError):
        pytc.tree_unfilter(t).a = 1

    hash(t)

    t = Test()
    pytc.tree_unfilter(t, where=t == t)
    pytc.tree_filter(t, where=t == t)

    @pytc.treeclass
    class Test:
        a: int

    t = pytc.tree_filter(Test(100), where=Test(100) == Test(100))

    assert pytc.is_treeclass_equal(t.at[...].set(0), t)
    assert pytc.is_treeclass_equal(t.at[...].apply(lambda x: x + 1), t)
    assert pytc.is_treeclass_equal(t.at[...].reduce(jnp.sin), 0.0)

    @pytc.treeclass
    class Test:
        x: jnp.ndarray

        def __init__(self, x):
            self.x = x

    t = Test(jnp.array([1, 2, 3]))
    assert pytc.is_treeclass_equal(t.at[...].set(None), Test(x=None))

    @pytc.treeclass
    class t0:
        a: int = 1

    @pytc.treeclass
    class t1:
        a: int = t0()

    t = t1()
    assert pytc.is_treeclass_equal(
        pytc.tree_unfilter(pytc.tree_filter(t, where=t == t)), t
    )

    @pytc.treeclass
    class t2:
        a: int = t1()

    assert pytc.is_treeclass_equal(
        pytc.tree_unfilter(pytc.tree_filter(t2(), where=t2() == t2())), t2()
    )


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
    a = a.at["d"].set(pytc.tree_filter(a.d, where=lambda _: True))
    a = B()

    a = a.at["d"].set(pytc.tree_filter(a.d, where=lambda _: True))  # = a.d.freeze()


def test_filter_mask():
    @pytc.treeclass
    class Test:
        a: int = 1
        b: int = 2
        c: float = 3.0

    t = Test()

    assert jtu.tree_leaves(pytc.tree_filter(t, where=t == t)) == []


def test_filter_nondiff():
    @pytc.treeclass
    class Test:
        a: int = pytc.field(nondiff=True, default=1)
        b: str = "a"

    t = Test()

    assert jtu.tree_leaves(t) == ["a"]
    assert jtu.tree_leaves(pytc.tree_filter(t)) == []
    assert jtu.tree_leaves(pytc.tree_unfilter(pytc.tree_filter(t))) == ["a"]
    assert pytc.is_treeclass_equal(t, pytc.tree_unfilter(pytc.tree_filter(t)))

    @pytc.treeclass
    class T0:
        a: Test = Test()

    t = T0()

    assert jtu.tree_leaves(t) == ["a"]
    assert jtu.tree_leaves(pytc.tree_filter(t)) == []
    assert jtu.tree_leaves(pytc.tree_unfilter(pytc.tree_filter(t))) == ["a"]
    assert pytc.is_treeclass_equal(t, pytc.tree_unfilter(pytc.tree_filter(t)))

    assert jtu.tree_leaves(t) == ["a"]
    assert jtu.tree_leaves(pytc.tree_filter(t, where=t == t)) == []
    assert jtu.tree_leaves(pytc.tree_unfilter(pytc.tree_filter(t, where=t == t))) == [
        "a"
    ]
    assert pytc.is_treeclass_equal(
        t, pytc.tree_unfilter(pytc.tree_filter(t, where=t == t))
    )


def test_filter_nondiff_func():
    @pytc.treeclass
    class Test:
        a: int = 1.0
        b: int = 2
        c: int = 3
        act: Callable = jax.nn.tanh

        def __call__(self, x):
            return self.act(x + self.a)

    @jax.value_and_grad
    def loss_func(model):
        return jnp.mean((model(1.0) - 0.5) ** 2)

    @jax.jit
    def update(model):
        value, grad = loss_func(model)
        return value, model - 1e-3 * grad

    model = Test()
    print(model)
    # Test(a=1.0,b=2,c=3,act=tanh(x))

    model = pytc.tree_filter(model)
    # print(f"{model!r}")
    # Test(a=1.0,*b=2,*c=3,*act=tanh(x))

    for _ in range(1, 20001):
        value, model = update(model)

    # print(model)
    # Test(a=-0.45068058,*b=2,*c=3,*act=tanh(x))
    assert model.a == pytest.approx(-0.45068058, 1e-5)


def test_filter_nondiff_with_mask():
    @pytc.treeclass
    class L0:
        a: int = 1
        b: int = 2
        c: int = 3

    @pytc.treeclass
    class L1:
        a: int = 1
        b: int = 2
        c: int = 3
        d: L0 = L0()

    @pytc.treeclass
    class L2:
        a: int = 10
        b: int = 20
        c: int = 30
        d: L1 = L1()

    t = L2()

    t = (
        t.at["d"]
        .at["d"]
        .set(pytc.tree_filter(t.d.d, where=L0(a=True, b=True, c=False)))
    )
    assert jtu.tree_leaves(t) == [10, 20, 30, 1, 2, 3, 3]

    with pytest.raises(TypeError):
        pytc.tree_filter(t, where=t.d)


def test_non_dataclass_input_to_filter():
    with pytest.raises(TypeError):
        pytc.tree_filter(1)

    with pytest.raises(TypeError):
        pytc.tree_unfilter(1)
