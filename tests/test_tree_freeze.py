from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

import pytreeclass as pytc
from pytreeclass._src.tree_freeze import _HashableWrapper, tree_hash


def test_freeze_unfreeze():
    class A(pytc.TreeClass, leafwise=True):
        a: int
        b: int

    a = A(1, 2)
    b = a.at[...].apply(pytc.freeze)
    c = (
        a.at["a"]
        .apply(pytc.unfreeze, is_leaf=pytc.is_frozen)
        .at["b"]
        .apply(pytc.unfreeze, is_leaf=pytc.is_frozen)
    )

    assert jtu.tree_leaves(a) == [1, 2]
    assert jtu.tree_leaves(b) == []
    assert jtu.tree_leaves(c) == [1, 2]

    assert pytc.unfreeze(pytc.freeze(1.0)) == 1.0

    class A(pytc.TreeClass, leafwise=True):
        a: int
        b: int

    class B(pytc.TreeClass, leafwise=True):
        c: int = 3
        d: A = A(1, 2)

    class A(pytc.TreeClass, leafwise=True):
        a: int
        b: int

    a = A(1, 2)
    b = jtu.tree_map(pytc.freeze, a)
    c = (
        a.at["a"]
        .apply(pytc.unfreeze, is_leaf=pytc.is_frozen)
        .at["b"]
        .apply(pytc.unfreeze, is_leaf=pytc.is_frozen)
    )

    assert jtu.tree_leaves(a) == [1, 2]
    assert jtu.tree_leaves(b) == []
    assert jtu.tree_leaves(c) == [1, 2]

    class l0(pytc.TreeClass, leafwise=True):
        a: int = 0

    class l1(pytc.TreeClass, leafwise=True):
        b: l0 = l0()

    class l2(pytc.TreeClass, leafwise=True):
        c: l1 = l1()

    t = jtu.tree_map(pytc.freeze, l2())

    assert jtu.tree_leaves(t) == []
    assert jtu.tree_leaves(t.c) == []
    assert jtu.tree_leaves(t.c.b) == []

    class l1(pytc.TreeClass, leafwise=True):
        def __init__(self):
            self.b = l0()

    class l2(pytc.TreeClass, leafwise=True):
        def __init__(self):
            self.c = l1()

    t = jtu.tree_map(pytc.freeze, l2())
    assert jtu.tree_leaves(t.c) == []
    assert jtu.tree_leaves(t.c.b) == []


def test_freeze_errors():
    class T:
        pass

    class Test(pytc.TreeClass, leafwise=True):
        a: Any

    t = Test(T())

    # with pytest.raises(Exception):
    t.at[...].set(0)

    with pytest.raises(TypeError):
        t.at[...].apply(jnp.sin)

    t.at[...].reduce(jnp.sin)


def test_freeze_with_ops():
    class A(pytc.TreeClass, leafwise=True):
        a: int
        b: int

    class B(pytc.TreeClass, leafwise=True):
        c: int = 3
        d: A = A(1, 2)

    class Test(pytc.TreeClass, leafwise=True):
        a: int = 1
        b: float = pytc.freeze(1.0)
        c: str = pytc.freeze("test")

    t = Test()
    assert jtu.tree_leaves(t) == [1]

    with pytest.raises(AttributeError):
        jtu.tree_map(pytc.freeze, t).a = 1

    with pytest.raises(AttributeError):
        jtu.tree_map(pytc.unfreeze, t).a = 1

    hash(t)

    t = Test()
    jtu.tree_map(pytc.unfreeze, t, is_leaf=pytc.is_frozen)
    jtu.tree_map(pytc.freeze, t)

    class Test(pytc.TreeClass, leafwise=True):
        a: int

    t = jtu.tree_map(pytc.freeze, (Test(100)))

    assert pytc.is_tree_equal(t.at[...].set(0), t)
    assert pytc.is_tree_equal(t.at[...].apply(lambda x: x + 1), t)
    assert pytc.is_tree_equal(t.at[...].reduce(jnp.sin, initializer=0), 0.0)

    class Test(pytc.TreeClass, leafwise=True):
        x: jnp.ndarray

        def __init__(self, x):
            self.x = x

    t = Test(jnp.array([1, 2, 3]))
    assert pytc.is_tree_equal(t.at[...].set(None), Test(x=None))

    class t0:
        a: int = 1

    class t1:
        a: int = t0()

    t = t1()


def test_freeze_diagram():
    class A(pytc.TreeClass, leafwise=True):
        a: int
        b: int

    class B(pytc.TreeClass, leafwise=True):
        c: int = 3
        d: A = A(1, 2)

    a = B()
    a = a.at["d"].set(pytc.freeze(a.d))
    a = B()

    a = a.at["d"].set(pytc.freeze(a.d))  # = a.d.freeze()


def test_freeze_mask():
    class Test(pytc.TreeClass, leafwise=True):
        a: int = 1
        b: int = 2
        c: float = 3.0

    t = Test()

    assert jtu.tree_leaves(jtu.tree_map(pytc.freeze, t)) == []


def test_freeze_nondiff():
    class Test(pytc.TreeClass, leafwise=True):
        a: int = pytc.freeze(1)
        b: str = "a"

    t = Test()

    assert jtu.tree_leaves(t) == ["a"]
    assert jtu.tree_leaves(jtu.tree_map(pytc.freeze, t)) == []
    assert jtu.tree_leaves(
        (jtu.tree_map(pytc.freeze, t))
        .at["b"]
        .apply(pytc.unfreeze, is_leaf=pytc.is_frozen)
    ) == ["a"]

    class T0(pytc.TreeClass, leafwise=True):
        a: Test = Test()

    t = T0()

    assert jtu.tree_leaves(t) == ["a"]
    assert jtu.tree_leaves(jtu.tree_map(pytc.freeze, t)) == []

    assert jtu.tree_leaves(t) == ["a"]
    assert jtu.tree_leaves(jtu.tree_map(pytc.freeze, t)) == []


def test_freeze_nondiff_with_mask():
    class L0(pytc.TreeClass, leafwise=True):
        a: int = 1
        b: int = 2
        c: int = 3

    class L1(pytc.TreeClass, leafwise=True):
        a: int = 1
        b: int = 2
        c: int = 3
        d: L0 = L0()

    class L2(pytc.TreeClass, leafwise=True):
        a: int = 10
        b: int = 20
        c: int = 30
        d: L1 = L1()

    t = L2()
    t = t.at["d"].at["d"].at["a"].apply(pytc.freeze)
    t = t.at["d"].at["d"].at["b"].apply(pytc.freeze)

    assert jtu.tree_leaves(t) == [10, 20, 30, 1, 2, 3, 3]


def test_non_dataclass_input_to_freeze():
    assert jtu.tree_leaves(pytc.freeze(1)) == []


def test_tree_freeze():
    class l0(pytc.TreeClass, leafwise=True):
        x: int = 2
        y: int = 3

    class l1(pytc.TreeClass, leafwise=True):
        a: int = 1
        b: l0 = l0()

    tree = l1()

    assert jtu.tree_leaves(tree) == [1, 2, 3]
    assert jtu.tree_leaves(jtu.tree_map(pytc.freeze, tree)) == []
    assert jtu.tree_leaves(jtu.tree_map(pytc.freeze, tree)) == []
    assert jtu.tree_leaves(tree.at[...].apply(pytc.freeze)) == []
    assert jtu.tree_leaves(tree.at[tree > 1].apply(pytc.freeze)) == [1]
    assert jtu.tree_leaves(tree.at[tree == 1].apply(pytc.freeze)) == [2, 3]
    assert jtu.tree_leaves(tree.at[tree < 1].apply(pytc.freeze)) == [1, 2, 3]

    assert jtu.tree_leaves(tree.at["a"].apply(pytc.freeze)) == [2, 3]
    assert jtu.tree_leaves(tree.at["b"].apply(pytc.freeze)) == [1]
    assert jtu.tree_leaves(tree.at["b"].at["x"].apply(pytc.freeze)) == [1, 3]
    assert jtu.tree_leaves(tree.at["b"].at["y"].apply(pytc.freeze)) == [1, 2]


def test_tree_unfreeze():
    class l0(pytc.TreeClass, leafwise=True):
        x: int = 2
        y: int = 3

    class l1(pytc.TreeClass, leafwise=True):
        a: int = 1
        b: l0 = l0()

    tree = l1()

    frozen_tree = tree.at[...].apply(pytc.freeze)
    assert jtu.tree_leaves(frozen_tree) == []

    mask = tree == tree
    unfrozen_tree = frozen_tree.at[mask].apply(pytc.unfreeze, is_leaf=pytc.is_frozen)  # fmt: skip
    assert jtu.tree_leaves(unfrozen_tree) == [1, 2, 3]

    mask = tree > 1
    unfrozen_tree = frozen_tree.at[mask].apply(pytc.unfreeze, is_leaf=pytc.is_frozen)  # fmt: skip
    assert jtu.tree_leaves(unfrozen_tree) == [2, 3]

    unfrozen_tree = frozen_tree.at["a"].apply(pytc.unfreeze, is_leaf=pytc.is_frozen)  # fmt: skip
    assert jtu.tree_leaves(unfrozen_tree) == [1]

    unfrozen_tree = frozen_tree.at["b"].apply(pytc.unfreeze, is_leaf=pytc.is_frozen)  # fmt: skip
    assert jtu.tree_leaves(unfrozen_tree) == [2, 3]


def test_tree_freeze_unfreeze():
    class l0(pytc.TreeClass, leafwise=True):
        x: int = 2
        y: int = 3

    class l1(pytc.TreeClass, leafwise=True):
        a: int = 1
        b: l0 = l0()

    tree = l1()

    mask = tree == tree
    frozen_tree = tree.at[...].apply(pytc.freeze)
    unfrozen_tree = frozen_tree.at[mask].apply(pytc.unfreeze, is_leaf=pytc.is_frozen)  # fmt: skip
    assert jtu.tree_leaves(unfrozen_tree) == [1, 2, 3]

    frozen_tree = tree.at["a"].apply(pytc.freeze)
    unfrozen_tree = frozen_tree.at["a"].apply(pytc.unfreeze, is_leaf=pytc.is_frozen)  # fmt: skip
    assert jtu.tree_leaves(unfrozen_tree) == [1, 2, 3]


def test_freeze_nondiff_func():
    class Test(pytc.TreeClass, leafwise=True):
        a: int = 1.0
        b: int = 2
        c: int = 3
        act: Callable = jax.nn.tanh

        def __call__(self, x):
            return self.act(x + self.a)

    @jax.value_and_grad
    def loss_func(model):
        model = model.at[...].apply(pytc.unfreeze, is_leaf=pytc.is_frozen)
        return jnp.mean((model(1.0) - 0.5) ** 2)

    @jax.jit
    def update(model):
        value, grad = loss_func(model)
        return value, model - 1e-3 * grad

    model = Test()
    # Test(a=1.0,b=2,c=3,act=tanh(x))

    mask = jtu.tree_map(pytc.is_nondiff, model)
    model = model.at[mask].apply(pytc.freeze)
    # Test(a=1.0,*b=2,*c=3,*act=tanh(x))

    for _ in range(1, 20001):
        _, model = update(model)

    # print(model)
    # Test(a=-0.45068058,*b=2,*c=3,*act=tanh(x))
    assert model.a == pytest.approx(-0.45068058, 1e-5)


def test_wrapper():
    # only apply last wrapper
    assert hash((pytc.freeze(1))) == tree_hash(1)

    lhs = _HashableWrapper(1)
    # test getter
    assert lhs.__wrapped__ == 1
    assert lhs.__wrapped__

    # comparison with the wrapped object
    assert lhs != 1
    # hash of the wrapped object
    assert hash(lhs) == tree_hash(1)

    # test immutability
    frozen_value = pytc.freeze(1)

    with pytest.raises(AttributeError):
        frozen_value.__wrapped__ = 2

    assert pytc.freeze(1) == pytc.freeze(1)
    assert f"{pytc.freeze(1)!r}" == "#1"

    wrapped = pytc.freeze(1)

    with pytest.raises(AttributeError):
        delattr(wrapped, "__wrapped__")

    assert wrapped != 1
