from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

import pytreeclass as pytc
from pytreeclass._src.tree_base import ImmutableTreeError
from pytreeclass._src.tree_freeze import _FrozenWrapper, _HashableWrapper


def test_freeze_unfreeze():
    @pytc.treeclass
    class A:
        a: int
        b: int

    a = A(1, 2)
    b = a.at[...].apply(pytc.tree_freeze)
    c = a.at["a"].apply(pytc.tree_unfreeze).at["b"].apply(pytc.tree_unfreeze)

    assert jtu.tree_leaves(a) == [1, 2]
    assert jtu.tree_leaves(b) == []
    assert jtu.tree_leaves(c) == [1, 2]

    assert pytc.tree_freeze(pytc.tree_freeze(1.0)).unwrap() == 1.0

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
    b = pytc.tree_freeze(a)
    c = a.at["a"].apply(pytc.tree_unfreeze).at["b"].apply(pytc.tree_unfreeze)

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

    t = pytc.tree_freeze(l2())

    assert jtu.tree_leaves(t) == []
    assert jtu.tree_leaves(t.c) == []
    assert jtu.tree_leaves(t.c.b) == []

    @pytc.treeclass
    class l1:
        def __init__(self):
            self.b = l0()

    @pytc.treeclass
    class l2:
        def __init__(self):
            self.c = l1()

    t = pytc.tree_freeze(l2())
    assert jtu.tree_leaves(t.c) == []
    assert jtu.tree_leaves(t.c.b) == []


def test_freeze_errors():
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


def test_freeze_with_ops():
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
        b: float = pytc.frozen(1.0)
        c: str = pytc.frozen("test")

    t = Test()
    assert jtu.tree_leaves(t) == [1]

    with pytest.raises(ImmutableTreeError):
        pytc.tree_freeze(t).a = 1

    with pytest.raises(ImmutableTreeError):
        pytc.tree_unfreeze(t).a = 1

    hash(t)

    t = Test()
    pytc.tree_unfreeze(t)
    pytc.tree_freeze(t)

    @pytc.treeclass
    class Test:
        a: int

    t = pytc.tree_freeze(Test(100))

    assert pytc.is_tree_equal(t.at[...].set(0), t)
    assert pytc.is_tree_equal(t.at[...].apply(lambda x: x + 1), t)
    assert pytc.is_tree_equal(t.at[...].reduce(jnp.sin), 0.0)

    @pytc.treeclass
    class Test:
        x: jnp.ndarray

        def __init__(self, x):
            self.x = x

    t = Test(jnp.array([1, 2, 3]))
    assert pytc.is_tree_equal(t.at[...].set(None), Test(x=None))

    @pytc.treeclass
    class t0:
        a: int = 1

    @pytc.treeclass
    class t1:
        a: int = t0()

    t = t1()


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
    a = a.at["d"].set(pytc.tree_freeze(a.d))
    a = B()

    a = a.at["d"].set(pytc.tree_freeze(a.d))  # = a.d.freeze()


def test_freeze_mask():
    @pytc.treeclass
    class Test:
        a: int = 1
        b: int = 2
        c: float = 3.0

    t = Test()

    assert jtu.tree_leaves(pytc.tree_freeze(t)) == []


def test_freeze_nondiff():
    @pytc.treeclass
    class Test:
        a: int = pytc.frozen(1)
        b: str = "a"

    t = Test()

    assert jtu.tree_leaves(t) == ["a"]
    assert jtu.tree_leaves(pytc.tree_freeze(t)) == []
    assert jtu.tree_leaves(
        (pytc.tree_freeze(t)).at["b"].apply(pytc.tree_unfreeze, is_leaf=pytc.is_frozen)
    ) == ["a"]

    @pytc.treeclass
    class T0:
        a: Test = Test()

    t = T0()

    assert jtu.tree_leaves(t) == ["a"]
    assert jtu.tree_leaves(pytc.tree_freeze(t)) == []

    assert jtu.tree_leaves(t) == ["a"]
    assert jtu.tree_leaves(pytc.tree_freeze(t)) == []


def test_freeze_nondiff_with_mask():
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
    t = t.at["d"].at["d"].at["a"].apply(pytc.tree_freeze)
    t = t.at["d"].at["d"].at["b"].apply(pytc.tree_freeze)

    assert jtu.tree_leaves(t) == [10, 20, 30, 1, 2, 3, 3]


def test_non_dataclass_input_to_freeze():
    assert jtu.tree_leaves(pytc.tree_freeze(1)) == []


def test_tree_freeze():
    @pytc.treeclass
    class l0:
        x: int = 2
        y: int = 3

    @pytc.treeclass
    class l1:
        a: int = 1
        b: l0 = l0()

    tree = l1()

    assert jtu.tree_leaves(tree) == [1, 2, 3]
    assert jtu.tree_leaves(pytc.tree_freeze(tree)) == []
    assert jtu.tree_leaves(jtu.tree_map(pytc.tree_freeze, tree)) == []
    assert jtu.tree_leaves(tree.at[...].apply(pytc.tree_freeze)) == []
    assert jtu.tree_leaves(tree.at[tree > 1].apply(pytc.tree_freeze)) == [1]
    assert jtu.tree_leaves(tree.at[tree == 1].apply(pytc.tree_freeze)) == [2, 3]
    assert jtu.tree_leaves(tree.at[tree < 1].apply(pytc.tree_freeze)) == [1, 2, 3]

    assert jtu.tree_leaves(tree.at["a"].apply(pytc.tree_freeze)) == [2, 3]
    assert jtu.tree_leaves(tree.at["b"].apply(pytc.tree_freeze)) == [1]
    assert jtu.tree_leaves(tree.at["b"].at["x"].apply(pytc.tree_freeze)) == [1, 3]
    assert jtu.tree_leaves(tree.at["b"].at["y"].apply(pytc.tree_freeze)) == [1, 2]


def test_tree_unfreeze():
    @pytc.treeclass
    class l0:
        x: int = 2
        y: int = 3

    @pytc.treeclass
    class l1:
        a: int = 1
        b: l0 = l0()

    tree = l1()

    frozen_tree = tree.at[...].apply(pytc.tree_freeze)
    assert jtu.tree_leaves(frozen_tree) == []

    mask = tree == tree
    unfrozen_tree = frozen_tree.at[mask].apply(pytc.tree_unfreeze, is_leaf=pytc.is_frozen)  # fmt: skip
    assert jtu.tree_leaves(unfrozen_tree) == [1, 2, 3]

    mask = tree > 1
    unfrozen_tree = frozen_tree.at[mask].apply(pytc.tree_unfreeze, is_leaf=pytc.is_frozen)  # fmt: skip
    assert jtu.tree_leaves(unfrozen_tree) == [2, 3]

    unfrozen_tree = frozen_tree.at["a"].apply(pytc.tree_unfreeze, is_leaf=pytc.is_frozen)  # fmt: skip
    assert jtu.tree_leaves(unfrozen_tree) == [1]

    unfrozen_tree = frozen_tree.at["b"].apply(pytc.tree_unfreeze, is_leaf=pytc.is_frozen)  # fmt: skip
    assert jtu.tree_leaves(unfrozen_tree) == [2, 3]


def test_tree_freeze_unfreeze():
    @pytc.treeclass
    class l0:
        x: int = 2
        y: int = 3

    @pytc.treeclass
    class l1:
        a: int = 1
        b: l0 = l0()

    tree = l1()

    mask = tree == tree
    frozen_tree = tree.at[...].apply(pytc.tree_freeze)
    unfrozen_tree = frozen_tree.at[mask].apply(pytc.tree_unfreeze, is_leaf=pytc.is_frozen)  # fmt: skip
    assert jtu.tree_leaves(unfrozen_tree) == [1, 2, 3]

    frozen_tree = tree.at["a"].apply(pytc.tree_freeze)
    unfrozen_tree = frozen_tree.at["a"].apply(pytc.tree_unfreeze, is_leaf=pytc.is_frozen)  # fmt: skip
    assert jtu.tree_leaves(unfrozen_tree) == [1, 2, 3]


def test_freeze_nondiff_func():
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
    # Test(a=1.0,b=2,c=3,act=tanh(x))

    mask = jtu.tree_map(pytc.is_nondiff, model)
    model = model.at[mask].apply(pytc.tree_freeze)
    # Test(a=1.0,*b=2,*c=3,*act=tanh(x))

    for _ in range(1, 20001):
        _, model = update(model)

    # print(model)
    # Test(a=-0.45068058,*b=2,*c=3,*act=tanh(x))
    assert model.a == pytest.approx(-0.45068058, 1e-5)


def test_wrapper():
    # only apply last wrapper
    assert hash((pytc.tree_freeze(1))) == hash(1)

    lhs = _HashableWrapper(1)
    # test getter
    assert lhs.__wrapped__ == 1
    assert lhs.__wrapped__

    # comparison with the wrapped object
    assert lhs != 1
    # hash of the wrapped object
    assert hash(lhs) == hash(1)

    # test immutability
    frozen_value = _FrozenWrapper(1)

    with pytest.raises(ValueError):
        frozen_value.__wrapped__ = 2

    assert _FrozenWrapper(1) == _FrozenWrapper(1)
    assert f"{_FrozenWrapper(1)!r}" == "#1"
