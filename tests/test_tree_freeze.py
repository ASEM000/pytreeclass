# Copyright 2023 pytreeclass authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import copy
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

import pytreeclass as tc
from pytreeclass._src.tree_util import tree_hash


def test_freeze_unfreeze():
    @tc.leafwise
    @tc.autoinit
    class A(tc.TreeClass):
        a: int
        b: int

    a = A(1, 2)
    b = a.at[...].apply(tc.freeze)
    c = (
        a.at["a"]
        .apply(tc.unfreeze, is_leaf=tc.is_frozen)
        .at["b"]
        .apply(tc.unfreeze, is_leaf=tc.is_frozen)
    )

    assert jtu.tree_leaves(a) == [1, 2]
    assert jtu.tree_leaves(b) == []
    assert jtu.tree_leaves(c) == [1, 2]

    assert tc.unfreeze(tc.freeze(1.0)) == 1.0

    @tc.leafwise
    @tc.autoinit
    class A(tc.TreeClass):
        a: int
        b: int

    @tc.leafwise
    @tc.autoinit
    class B(tc.TreeClass):
        c: int = 3
        d: A = A(1, 2)

    @tc.leafwise
    @tc.autoinit
    class A(tc.TreeClass):
        a: int
        b: int

    a = A(1, 2)
    b = jtu.tree_map(tc.freeze, a)
    c = (
        a.at["a"]
        .apply(tc.unfreeze, is_leaf=tc.is_frozen)
        .at["b"]
        .apply(tc.unfreeze, is_leaf=tc.is_frozen)
    )

    assert jtu.tree_leaves(a) == [1, 2]
    assert jtu.tree_leaves(b) == []
    assert jtu.tree_leaves(c) == [1, 2]

    @tc.leafwise
    @tc.autoinit
    class l0(tc.TreeClass):
        a: int = 0

    @tc.leafwise
    @tc.autoinit
    class l1(tc.TreeClass):
        b: l0 = l0()

    @tc.leafwise
    @tc.autoinit
    class l2(tc.TreeClass):
        c: l1 = l1()

    t = jtu.tree_map(tc.freeze, l2())

    assert jtu.tree_leaves(t) == []
    assert jtu.tree_leaves(t.c) == []
    assert jtu.tree_leaves(t.c.b) == []

    @tc.leafwise
    class l1(tc.TreeClass):
        def __init__(self):
            self.b = l0()

    @tc.leafwise
    class l2(tc.TreeClass):
        def __init__(self):
            self.c = l1()

    t = jtu.tree_map(tc.freeze, l2())
    assert jtu.tree_leaves(t.c) == []
    assert jtu.tree_leaves(t.c.b) == []


def test_freeze_errors():
    class T:
        pass

    @tc.leafwise
    @tc.autoinit
    class Test(tc.TreeClass):
        a: Any

    t = Test(T())

    # with pytest.raises(Exception):
    t.at[...].set(0)

    with pytest.raises(TypeError):
        t.at[...].apply(jnp.sin)

    t.at[...].reduce(jnp.sin)


def test_freeze_with_ops():
    @tc.leafwise
    @tc.autoinit
    class A(tc.TreeClass):
        a: int
        b: int

    @tc.leafwise
    @tc.autoinit
    class B(tc.TreeClass):
        c: int = 3
        d: A = A(1, 2)

    @tc.leafwise
    @tc.autoinit
    class Test(tc.TreeClass):
        a: int = 1
        b: float = tc.freeze(1.0)
        c: str = tc.freeze("test")

    t = Test()
    assert jtu.tree_leaves(t) == [1]

    with pytest.raises(AttributeError):
        jtu.tree_map(tc.freeze, t).a = 1

    with pytest.raises(AttributeError):
        jtu.tree_map(tc.unfreeze, t).a = 1

    hash(t)

    t = Test()
    jtu.tree_map(tc.unfreeze, t, is_leaf=tc.is_frozen)
    jtu.tree_map(tc.freeze, t)

    @tc.leafwise
    @tc.autoinit
    class Test(tc.TreeClass):
        a: int

    t = jtu.tree_map(tc.freeze, (Test(100)))

    with pytest.raises(LookupError):
        tc.is_tree_equal(t.at[...].set(0), t)

    with pytest.raises(LookupError):
        tc.is_tree_equal(t.at[...].apply(lambda x: x + 1), t)

    with pytest.raises(LookupError):
        tc.is_tree_equal(t.at[...].reduce(jnp.add, initializer=0), t)

    @tc.leafwise
    class Test(tc.TreeClass):
        def __init__(self, x):
            self.x = x

    t = Test(jnp.array([1, 2, 3]))
    assert tc.is_tree_equal(t.at[...].set(None), Test(x=None))

    class t0:
        a: int = 1

    class t1:
        a: int = t0()

    t = t1()


def test_freeze_diagram():
    @tc.leafwise
    @tc.autoinit
    class A(tc.TreeClass):
        a: int
        b: int

    @tc.leafwise
    @tc.autoinit
    class B(tc.TreeClass):
        c: int = 3
        d: A = A(1, 2)

    a = B()
    a = a.at["d"].set(tc.freeze(a.d))
    a = B()

    a = a.at["d"].set(tc.freeze(a.d))  # = a.d.freeze()


def test_freeze_mask():
    @tc.leafwise
    @tc.autoinit
    class Test(tc.TreeClass):
        a: int = 1
        b: int = 2
        c: float = 3.0

    t = Test()

    assert jtu.tree_leaves(jtu.tree_map(tc.freeze, t)) == []


def test_freeze_nondiff():
    @tc.leafwise
    @tc.autoinit
    class Test(tc.TreeClass):
        a: int = tc.freeze(1)
        b: str = "a"

    t = Test()

    assert jtu.tree_leaves(t) == ["a"]
    assert jtu.tree_leaves(jtu.tree_map(tc.freeze, t)) == []
    assert jtu.tree_leaves(
        (jtu.tree_map(tc.freeze, t)).at["b"].apply(tc.unfreeze, is_leaf=tc.is_frozen)
    ) == ["a"]

    @tc.leafwise
    @tc.autoinit
    class T0(tc.TreeClass):
        a: Test = Test()

    t = T0()

    assert jtu.tree_leaves(t) == ["a"]
    assert jtu.tree_leaves(jtu.tree_map(tc.freeze, t)) == []

    assert jtu.tree_leaves(t) == ["a"]
    assert jtu.tree_leaves(jtu.tree_map(tc.freeze, t)) == []


def test_freeze_nondiff_with_mask():
    @tc.leafwise
    @tc.autoinit
    class L0(tc.TreeClass):
        a: int = 1
        b: int = 2
        c: int = 3

    @tc.leafwise
    @tc.autoinit
    class L1(tc.TreeClass):
        a: int = 1
        b: int = 2
        c: int = 3
        d: L0 = L0()

    @tc.leafwise
    @tc.autoinit
    class L2(tc.TreeClass):
        a: int = 10
        b: int = 20
        c: int = 30
        d: L1 = L1()

    t = L2()
    t = t.at["d"]["d"]["a"].apply(tc.freeze)
    t = t.at["d"]["d"]["b"].apply(tc.freeze)

    assert jtu.tree_leaves(t) == [10, 20, 30, 1, 2, 3, 3]


def test_non_dataclass_input_to_freeze():
    assert jtu.tree_leaves(tc.freeze(1)) == []


def test_tree_mask():
    @tc.leafwise
    @tc.autoinit
    class l0(tc.TreeClass):
        x: int = 2
        y: int = 3

    @tc.leafwise
    @tc.autoinit
    class l1(tc.TreeClass):
        a: int = 1
        b: l0 = l0()

    tree = l1()

    assert jtu.tree_leaves(tree) == [1, 2, 3]
    assert jtu.tree_leaves(jtu.tree_map(tc.freeze, tree)) == []
    assert jtu.tree_leaves(jtu.tree_map(tc.freeze, tree)) == []
    assert jtu.tree_leaves(tree.at[...].apply(tc.freeze)) == []
    assert jtu.tree_leaves(tree.at[tree > 1].apply(tc.freeze)) == [1]
    assert jtu.tree_leaves(tree.at[tree == 1].apply(tc.freeze)) == [2, 3]
    assert jtu.tree_leaves(tree.at[tree < 1].apply(tc.freeze)) == [1, 2, 3]

    assert jtu.tree_leaves(tree.at["a"].apply(tc.freeze)) == [2, 3]
    assert jtu.tree_leaves(tree.at["b"].apply(tc.freeze)) == [1]
    assert jtu.tree_leaves(tree.at["b"]["x"].apply(tc.freeze)) == [1, 3]
    assert jtu.tree_leaves(tree.at["b"]["y"].apply(tc.freeze)) == [1, 2]


def test_tree_unmask():
    @tc.leafwise
    @tc.autoinit
    class l0(tc.TreeClass):
        x: int = 2
        y: int = 3

    @tc.leafwise
    @tc.autoinit
    class l1(tc.TreeClass):
        a: int = 1
        b: l0 = l0()

    tree = l1()

    frozen_tree = tree.at[...].apply(tc.freeze)
    assert jtu.tree_leaves(frozen_tree) == []

    mask = tree == tree
    unfrozen_tree = frozen_tree.at[mask].apply(tc.unfreeze, is_leaf=tc.is_frozen)  # fmt: skip
    assert jtu.tree_leaves(unfrozen_tree) == [1, 2, 3]

    mask = tree > 1
    unfrozen_tree = frozen_tree.at[mask].apply(tc.unfreeze, is_leaf=tc.is_frozen)  # fmt: skip
    assert jtu.tree_leaves(unfrozen_tree) == [2, 3]

    unfrozen_tree = frozen_tree.at["a"].apply(tc.unfreeze, is_leaf=tc.is_frozen)  # fmt: skip
    assert jtu.tree_leaves(unfrozen_tree) == [1]

    unfrozen_tree = frozen_tree.at["b"].apply(tc.unfreeze, is_leaf=tc.is_frozen)  # fmt: skip
    assert jtu.tree_leaves(unfrozen_tree) == [2, 3]


def test_tree_mask_unfreeze():
    @tc.leafwise
    @tc.autoinit
    class l0(tc.TreeClass):
        x: int = 2
        y: int = 3

    @tc.leafwise
    @tc.autoinit
    class l1(tc.TreeClass):
        a: int = 1
        b: l0 = l0()

    tree = l1()

    mask = tree == tree
    frozen_tree = tree.at[...].apply(tc.freeze)
    unfrozen_tree = frozen_tree.at[mask].apply(tc.unfreeze, is_leaf=tc.is_frozen)  # fmt: skip
    assert jtu.tree_leaves(unfrozen_tree) == [1, 2, 3]

    frozen_tree = tree.at["a"].apply(tc.freeze)
    unfrozen_tree = frozen_tree.at["a"].apply(tc.unfreeze, is_leaf=tc.is_frozen)  # fmt: skip
    assert jtu.tree_leaves(unfrozen_tree) == [1, 2, 3]


def test_freeze_nondiff_func():
    @tc.leafwise
    @tc.autoinit
    class Test(tc.TreeClass):
        a: int = 1.0
        b: int = 2
        c: int = 3
        act: Callable = jax.nn.tanh

        def __call__(self, x):
            return self.act(x + self.a)

    @jax.value_and_grad
    def loss_func(model):
        model = model.at[...].apply(tc.unfreeze, is_leaf=tc.is_frozen)
        return jnp.mean((model(1.0) - 0.5) ** 2)

    @jax.jit
    def update(model):
        value, grad = loss_func(model)
        return value, model - 1e-3 * grad

    model = Test()
    # Test(a=1.0,b=2,c=3,act=tanh(x))

    mask = jtu.tree_map(tc.is_nondiff, model)
    model = model.at[mask].apply(tc.freeze)
    # Test(a=1.0,*b=2,*c=3,*act=tanh(x))

    for _ in range(1, 20001):
        _, model = update(model)

    # print(model)
    # Test(a=-0.45068058,*b=2,*c=3,*act=tanh(x))
    assert model.a == pytest.approx(-0.45068058, 1e-5)


def test_wrapper():
    # only apply last wrapper
    assert hash((tc.freeze(1))) == tree_hash(1)

    # lhs = _HashableWrapper(1)
    # # test getter
    # assert lhs.__wrapped__ == 1
    # assert lhs.__wrapped__

    # # comparison with the wrapped object
    # assert lhs != 1
    # # hash of the wrapped object
    # assert hash(lhs) == tree_hash(1)

    # test immutability
    frozen_value = tc.freeze(1)

    with pytest.raises(AttributeError):
        frozen_value.__wrapped__ = 2

    assert tc.freeze(1) == tc.freeze(1)
    assert f"{tc.freeze(1)!r}" == "#1"

    wrapped = tc.freeze(1)

    with pytest.raises(AttributeError):
        delattr(wrapped, "__wrapped__")

    assert wrapped != 1


def test_tree_mask_tree_unmask():
    tree = [1, 2, 3.0]
    assert jtu.tree_leaves(tc.tree_mask(tree)) == [3.0]
    assert jtu.tree_leaves(tc.tree_unmask(tc.tree_mask(tree))) == [1, 2, 3.0]

    mask_func = lambda x: x < 2
    assert jtu.tree_leaves(tc.tree_mask(tree, mask_func)) == [2, 3.0]

    frozen_array = tc.tree_mask(jnp.ones((5, 5)), mask=lambda _: True)

    assert frozen_array == frozen_array
    assert not (frozen_array == tc.freeze(jnp.ones((5, 6))))
    assert not (frozen_array == tc.freeze(jnp.ones((5, 5)).astype(jnp.uint8)))
    assert hash(frozen_array) == hash(frozen_array)

    assert tc.freeze(tc.freeze(1)) == tc.freeze(1)

    assert tc.tree_mask({"a": 1}, mask={"a": True}) == {"a": tc.freeze(1)}

    with pytest.raises(ValueError):
        tc.tree_mask({"a": 1}, mask=1.0)

    assert copy.copy(tc.freeze(1)) == tc.freeze(1)

    with pytest.raises(NotImplementedError):
        tc.freeze(1) + 1
