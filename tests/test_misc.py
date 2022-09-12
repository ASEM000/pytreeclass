from typing import Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import pytest

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

    model = filter_nondiff(model)
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

    t = t.at["d"].at["d"].set(filter_nondiff(t.d.d, where=L0(a=True, b=True, c=False)))
    assert jtu.tree_leaves(t) == [10, 20, 30, 1, 2, 3, 3]

    with pytest.raises(TypeError):
        filter_nondiff(t, where=t.d)
