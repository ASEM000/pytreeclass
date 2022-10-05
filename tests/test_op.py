import jax.numpy as jnp
import pytest

import pytreeclass as pytc
from pytreeclass._src.tree_op import _dispatched_op_tree_map

# @pytc.treeclass
# class Test:
#     a: float
#     b: float
#     c: float
#     name: str = pytc.nondiff_field()


def test_ops():
    @pytc.treeclass
    class Test:
        a: float
        b: float
        c: float
        name: str = pytc.field(nondiff=True)

    A = Test(10, 20, 30, ("A"))
    # binary operations

    assert (A + A) == Test(20, 40, 60, ("A"))
    assert (A - A) == Test(0, 0, 0, ("A"))
    # assert ((A["a"] + A) | A) == Test(20, 20, 30, ("A"))
    assert A.at[...].reduce(lambda x, y: x + jnp.sum(y)) == jnp.array(60)
    assert abs(A) == A

    @pytc.treeclass
    class Test:
        a: float
        b: float
        name: str = pytc.field(nondiff=True)

    A = Test(-10, 20, ("A"))

    # magic ops
    assert abs(A) == Test(10, 20, ("A"))
    assert A + A == Test(-20, 40, ("A"))
    assert A == A
    assert A // 2 == Test(-5, 10, ("A"))
    assert A / 2 == Test(-5.0, 10.0, ("A"))
    assert (A > A) == Test(False, False, ("A"))
    assert (A >= A) == Test(True, True, ("A"))
    assert (A <= A) == Test(True, True, ("A"))
    assert -A == Test(10, -20, ("A"))
    assert A * A == Test(100, 400, ("A"))
    assert A**A == Test((-10) ** (-10), 20**20, ("A"))
    assert A - A == Test(0, 0, ("A"))

    # unary operations
    assert abs(A) == Test(10, 20, ("A"))
    assert -A == Test(-10, -20, ("A"))
    assert +A == Test(10, 20, ("A"))
    assert ~A == Test(~10, ~20, ("A"))


def test_op_errors():
    @pytc.treeclass
    class Test:
        a: float
        b: float
        c: float
        name: str = pytc.field(nondiff=True)

    A = Test(10, 20, 30, ("A"))

    with pytest.raises(TypeError):
        A + "s"

    with pytest.raises(NotImplementedError):
        A == (1,)


def test_dispatched_tree_map():

    with pytest.raises(NotImplementedError):

        class A:
            ...

        _dispatched_op_tree_map(lambda x, y: x, 1, A())
