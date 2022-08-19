import jax.numpy as jnp
import pytest

from pytreeclass import treeclass
from pytreeclass.src.tree_util import is_treeclass_equal, static_value


@treeclass
class Test:
    a: float
    b: float
    c: float
    name: str


def test_ops():
    @treeclass
    class Test:
        a: float
        b: float
        c: float
        name: str

    A = Test(10, 20, 30, static_value("A"))
    # binary operations

    assert (A + A) == Test(20, 40, 60, static_value("A"))
    assert (A - A) == Test(0, 0, 0, static_value("A"))
    # assert ((A["a"] + A) | A) == Test(20, 20, 30 ,static_value("A"))
    # assert A.reduce_mean() == jnp.array(60)
    assert abs(A) == A

    @treeclass
    class Test:
        a: int
        b: int
        name: str

    A = Test(-10, 20, static_value("A"))

    # magic ops
    assert abs(A) == Test(10, 20, static_value("A"))
    assert A + A == Test(-20, 40, static_value("A"))
    assert A == A
    assert A // 2 == Test(-5, 10, static_value("A"))
    assert A / 2 == Test(-5.0, 10.0, static_value("A"))
    assert (A > A) == Test(False, False, static_value("A"))
    assert (A >= A) == Test(True, True, static_value("A"))
    assert (A <= A) == Test(True, True, static_value("A"))
    assert -A == Test(10, -20, static_value("A"))
    assert A * A == Test(100, 400, static_value("A"))
    assert A**A == Test((-10) ** (-10), 20**20, static_value("A"))
    assert A - A == Test(0, 0, static_value("A"))

    with pytest.raises(TypeError):
        A + "s"

    with pytest.raises(NotImplementedError):
        A == (1,)

    assert abs(A) == Test(10, 20, static_value("A"))

    # numpy ops
    A = Test(a=jnp.array([-10, -10]), b=1, name="A")


def test_asdict():
    A = Test(10, 20, 30, static_value("A"))
    assert A.asdict() == {"a": 10, "b": 20, "c": 30, "name": "A"}


def test_or():
    @treeclass
    class test:
        a: jnp.ndarray
        b: jnp.ndarray

    x = test(jnp.array([]), jnp.array([4, 5, 6]))
    y = test(jnp.array([1, 2, 3]), jnp.array([]))

    assert is_treeclass_equal((x | y), test(jnp.array([1, 2, 3]), jnp.array([4, 5, 6])))

    # with pytest.raises(ValueError):
    x = test(jnp.array([]), jnp.array([]))
    y = test(jnp.array([1, 2, 3]), None)
    assert is_treeclass_equal(x | y, test(jnp.array([1, 2, 3]), jnp.array([])))
