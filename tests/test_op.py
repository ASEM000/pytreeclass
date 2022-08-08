import jax.numpy as jnp
import pytest

from pytreeclass import static_field, treeclass


@treeclass
class Test:
    a: float
    b: float
    c: float
    name: str = static_field()


def test_ops():
    @treeclass
    class Test:
        a: float
        b: float
        c: float
        name: str = static_field(metadata={"static": True})

    A = Test(10, 20, 30, "A")
    # binary operations

    assert (A + A) == Test(20, 40, 60, "A")
    assert (A - A) == Test(0, 0, 0, "A")
    # assert ((A["a"] + A) | A) == Test(20, 20, 30, "A")
    # assert A.reduce_mean() == jnp.array(60)
    assert abs(A) == A

    @treeclass
    class Test:
        a: int
        b: int
        name: str = static_field()

    A = Test(-10, 20, "A")

    # magic ops
    assert abs(A) == Test(10, 20, "A")
    assert A + A == Test(-20, 40, "A")
    assert A == A
    assert A // 2 == Test(-5, 10, "A")
    assert A / 2 == Test(-5.0, 10.0, "A")
    assert (A > A) == Test(False, False, "A")
    assert (A >= A) == Test(True, True, "A")
    assert (A <= A) == Test(True, True, "A")
    assert -A == Test(10, -20, "A")
    assert A * A == Test(100, 400, "A")
    assert A**A == Test((-10) ** (-10), 20**20, "A")
    assert A - A == Test(0, 0, "A")

    with pytest.raises(NotImplementedError):
        A + "s"

    with pytest.raises(NotImplementedError):
        A == (1,)

    assert abs(A) == Test(10, 20, "A")

    # numpy ops
    A = Test(a=jnp.array([-10, -10]), b=1, name="A")


def test_asdict():
    A = Test(10, 20, 30, "A")
    assert A.asdict() == {"a": 10, "b": 20, "c": 30, "name": "A"}
