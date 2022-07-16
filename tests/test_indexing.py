import jax
import jax.numpy as jnp
import pytest

from pytreeclass import treeclass
from pytreeclass.src.utils import is_treeclass_equal


@treeclass
class Test:
    a: float
    b: float
    c: float
    d: jnp.ndarray
    name: str


def test_getter_by_param():
    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), "A")

    B = A.at["a"].get()
    assert is_treeclass_equal(B, Test(10, None, None, None, "A"))

    B = A.at["a", "b"].get()
    assert is_treeclass_equal(B, Test(10, 20, None, None, "A"))

    B = A.at[""].get()
    assert is_treeclass_equal(B, Test(None, None, None, None, "A"))

    B = A.at["a", "b", "c"].get()
    assert is_treeclass_equal(B, Test(10, 20, 30, None, "A"))

    B = A.at["a", "b", "c", "d"].get()
    assert is_treeclass_equal(B, Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), "A"))


def test_getter_by_model():
    @treeclass
    class level1:
        a: int
        b: int
        c: int

    @treeclass
    class level2:
        d: level1
        e: level1

    A = level2(
        d=level1(a=1, b=10, c=jnp.array([1, 2, 3, 4, 5])),
        e=level1(a=2, b=20, c=jnp.array([-1, -2, -3, -4, -5])),
    )

    B = A.at[A > 0].get()
    C = level2(
        d=level1(a=1, b=10, c=jnp.array([1, 2, 3, 4, 5])),
        e=level1(a=2, b=20, c=jnp.array([])),
    )

    assert is_treeclass_equal(B, C)

    B = A.at[A == 0].get()
    C = level2(
        d=level1(a=None, b=None, c=jnp.array([])),
        e=level1(a=None, b=None, c=jnp.array([])),
    )

    assert is_treeclass_equal(B, C)

    with pytest.raises(ValueError):
        B = A.at[A].get()

    with pytest.raises(NotImplementedError):
        B = A.at[0].get()


def test_setter_by_param():
    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), "A")

    B = A.at["a"].set(0)
    assert is_treeclass_equal(B, Test(0, 20, 30, jnp.array([1, 2, 3, 4, 5]), "A"))

    B = A.at["a", "b"].set(0)
    assert is_treeclass_equal(B, Test(0, 0, 30, jnp.array([1, 2, 3, 4, 5]), "A"))

    B = A.at["a", "b", "c"].set(0)
    assert is_treeclass_equal(B, Test(0, 0, 0, jnp.array([1, 2, 3, 4, 5]), "A"))


def test_setter_by_model():
    @treeclass
    class level1:
        a: int
        b: int
        c: int

    @treeclass
    class level2:
        d: level1
        e: level1

    A = level2(
        d=level1(a=1, b=10, c=jnp.array([1, 2, 3, 4, 5])),
        e=level1(a=2, b=20, c=jnp.array([-1, -2, -3, -4, -5])),
    )

    B = A.at[A < 0].set(100)
    C = level2(
        d=level1(a=1, b=10, c=jnp.array([1, 2, 3, 4, 5])),
        e=level1(a=2, b=20, c=jnp.array([100, 100, 100, 100, 100])),
    )

    assert is_treeclass_equal(B, C)

    A = Test(10, 20, 30, jnp.array([1, 2, 3, 4, 5]), "A")

    with pytest.raises(ValueError):
        B = A.at[A].set(0)

    with pytest.raises(NotImplementedError):
        B = A.at[0].set(0)
