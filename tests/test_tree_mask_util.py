import jax.numpy as jnp
import numpy.testing as npt
import pytest

import pytreeclass as pytc
from pytreeclass.src.tree_mask_util import (
    is_inexact,
    is_inexact_array,
    logical_all,
    logical_and,
    logical_not,
    logical_or,
    where,
)
from pytreeclass.src.tree_util import is_treeclass_equal


def test_is_inexact_array():
    assert is_inexact_array([jnp.array(1.0), 1, 1.0, complex(1)]) == [
        True,
        False,
        False,
        False,
    ]


def test_is_inexact():
    assert is_inexact([jnp.array(1.0), 1, 1.0, complex(1)]) == [True, False, True, True]


def test_logical_not():
    npt.assert_allclose(
        logical_not([jnp.array([True, False, True, False])])[0],
        jnp.array([False, True, False, True]),
    )

    assert logical_not([True, False]) == [False, True]


def test_logical_or():

    assert logical_or([True, False], [True, False]) == [True, False]

    npt.assert_allclose(
        logical_or(
            [jnp.array([True, False, True, False])],
            [jnp.array([True, False, False, True])],
        )[0],
        jnp.array([True, False, True, True]),
    )

    with pytest.raises(ValueError):
        logical_or([True, jnp.array([1, 2, 3])], [True, False])


def test_logical_and():

    assert logical_and([True, False], [True, False]) == [True, False]

    npt.assert_allclose(
        logical_and(
            [jnp.array([True, False, True, False])],
            [jnp.array([True, False, False, True])],
        )[0],
        jnp.array([True, False, False, False]),
    )

    with pytest.raises(ValueError):
        logical_and([True, jnp.array([1, 2, 3])], [True, False])


def test_where():
    @pytc.treeclass
    class Test:
        a: int
        b: jnp.ndarray

    assert is_treeclass_equal(
        where(Test(1, jnp.array([1, 2, 3])) > 1, 100, 0),
        Test(0, jnp.array([0, 100, 100])),
    )


def test_all():
    @pytc.treeclass
    class Test:
        a: int
        b: jnp.ndarray
        c: bool
        d: bool

    assert jnp.array_equal(
        logical_all(Test(True, jnp.array([True, True, True]), True, True)), True
    )
    assert jnp.array_equal(
        logical_all(Test(False, jnp.array([True, True, True]), True, True)), False
    )
    assert jnp.array_equal(
        logical_all(Test(True, jnp.array([True, False, True]), True, True)), False
    )
    assert jnp.array_equal(
        logical_all(Test(True, jnp.array([True, False, True]), True, False)), False
    )
    assert jnp.array_equal(
        logical_all(Test(True, ([True, False, True]), True, False)), False
    )
    assert jnp.array_equal(
        logical_all(Test(True, (True, False, True), True, False)), False
    )
