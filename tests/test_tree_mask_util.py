import jax.numpy as jnp
import numpy.testing as npt

from pytreeclass.src.tree_mask_util import (
    is_inexact,
    is_inexact_array,
    logical_and,
    logical_not,
    logical_or,
)


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


def test_logical_and():

    assert logical_and([True, False], [True, False]) == [True, False]

    npt.assert_allclose(
        logical_and(
            [jnp.array([True, False, True, False])],
            [jnp.array([True, False, False, True])],
        )[0],
        jnp.array([True, False, False, False]),
    )
