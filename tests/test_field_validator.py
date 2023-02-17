import jax.numpy as jnp
import pytest

from pytreeclass._src.tree_decorator import (
    enum_validator,
    invert_validator,
    range_validator,
    shape_validator,
    type_validator,
)


def test_shape_validator():

    x = jnp.ones([1, 2, 3])

    with pytest.raises(TypeError):
        shape_validator((1.0, 2.0))(x)
    with pytest.raises(ValueError):
        shape_validator((1, 2, 3, 4))(x)
    with pytest.raises(ValueError):
        shape_validator((1, 2, None, None))(x)

    shape_validator((None, None, None))(x)
    shape_validator((1, None, None))(x)
    shape_validator((1, 2, None))(x)
    shape_validator((1, 2, 3))(x)
    shape_validator((1, ...))(x)
    shape_validator((1, 2, ...))(x)
    shape_validator((1, 2, 3, ...))(x)
    shape_validator((..., 3))(x)

    with pytest.raises(ValueError):
        shape_validator((..., ...))(x)


def test_range_validator():
    range_validator(0, 1)(0.5)
    range_validator(0, 1)(0)  # 0 is in the range
    range_validator(0, 1)(1)  # 1 is in the range

    with pytest.raises(ValueError):
        range_validator(0, 1)(2)

    with pytest.raises(ValueError):
        range_validator(0, 1)(-1)

    with pytest.raises(ValueError):
        range_validator(0, 1)([1, 2])


def test_type_validator():
    type_validator(int)(1)
    type_validator(int)(0)
    type_validator(int)(-1)
    type_validator(int)(2**64)

    type_validator(float)(1.0)

    type_validator(str)("hello")
    type_validator(str)("")

    type_validator(bool)(True)
    type_validator(bool)(False)

    type_validator((int, float))(1)
    type_validator((int, float))(1.0)

    with pytest.raises(TypeError):
        type_validator(int)("hello")

    with pytest.raises(TypeError):
        type_validator(float)("hello")

    with pytest.raises(TypeError):
        type_validator(str)(1)

    with pytest.raises(TypeError):
        type_validator(bool)(1)

    with pytest.raises(TypeError):
        type_validator((int, float))("hello")


def test_enum_validator():

    enum_validator((1, 2, 3))(1)
    enum_validator((1, 2, 3))(2)
    enum_validator(1)(1)

    with pytest.raises(ValueError):
        enum_validator((1, 2, 3))(0)


def test_invert_validator():
    invert_validator(range_validator(0, 1))(2)
    invert_validator(range_validator(0, 1))(-1)

    with pytest.raises(Exception):
        invert_validator(range_validator(0, 1))(0.5)

    with pytest.raises(Exception):
        invert_validator(range_validator(0, 1))(0)

    invert_validator(type_validator(int))("hello")
    invert_validator(type_validator(int))(1.0)

    with pytest.raises(Exception):
        invert_validator(type_validator(int))(1)

    invert_validator(enum_validator((1, 2, 3)))(0)

    with pytest.raises(Exception):
        invert_validator(enum_validator((1, 2, 3)))(1)

    invert_validator(shape_validator((1, 2, 3)))(jnp.ones([1, 2, 4]))

    with pytest.raises(Exception):
        invert_validator(shape_validator((1, 2, 3)))(jnp.ones([1, 2, 3]))
