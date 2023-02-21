# import jax.numpy as jnp
# import pytest

# from pytreeclass._src.tree_decorator import (
#     validate_array,
#     validate_range,
#     validate_selection,
#     validate_type,
# )


# def test_validate_array():

#     x = jnp.ones([1, 2, 3])

#     with pytest.raises(TypeError):
#         validate_array((1.0, 2.0))(x)
#     with pytest.raises(ValueError):
#         validate_array((1, 2, 3, 4))(x)
#     with pytest.raises(ValueError):
#         validate_array((1, 2, None, None))(x)

#     with pytest.raises(ValueError):
#         validate_array((1, 2, 4))(x)

#     validate_array((None, None, None))(x)
#     validate_array((1, None, None))(x)
#     validate_array((1, 2, None))(x)
#     validate_array((1, 2, 3))(x)
#     validate_array((1, ...))(x)
#     validate_array((1, 2, ...))(x)
#     validate_array((1, 2, 3, ...))(x)
#     validate_array((..., 3))(x)

#     x = jnp.ones([1, 2, 3], dtype=jnp.float32)
#     validate_array((..., 3), dtype=jnp.float32)(x)

#     x = jnp.ones([1, 2, 3], dtype=jnp.uint8)
#     validate_array((..., 3), dtype=jnp.uint8)(x)

#     with pytest.raises(TypeError):
#         validate_array((..., 3), dtype=jnp.float32)(x)

#     with pytest.raises(ValueError):
#         validate_array((..., ...))(x)

#     with pytest.raises(TypeError):
#         validate_array((2, 2, 3))(1)


# def test_validate_range():
#     validate_range(0, 1)(0.5)
#     validate_range(0, 1)(0)  # 0 is in the range
#     validate_range(0, 1)(1)  # 1 is in the range

#     with pytest.raises(ValueError):
#         validate_range(0, 1)(2)

#     with pytest.raises(ValueError):
#         validate_range(0, 1)(-1)

#     with pytest.raises(ValueError):
#         validate_range(0, 1)([1, 2])


# def test_validate_type():
#     validate_type(int)(1)
#     validate_type(int)(0)
#     validate_type(int)(-1)
#     validate_type(int)(2**64)

#     validate_type(float)(1.0)

#     validate_type(str)("hello")
#     validate_type(str)("")

#     validate_type(bool)(True)
#     validate_type(bool)(False)

#     validate_type((int, float))(1)
#     validate_type((int, float))(1.0)

#     with pytest.raises(TypeError):
#         validate_type(int)("hello")

#     with pytest.raises(TypeError):
#         validate_type(float)("hello")

#     with pytest.raises(TypeError):
#         validate_type(str)(1)

#     with pytest.raises(TypeError):
#         validate_type(bool)(1)

#     with pytest.raises(TypeError):
#         validate_type((int, float))("hello")


# def test_validate_selection():

#     validate_selection((1, 2, 3))(1)
#     validate_selection((1, 2, 3))(2)
#     validate_selection(1)(1)

#     with pytest.raises(ValueError):
#         validate_selection((1, 2, 3))(0)
