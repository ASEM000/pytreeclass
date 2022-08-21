# import jax
import jax.numpy as jnp
import jax.tree_util as jtu

# import numpy as np
import numpy.testing as npt

import pytreeclass as pytc
from pytreeclass import static_field

# import pytest


def test_static():
    @pytc.treeclass
    class Test:
        a: int = 1
        b: int = 2
        c: int = 3

    test = Test()

    assert jtu.tree_leaves(test.at[...].static()) == []
    assert jtu.tree_leaves(test.at[test > 1].static()) == [1]
    assert jtu.tree_leaves(test.at[test < 2].static()) == [2, 3]

    assert jtu.tree_leaves(test.at[test == int].static()) == []
    assert jtu.tree_leaves(test.at[test == "a"].static()) == [2, 3]
    assert jtu.tree_leaves(test.at[(test == "a") | (test == "b")].static()) == [3]

    assert jtu.tree_leaves(test.at[test == {"a": 1}].static()) == [1, 2, 3]

    @pytc.treeclass
    class Test:
        a: jnp.ndarray = jnp.array([1, 2, 3])
        b: jnp.ndarray = jnp.array([4, 5, 6])

    test = Test()

    assert jtu.tree_leaves(test.at[...].static()) == []
    npt.assert_allclose(
        jtu.tree_leaves(test.at[test > 1].static())[0], jnp.array([1, 2, 3])
    )
    assert jtu.tree_leaves(test.at[test > 0].static()) == []

    @pytc.treeclass
    class Test:
        a: jnp.ndarray = static_field(default=jnp.array([1, 2, 3]))
        b: jnp.ndarray = static_field(default=jnp.array([4, 5, 6]))

    test = Test()

    assert jtu.tree_leaves(test) == []

    @pytc.treeclass
    class Test:
        a: jnp.ndarray = static_field(default=jnp.array([1, 2, 3]))
        b: jnp.ndarray = jnp.array([4, 5, 6])

    test = Test()
    npt.assert_allclose(jtu.tree_leaves(test)[0], jnp.array([4, 5, 6]))