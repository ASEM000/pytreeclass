# import jax
import jax.numpy as jnp
import jax.tree_util as jtu

# import numpy as np
import numpy.testing as npt

import pytreeclass as pytc

# from pytreeclass import nondiff_field

# import pytest


def test_static():
    @pytc.treeclass
    class Test:
        a: int = 1
        b: int = 2
        c: int = 3

    test = Test()

    @pytc.treeclass
    class Test:
        a: jnp.ndarray = jnp.array([1, 2, 3])
        b: jnp.ndarray = jnp.array([4, 5, 6])

    test = Test()

    @pytc.treeclass
    class Test:
        a: jnp.ndarray = pytc.field(nondiff=True, default=jnp.array([1, 2, 3]))
        b: jnp.ndarray = pytc.field(nondiff=True, default=jnp.array([4, 5, 6]))

    test = Test()

    assert jtu.tree_leaves(test) == []

    @pytc.treeclass
    class Test:
        a: jnp.ndarray = pytc.field(nondiff=True, default=jnp.array([1, 2, 3]))
        b: jnp.ndarray = jnp.array([4, 5, 6])

    test = Test()
    npt.assert_allclose(jtu.tree_leaves(test)[0], jnp.array([4, 5, 6]))
