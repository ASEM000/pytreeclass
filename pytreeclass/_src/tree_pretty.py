# This script defines the base class for the tree repr/str/summary/tree_box.
# this base class handles visualization of the treeclass

from __future__ import annotations

import jax.numpy as jnp

from pytreeclass._src.misc import cached_method
from pytreeclass.tree_viz.tree_box import tree_box
from pytreeclass.tree_viz.tree_pprint import tree_diagram, tree_repr, tree_str
from pytreeclass.tree_viz.tree_summary import tree_summary


class _treePretty:
    """Base class for tree repr/str/summary/tree_box"""

    @cached_method
    def __repr__(self) -> str:
        """pretty print pytree instance"""
        # since the instance is immutable, the result of the tree_repr
        # should not change, thus we can cache it
        return tree_repr(self, width=60)

    @cached_method
    def __str__(self):
        """pretty print pytree instance"""
        # since the instance is immutable, the result of the tree_str
        # should not change, thus we can cache it
        return tree_str(self, width=60)

    def summary(self, array: jnp.ndarray = None) -> str:
        """print a summary of the pytree instance

        Example:
            >>> import pytreeclass as pytc
            >>> @pytc.treeclass
            ... class A:
            ...     a: int = 1
            ...     b: float = 2.0
            >>> a = A()
            >>> print(a.summary())
            ┌────┬─────┬───────┬────────┬──────┐
            │Name│Type │Param #│Size    │Config│
            ├────┼─────┼───────┼────────┼──────┤
            │a   │int  │0(1)   │0.00B   │a=1   │
            │    │     │       │(28.00B)│      │
            ├────┼─────┼───────┼────────┼──────┤
            │b   │float│1(0)   │24.00B  │b=2.0 │
            │    │     │       │(0.00B) │      │
            └────┴─────┴───────┴────────┴──────┘
            Total count :   1(1)
            Dynamic count : 1(1)
            Frozen count :  0(0)
            ----------------------------------------
            Total size :    24.00B(28.00B)
            Dynamic size :  24.00B(28.00B)
            Frozen size :   0.00B(0.00B)
            ========================================
        """
        return tree_summary(self, array)

    @cached_method
    def tree_diagram(self) -> str:
        """Print a diagram of the pytree instance

        Example:
            >>> import pytreeclass as pytc
            >>> @pytc.treeclass
            ... class A:
            ...     a: int = 1
            ...     b: float = 2.0
            >>> a = A()
            >>> print(a.tree_diagram())
            A
                ├── a=1
                └── b=2.0
        """
        return tree_diagram(self)

    def tree_box(self, array: jnp.ndarray = None) -> str:
        """keras-like `plot_model` subclassing paradigm"""
        return tree_box(self, array)
