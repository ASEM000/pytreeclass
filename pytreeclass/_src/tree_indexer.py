# this script defines the tree_indexer class e.g. `.at` method for pytree

from __future__ import annotations

import copy
from collections.abc import Callable
from typing import Any, NamedTuple

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from pytreeclass._src.tree_freeze import _call_context

PyTree = Any
EllipsisType = type(Ellipsis)


def _is_leaf_bool(node: Any) -> bool:
    if hasattr(node, "dtype"):
        return node.dtype == "bool"
    return isinstance(node, bool)


def _check_valid_mask_leaf(where: Any):
    if not _is_leaf_bool(where) and where is not None:
        raise TypeError(f"All tree leaves must be boolean.Found {(where)}")
    return where


def _at_get_pytree(tree: PyTree, where: PyTree, is_leaf: Callable[[Any], bool]):
    def lhs_get(lhs: Any, where: Any):
        """Get pytree node value"""
        where = _check_valid_mask_leaf(where)
        if isinstance(lhs, (jnp.ndarray, np.ndarray)):
            return lhs[jnp.where(where)]
        return lhs if where else None

    return jtu.tree_map(lhs_get, tree, where, is_leaf=is_leaf)


def _at_set_pytree(
    tree: PyTree,
    where: PyTree,
    set_value: bool | int | float | complex | jnp.ndarray,
    is_leaf: Callable[[Any], bool],
):
    def lhs_set(lhs: Any, where: Any, set_value: Any):
        """Set pytree node value."""
        # fuse the boolean check here
        where = _check_valid_mask_leaf(where)
        if isinstance(lhs, (jnp.ndarray, np.ndarray)):
            if jnp.isscalar(set_value):
                return jnp.where(where, set_value, lhs)
            return set_value if jnp.all(where) else lhs
        return set_value if (where is True) else lhs

    if isinstance(set_value, type(tree)) and (
        jtu.tree_structure(tree, is_leaf=is_leaf)
        == jtu.tree_structure(set_value, is_leaf=is_leaf)
    ):
        # set_value leaf is set to tree leaf according to where leaf
        # for example lhs_tree.at[where].set(rhs_tree) will set rhs_tree leaves to lhs_tree leaves
        return jtu.tree_map(lhs_set, tree, where, set_value, is_leaf=is_leaf)

    # set_value is broadcasted to tree leaves
    # for example tree.at[where].set(1) will set all tree leaves to 1
    partial_lhs_set = lambda lhs, where: lhs_set(lhs, where, set_value)
    return jtu.tree_map(partial_lhs_set, tree, where, is_leaf=is_leaf)


def _at_apply_pytree(
    tree: PyTree,
    where: PyTree,
    func: Callable[[Any], Any],
    is_leaf: Callable[[Any], bool],
):
    def lhs_apply(lhs: Any, where: bool):
        """Set pytree node"""
        where = _check_valid_mask_leaf(where)
        value = func(lhs)
        if isinstance(lhs, (jnp.ndarray, np.ndarray)):
            if jnp.isscalar(value):
                return jnp.where(where, value, lhs)
            return value if jnp.all(where) else lhs
        return value if (where is True) else lhs

    return jtu.tree_map(lhs_apply, tree, where, is_leaf=is_leaf)


""" .at[...].reduce() """


def _at_reduce_pytree(
    tree: PyTree,
    where: PyTree,
    func: Callable[[Any], Any],
    is_leaf: Callable[[Any], bool] | None,
    initializer: Any,
):
    return jtu.tree_reduce(func, tree.at[where].get(is_leaf=is_leaf), initializer)


class PyTreeIndexer(NamedTuple):
    tree: PyTree
    where: PyTree

    def get(self, *, is_leaf: Callable[[Any], bool] | None = None):
        return _at_get_pytree(self.tree, self.where, is_leaf=is_leaf)

    def set(self, set_value, *, is_leaf: Callable[[Any], bool] | None = None):
        return _at_set_pytree(copy.copy(self.tree), self.where, set_value, is_leaf)

    def apply(self, func, *, is_leaf: Callable[[Any], bool] | None = None):
        return _at_apply_pytree(copy.copy(self.tree), self.where, func, is_leaf)

    def reduce(
        self, func, *, is_leaf: Callable[[Any], bool] | None = None, initializer=0
    ):
        return _at_reduce_pytree(self.tree, self.where, func, is_leaf, initializer)

    def __repr__(self) -> str:
        return f"where=({self.where})"

    def __str__(self) -> str:
        return f"where=({self.where})"


def _get_child(item: Any, path: list[str]) -> Any:
    """recursive getter"""
    # this function gets a certain attribute value based on a
    # sequence of strings.
    # for example _getter(item , ["a", "b", "c"]) is equivalent to item.a.b.c
    if len(path) == 1:
        return getattr(item, path[0])
    return _get_child(getattr(item, path[0]), path[1:])


def _get_parent_node_and_child_name(tree: Any, path: list[str]) -> tuple[Any, str]:
    if len(path) == 1:
        return (tree, path[0])
    return _get_parent_node_and_child_name(getattr(tree, path[0]), path[1:])


def _check_structure_mismatch(tree, attr_name: str):
    if hasattr(tree, attr_name):
        return

    msg = f"Error in retrieving `{attr_name}` from parent subtree {tree}.\n"
    msg += "This is likely due to reducing the parent subtree to a single leaf node.\n"
    msg += "This can happen if `is_leaf` is defined to reduce the parent subtree to a single leaf node.\n"
    raise AttributeError(msg)


def _at_set_str(
    tree: PyTree,
    path: list[str],
    set_value: Any,
    is_leaf: Callable[[Any], bool] | None = None,
):
    """Applies a function to a certain attribute of a tree based on a path using jax.tree_map and a mask."""
    # In essence this function retrieves the direct parent of the attribute
    # and applies the function to the attribute using a mask, that is False for all except at the attribute

    def recurse(path: list[str], value: Any, depth: int = 0):
        parent, child_name = _get_parent_node_and_child_name(tree, path)

        if not hasattr(parent, child_name):
            raise AttributeError(f"{child_name} is not a valid attribute of {parent}")

        if depth > 0:
            # non-leaf parent case setting the connection between the parent and the child
            parent.__dict__[child_name] = value
            return tree if len(path) == 1 else recurse(path[:-1], parent, depth + 1)

        # leaf parent case
        # create a mask that is True for the attribute and False for the rest
        parent_mask = jtu.tree_map(lambda _: False, parent, is_leaf=is_leaf)

        # check if parent subtree has not been reduced to a single leaf node
        _check_structure_mismatch(parent_mask, child_name)

        # masking the parent = False, and child = True
        child_mask = getattr(parent_mask, child_name)
        child_mask = jtu.tree_map(lambda _: True, child_mask)
        parent_mask.__dict__[child_name] = child_mask
        parent = _at_set_pytree(parent, parent_mask, set_value, is_leaf=is_leaf)
        return parent if len(path) == 1 else recurse(path[:-1], parent, depth + 1)

    return recurse(path, set_value, depth=0)


def _at_apply_str(
    tree: PyTree,
    path: list[str],
    func: Callable,
    is_leaf: Callable[[Any], bool] | None = None,
):
    """Applies a function to a certain attribute of a tree based on a path using jax.tree_map and a mask."""
    # In essence this function retrieves the direct parent of the attribute
    # and applies the function to the attribute using a mask, that is False for all except at the attribute

    def recurse(path: list[str], value: Any, depth: int = 0):
        parent, child_name = _get_parent_node_and_child_name(tree, path)

        if not hasattr(parent, child_name):
            raise AttributeError(f"{child_name} is not a valid attribute of {parent}")

        if depth > 0:
            # non-leaf parent case setting the connection between the parent and the child
            parent.__dict__[child_name] = value
            return tree if len(path) == 1 else recurse(path[:-1], parent, depth + 1)

        # leaf parent case
        # create a mask that is True for the attribute and False for the rest
        parent_mask = jtu.tree_map(lambda _: False, parent, is_leaf=is_leaf)

        # check if parent subtree has not been reduced to a single leaf node
        _check_structure_mismatch(parent_mask, child_name)

        # masking the parent = False, and child = True
        child_mask = getattr(parent_mask, child_name)
        child_mask = jtu.tree_map(lambda _: True, child_mask)
        parent_mask.__dict__[child_name] = child_mask
        parent = _at_apply_pytree(parent, parent_mask, func, is_leaf=is_leaf)
        return parent if len(path) == 1 else recurse(path[:-1], parent, depth + 1)

    return recurse(path, func, depth=0)


class StrIndexer(NamedTuple):
    tree: PyTree
    where: str

    def get(self):
        # x.at["a"].get() returns x.a
        return _get_child(self.tree, self.where.split("."))

    def set(self, set_value, *, is_leaf: Callable[[Any], bool] | None = None):
        where = self.where.split(".")
        return _at_set_str(copy.copy(self.tree), where, set_value, is_leaf=is_leaf)

    def apply(self, func, *, is_leaf: Callable[[Any], bool] | None = None):
        where = self.where.split(".")
        return _at_apply_str(copy.copy(self.tree), where, func, is_leaf=is_leaf)

    def __call__(self, *a, **k):
        with _call_context(self.tree) as tree:
            method = getattr(tree, self.where)
            return method(*a, **k), tree

    def __repr__(self) -> str:
        return f"where=({self.where!r})"

    def __str__(self) -> str:
        return f"where=({self.where})"


def _str_nested_indexer(tree, where):
    class StrNestedIndexer(StrIndexer):
        def __getitem__(nested_self, nested_where: list[str]):
            # check if nested_where is a valid attribute of the current node before
            # proceeding to the next level and to give a more informative error message
            if hasattr(_get_child(tree, nested_self.where.split(".")), nested_where):
                nested_where = nested_self.where + "." + nested_where
                return StrNestedIndexer(tree=tree, where=nested_where)

            msg = f"{nested_where} is not a valid attribute of {_get_child(tree, nested_self.where)}"
            raise AttributeError(msg)

        def __getattr__(nested_self, name):
            # support nested `.at``
            # for example `.at[A].at[B]` represents model.A.B
            if name == "at":
                # pass the current tree and the current path to the next `.at`
                return StrNestedIndexer(tree=tree, where=nested_self.where)

            msg = f"{name} is not a valid attribute of {nested_self}\n"
            msg += f"Did you mean to use .at[{name!r}]?"
            raise AttributeError(msg)

    return StrNestedIndexer(tree=tree, where=where)


def _pytree_nested_indexer(tree, where):
    class PyTreeNestedIndexer(PyTreeIndexer):
        def __getitem__(nested_self, nested_where):
            # here the case is .at[cond1].at[cond2] <-> .at[ cond1 and cond2 ]
            nested_where = nested_self.where & nested_where
            return PyTreeNestedIndexer(tree=tree, where=nested_where)

        def __getattr__(nested_self, name):
            # support for nested `.at`
            # e.g. `tree.at[tree>0].at[tree == str ]
            # corrsponds to (tree>0 and tree == str`)
            if name == "at":
                # pass the current where condition to the next level
                return PyTreeNestedIndexer(tree=tree, where=nested_self.where)

            msg = f"{name} is not a valid attribute of {nested_self}\n"
            msg += f"Did you mean to use .at[{name!r}]?"
            raise AttributeError(msg)

    return PyTreeNestedIndexer(tree=tree, where=where)


def _at_indexer(tree):
    class AtIndexer:
        def __getitem__(_, where):
            if isinstance(where, str):
                return _str_nested_indexer(tree=tree, where=where)

            if isinstance(where, type(tree)):
                # indexing by boolean pytree
                return _pytree_nested_indexer(tree=tree, where=where)

            if isinstance(where, EllipsisType):
                # Ellipsis as an alias for all elements
                # model.at[model == model ] <--> model.at[...]
                return tree.at[tree == tree]

            raise NotImplementedError(
                f"Indexing with {type(where)} is not implemented.\n"
                "Example of supported indexing:\n\n"
                "@pytc.treeclass\n"
                f"class {tree.__class__.__name__}:\n"
                "    ...\n\n"
                f">>> tree = {tree.__class__.__name__}(...)\n"
                "# indexing by boolean pytree\n"
                ">>> tree.at[tree > 0].get()\n\n"
                "# indexing by string\n"
                ">>> tree.at[`field_name`].get()"
            )

    return AtIndexer()
