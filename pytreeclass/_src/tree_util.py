from __future__ import annotations

from types import FunctionType
from typing import Any, Callable

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from pytreeclass._src.dispatch import dispatch

PyTree = Any


def is_treeclass(tree):
    """check if a class is treeclass"""
    return hasattr(tree, "__immutable_pytree__")


def is_treeclass_leaf_bool(node):
    """assert if treeclass leaf is boolean (for boolen indexing)"""
    if isinstance(node, jnp.ndarray):
        return node.dtype == "bool"
    else:
        return isinstance(node, bool)


def is_treeclass_leaf(tree):
    """assert if a node is treeclass leaf"""
    if is_treeclass(tree):

        return is_treeclass(tree) and not any(
            [is_treeclass(tree.__dict__[fi.name]) for fi in _tree_fields(tree).values()]
        )
    else:
        return False


def is_treeclass_non_leaf(tree):
    return is_treeclass(tree) and not is_treeclass_leaf(tree)


def is_treeclass_equal(lhs, rhs):
    """Assert if two treeclasses are equal"""
    lhs_leaves, lhs_treedef = jtu.tree_flatten(lhs)
    rhs_leaves, rhs_treedef = jtu.tree_flatten(rhs)

    def is_node_equal(lhs_node, rhs_node):
        if isinstance(lhs_node, jnp.ndarray) and isinstance(rhs_node, jnp.ndarray):
            return jnp.array_equal(lhs_node, rhs_node)
        else:
            return lhs_node == rhs_node

    return (lhs_treedef == rhs_treedef) and all(
        [is_node_equal(lhs_leaves[i], rhs_leaves[i]) for i in range(len(lhs_leaves))]
    )


def is_treeclass_frozen(tree):
    """assert if a treeclass is frozen"""
    if is_treeclass(tree):
        return hasattr(tree, "__pytree_structure_cache__")
    else:
        return False


def tree_copy(tree):
    return jtu.tree_unflatten(*jtu.tree_flatten(tree)[::-1])


def _tree_mutate(tree):
    """Enable mutable behavior for a treeclass instance"""
    if is_treeclass(tree):
        object.__setattr__(tree, "__immutable_pytree__", False)
        for field_item in _tree_fields(tree).values():
            if hasattr(tree, field_item.name):
                _tree_mutate(getattr(tree, field_item.name))
    return tree


class _fieldDict(dict):
    """A dict used for `__pytree_structure__` attribute of a treeclass instance"""

    def __eq__(self, other):
        return self.__dict__ == other.__dict__


def _tree_structure(tree) -> tuple[dict[str, Any], dict[str, Any]]:
    """Return dynamic and static fields of the pytree instance"""
    # this property scans the class fields and returns a tuple of two dicts (dynamic, static)
    # that mark the tree leaves seen by JAX computations and the static(tree structure) that are
    # not seen by JAX computations. the scanning is done if the instance is not frozen.
    # otherwise the cached values are returned.
    dynamic = _fieldDict()

    # undeclared fields are the fields that are not defined in the dataclass fields
    static = _fieldDict(__undeclared_fields__=tree.__undeclared_fields__)

    for field_item in _tree_fields(tree).values():
        if field_item.metadata.get("static", False):
            static[field_item.name] = getattr(tree, field_item.name)
        else:
            dynamic[field_item.name] = getattr(tree, field_item.name)
    return (dynamic, static)


def _tree_immutate(tree):
    """Enable immutable behavior for a treeclass instance"""
    if is_treeclass(tree):
        object.__setattr__(tree, "__immutable_pytree__", True)
        for field_item in _tree_fields(tree).values():
            if hasattr(tree, field_item.name):
                _tree_immutate(getattr(tree, field_item.name))
    return tree


def _tree_fields(tree):
    """Return a dictionary of all fields in the dataclass"""
    # in case of explicit treebase with no `param` then
    # its preferable not to create a new dict and just point to `__dataclass_fields__`
    # ** another feature of using an instance variable to store extra fields is that:
    # we can shadow the fields in the dataclass by creating a similarly named field in
    # the `undeclared_fields` instance variable, this avoids mutating the class fields.
    # For example in {**a,**b},  b keys will override a keys if they exist in both dicts.
    # this feature is used in functions that can set the `static` metadata
    # to specific instance fields (e.g. `filter_non_diff`)

    return (
        tree.__dataclass_fields__
        if len(tree.__undeclared_fields__) == 0
        else {**tree.__dataclass_fields__, **tree.__undeclared_fields__}
    )


def _tree_hash(tree):
    """Return a hash of the tree"""

    def _hash_node(node):
        """hash the leaves of the tree"""
        if isinstance(node, set):
            return frozenset(node)
        elif isinstance(node, jnp.ndarray):
            return np.array(node).tobytes()
        else:
            return node

    return hash(
        (
            tuple(jtu.tree_map(_hash_node, jtu.tree_leaves(tree))),
            jtu.tree_structure(tree),
        )
    )


def tree_freeze(tree):
    def recursive_freeze(tree):
        # cache the tree structure (dynamic/static)
        if is_treeclass(tree):
            object.__setattr__(
                tree, "__pytree_structure_cache__", _tree_structure(tree)
            )
            for kw in _tree_fields(tree):
                recursive_freeze(tree.__dict__[kw])
        return tree

    return recursive_freeze(tree_copy(tree))


def tree_unfreeze(tree):
    # remove the cached frozen structure
    def recursive_unfreeze(tree):
        if is_treeclass(tree):
            if hasattr(tree, "__pytree_structure_cache__"):
                object.__delattr__(tree, "__pytree_structure_cache__")
            for kw in _tree_fields(tree):
                recursive_unfreeze(tree.__dict__[kw])
        return tree

    return recursive_unfreeze(tree_copy(tree))


def _node_true(node, array_as_leaves: bool = True):
    @dispatch(argnum=0)
    def _node_true(node):
        return True

    @_node_true.register(jnp.ndarray)
    def _(node):
        return jnp.ones_like(node).astype(jnp.bool_) if array_as_leaves else True

    return _node_true(node)


def _node_false(node, array_as_leaves: bool = True):
    @dispatch(argnum=0)
    def _node_false(node):
        return False

    @_node_false.register(jnp.ndarray)
    def _(node):
        return jnp.zeros_like(node).astype(jnp.bool_) if array_as_leaves else True

    return _node_false(node)


def _pytree_map(
    tree: PyTree,
    *,
    cond: Callable[[Any, Any, Any], bool] | PyTree,
    true_func: Callable[[Any, Any, Any], Any],
    attr_func: Callable[[Any, Any, Any], str],
    is_leaf: Callable[[Any, Any, Any], bool],
    false_func: Callable[[Any, Any, Any], Any] | None = None,
) -> PyTree:

    """
    traverse the dataclass fields in a depth first manner

    Here, we apply true_func to node_item if condition is true and vice versa
    we use attr_func to select the attribute to be updated in the dataclass and
    is_leaf to decide whether to continue the traversal or not.

    a `jtu.tree_map` like function for treeclass instances with the option to modify
    the dataclass fields in-place

    Args:
        tree (Any):
            dataclass to be traversed

        cond (Callable[[Any, Any,Any], bool]) | (PyTree):
            - Callable to be applied on each (tree,field_item,node_item) or
            - a pytree of the same tree structure, where each node is a bool
            that determines whether to apply true_func or false_func

        true_func (Callable[[Any, Any,Any], Any]):
            function applied if cond is true, accepts (tree,field_item,node_item)

        attr_func (Callable[[Any, Any,Any], str]):
            function that returns the attribute to be updated, accepts (tree,field_item,node_item)

        is_leaf (Callable[[Any,Any,Any], bool]):
            stops recursion if false on (tree,field_item,node_item)

        false_func (Callable[[Any, Any,Any], Any]):  Defaults to None.
            function applied if cond is false, accepts (tree,field_item,node_item)

    Returns:
        PyTree or dataclass : new dataclass with updated attributes
    """

    def _recurse_treeclass_mask(
        tree: PyTree,
        *,
        cond: PyTree,
        true_func: Callable[[Any, Any, Any], Any],
        attr_func: Callable[[Any, Any, Any], str],
        is_leaf: Callable[[Any, Any, Any], bool],
        false_func: Callable[[Any, Any, Any], Any],
    ):
        for field_item, cond_item in zip(
            _tree_fields(tree).values(), _tree_fields(cond).values()
        ):
            node_item = getattr(tree, field_item.name)
            cond_item = getattr(cond, cond_item.name)

            if is_leaf(tree, field_item, node_item):
                # skip if the node is a leaf
                continue

            if is_treeclass(node_item):
                _recurse_treeclass_mask(
                    tree=node_item,
                    cond=cond_item,
                    true_func=true_func,
                    false_func=false_func,
                    attr_func=attr_func,
                    is_leaf=is_leaf,
                )

            else:
                object.__setattr__(
                    tree,
                    attr_func(tree, field_item, node_item),
                    true_func(tree, field_item, node_item)
                    if cond_item
                    else false_func(tree, field_item, node_item),
                )

        return tree

    def _recurse_callable_mask(
        tree: PyTree,
        *,
        cond: PyTree,
        true_func: Callable[[Any, Any, Any], Any],
        attr_func: Callable[[Any, Any, Any], str],
        is_leaf: Callable[[Any, Any, Any], bool],
        false_func: Callable[[Any, Any, Any], Any],
        state: Any = None,
    ):
        for field_item in _tree_fields(tree).values():
            node_item = getattr(tree, field_item.name)

            if is_leaf(tree, field_item, node_item):
                continue

            if is_treeclass(node_item):
                _recurse_callable_mask(
                    tree=node_item,
                    cond=cond,
                    true_func=true_func,
                    false_func=false_func,
                    attr_func=attr_func,
                    is_leaf=is_leaf,
                    state=jtu.tree_all(cond(tree, field_item, node_item)) or state,
                )

            else:
                object.__setattr__(
                    tree,
                    attr_func(tree, field_item, node_item),
                    true_func(tree, field_item, node_item)
                    if (state or cond(tree, field_item, node_item))
                    else false_func(tree, field_item, node_item),
                )

        return tree

    if isinstance(cond, FunctionType):
        return _recurse_callable_mask(
            tree=tree_copy(tree),
            cond=cond,
            true_func=true_func,
            false_func=false_func,
            attr_func=attr_func,
            is_leaf=is_leaf,
            state=None,
        )
    elif isinstance(cond, type(tree)):
        return _recurse_treeclass_mask(
            tree=tree_copy(tree),
            cond=cond,
            true_func=true_func,
            false_func=false_func,
            attr_func=attr_func,
            is_leaf=is_leaf,
        )

    else:
        raise TypeError(
            f"cond must be Callable, or PyTree of same structure. Found {cond}"
        )
