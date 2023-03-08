# this script defines methods used in indexing using `at` property
# and decorator using to enable masking and math operations

from __future__ import annotations

import copy
import functools as ft
from collections.abc import Callable
from typing import Any, NamedTuple, Sequence

import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from pytreeclass._src.tree_freeze import _call_context
from pytreeclass._src.tree_trace import tree_map_with_trace

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


def _tree_copy(tree: PyTree) -> PyTree:
    """Return a copy of the tree"""
    return jtu.tree_unflatten(*jtu.tree_flatten(tree)[::-1])


def _tree_get_at_pytree(tree: PyTree, where: PyTree, is_leaf: Callable[[Any], bool]):
    def lhs_get(lhs: Any, where: Any):
        """Get pytree node value"""
        where = _check_valid_mask_leaf(where)
        if isinstance(lhs, (jnp.ndarray, np.ndarray)):
            # return empty array instead of None if condition is not met
            # not `jittable` as size of array changes
            return lhs[jnp.where(where)]
        return lhs if where else None

    return jtu.tree_map(lhs_get, tree, where, is_leaf=is_leaf)


def _tree_set_at_pytree(
    tree: PyTree,
    where: PyTree,
    set_value: Any,
    is_leaf: Callable[[Any], bool],
) -> PyTree:
    def lhs_set(lhs: Any, where: Any, set_value: Any):
        """Set pytree node value."""
        # fuse the boolean check here
        where = _check_valid_mask_leaf(where)
        if isinstance(lhs, (jnp.ndarray, np.ndarray)):
            if jnp.isscalar(set_value):
                # apply scalar set_value to lhs array if condition is met
                return jnp.where(where, set_value, lhs)
            # set_value is not scalar
            return set_value if jnp.all(where) else lhs
        # lhs is not an array and set_value, so we apply the set_value to the lhs if the condition is met
        return set_value if (where is True) else lhs

    if isinstance(set_value, type(tree)) and (
        jtu.tree_structure(tree, is_leaf=is_leaf)
        == jtu.tree_structure(set_value, is_leaf=is_leaf)
    ):
        # do not broadcast set_value if it is a pytree of same structure
        # for example tree.at[where].set(tree2) will set all tree leaves to tree2 leaves
        # if tree2 is a pytree of same structure as tree
        return jtu.tree_map(lhs_set, tree, where, set_value, is_leaf=is_leaf)

    # set_value is broadcasted to tree leaves
    # for example tree.at[where].set(1) will set all tree leaves to 1
    partial_lhs_set = lambda lhs, where: lhs_set(lhs, where, set_value)
    return jtu.tree_map(partial_lhs_set, tree, where, is_leaf=is_leaf)


def _tree_apply_at_pytree(
    tree: PyTree,
    where: PyTree,
    func: Callable[[Any], Any],
    is_leaf: Callable[[Any], bool],
) -> PyTree:
    def lhs_apply(lhs: Any, where: bool):
        """Set pytree node"""
        where = _check_valid_mask_leaf(where)
        value = func(lhs)
        if isinstance(lhs, (jnp.ndarray, np.ndarray)):
            if jnp.isscalar(value):
                # lhs is an array with scalar output
                return jnp.where(where, value, lhs)
            # set_value is not scalar
            return value if jnp.all(where) else lhs
        # lhs is not an array and value is not scalar
        return value if (where is True) else lhs

    return jtu.tree_map(lhs_apply, tree, where, is_leaf=is_leaf)


def _tree_reduce_at_pytree(
    tree: PyTree,
    where: PyTree,
    func: Callable[[Any], Any],
    is_leaf: Callable[[Any], bool] | None,
    initializer: Any,
) -> Any:
    return jtu.tree_reduce(func, tree.at[where].get(is_leaf=is_leaf), initializer)


class _TreeAtPyTree(NamedTuple):
    tree: PyTree
    where: PyTree

    def get(self, *, is_leaf: Callable[[Any], bool] | None = None):
        return _tree_get_at_pytree(self.tree, self.where, is_leaf=is_leaf)

    def set(self, set_value, *, is_leaf: Callable[[Any], bool] | None = None):
        return _tree_set_at_pytree(copy.copy(self.tree), self.where, set_value, is_leaf)

    def apply(self, func, *, is_leaf: Callable[[Any], bool] | None = None):
        return _tree_apply_at_pytree(copy.copy(self.tree), self.where, func, is_leaf)

    def reduce(self, func, *, is_leaf: Callable | None = None, initializer=0):
        return _tree_reduce_at_pytree(self.tree, self.where, func, is_leaf, initializer)

    def __repr__(self) -> str:
        return f"where=({self.where})"

    def __str__(self) -> str:
        return f"where=({self.where})"


def _tree_get_at_str(
    tree: PyTree, where: str, is_leaf: Callable[[Any], bool] | None = None
) -> PyTree:
    def map_func(trace, leaf):
        return leaf if where == trace.names[: len(where)] else None

    return tree_map_with_trace(map_func, tree, is_leaf=is_leaf)


def _tree_set_at_str(
    tree: PyTree,
    where: Sequence[str],
    set_value: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTree:
    """Applies a function to a certain attribute of a tree based on a where using jax.tree_map and a mask."""

    def map_func(trace, leaf):
        return set_value if where == trace.names[: len(where)] else leaf

    return tree_map_with_trace(map_func, tree, is_leaf=is_leaf)


def _tree_apply_at_str(
    tree: PyTree,
    where: Sequence[str],
    func: Callable,
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTree:
    """Applies a function to a certain attribute of a tree based on a path using jax.tree_map and a mask."""

    def map_func(trace, leaf):
        return func(leaf) if where == trace.names[: len(where)] else leaf

    return tree_map_with_trace(map_func, tree, is_leaf=is_leaf)


def _tree_reduce_at_str(
    tree: PyTree,
    where: Sequence[str],
    func: Callable[[Any, Any], Any],
    initializer: Any,
) -> PyTree:
    return jtu.tree_reduce(func, tree.at[where].get(), initializer)


class _TreeAtStr(NamedTuple):
    tree: PyTree
    where: Sequence[str]

    def get(self, *, is_leaf: Callable[[Any], bool] | None = None):
        return _tree_get_at_str(self.tree, self.where, is_leaf=is_leaf)

    def set(self, set_value: Any, *, is_leaf: Callable[[Any], bool] | None = None):
        return _tree_set_at_str(self.tree, self.where, set_value, is_leaf=is_leaf)

    def apply(self, func: Callable, *, is_leaf: Callable[[Any], bool] | None = None):
        return _tree_apply_at_str(self.tree, self.where, func, is_leaf=is_leaf)

    def reduce(self, func: Callable[[Any], Any], *, initializer: Any = 0):
        return _tree_reduce_at_str(self.tree, self.where, func, initializer)

    def __call__(self, *a, **k):
        # return the output of the method called on the attribute
        # along with the new tree
        # this method first creates a copy of the tree,
        # next it unfreezes the tree then calls the method on the attribute
        # and finally freezes the tree again
        with _call_context(self.tree) as tree:
            method = getattr(tree, ".".join(self.where))
            value = method(*a, **k)
        return value, tree

    def __repr__(self) -> str:
        where = ".".join(self.where)
        return f"where=({where!r})"

    def __str__(self) -> str:
        where = ".".join(self.where)
        return f"where=({where})"


def _tree_at_str(tree: PyTree, where: Sequence[str]) -> PyTree:
    class TreeAtStr(_TreeAtStr):
        def __getitem__(next_self, next_where: list[str]):
            # check if next_where is a valid attribute of the current node before
            # proceeding to the next level and to give a more informative error message
            next_where = (*next_self.where, next_where)
            return TreeAtStr(tree=tree, where=next_where)

        def __getattr__(next_self, name):
            # support nested `.at``
            # for example `.at[A].at[B]` represents model.A.B
            if name == "at":
                # pass the current tree and the current path to the next `.at`
                return TreeAtStr(tree=tree, where=next_self.where)

            msg = f"{name} is not a valid attribute of {next_self}\n"
            msg += f"Did you mean to use .at[{name!r}]?"
            raise AttributeError(msg)

    return TreeAtStr(tree=tree, where=where)


def _tree_at_pytree(tree: PyTree, where: Sequence[str]) -> PyTree:
    class TreeAtPyTree(_TreeAtPyTree):
        def __getitem__(next_self, next_where):
            # here the case is .at[cond1].at[cond2] <-> .at[ cond1 and cond2 ]
            next_where = next_self.where & next_where
            return TreeAtPyTree(tree=tree, where=next_where)

        def __getattr__(next_self, name):
            # support for nested `.at`
            # e.g. `tree.at[tree>0].at[tree == str ]
            # corrsponds to (tree>0 and tree == str`)
            if name == "at":
                # pass the current where condition to the next level
                return TreeAtPyTree(tree=tree, where=next_self.where)

            msg = f"{name} is not a valid attribute of {next_self}\n"
            msg += f"Did you mean to use .at[{name!r}]?"
            raise AttributeError(msg)

    return TreeAtPyTree(tree=tree, where=where)


def _tree_indexer(tree: PyTree) -> PyTree:
    class AtIndexer:
        def __getitem__(_, where):
            if isinstance(where, str):
                where = tuple(where.split("."))
                return _tree_at_str(tree=tree, where=where)
            if isinstance(where, tuple) and all(isinstance(w, str) for w in where):
                return _tree_at_str(tree=tree, where=where)

            if isinstance(where, type(tree)):
                # indexing by boolean pytree
                return _tree_at_pytree(tree=tree, where=where)

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


_non_partial = object()


class _Partial(ft.partial):
    def __call__(self, *args, **keywords) -> Callable:
        # https://stackoverflow.com/a/7811270
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is _non_partial else arg for arg in self.args)
        return self.func(*args, *iargs, **keywords)


@ft.lru_cache(maxsize=None)
def bcmap(
    func: Callable[..., Any], *, is_leaf: Callable[[Any], bool] | None = None
) -> Callable:
    """(map)s a function over pytrees leaves with automatic (b)road(c)asting for scalar arguments

    Args:
        func: the function to be mapped over the pytree
        is_leaf: a function that returns True if the argument is a leaf of the pytree

    Example:
        >>> @pytc.treeclass
        ... class Test:
        ...    a: tuple[int] = (1,2,3)
        ...    b: tuple[int] = (4,5,6)
        ...    c: jnp.ndarray = jnp.array([1,2,3])

        >>> tree = Test()
        >>> # 0 is broadcasted to all leaves of the pytree

        >>> print(pytc.bcmap(jnp.where)(tree>1, tree, 0))
        Test(a=(0,2,3), b=(4,5,6), c=[0 2 3])

        >>> print(pytc.bcmap(jnp.where)(tree>1, 0, tree))
        Test(a=(1,0,0), b=(0,0,0), c=[1 0 0])

        >>> # 1 is broadcasted to all leaves of the list pytree
        >>> bcmap(lambda x,y:x+y)([1,2,3],1)
        [2, 3, 4]

        >>> # trees are summed leaf-wise
        >>> bcmap(lambda x,y:x+y)([1,2,3],[1,2,3])
        [2, 4, 6]

        >>> # Non scalar second args case
        >>> bcmap(lambda x,y:x+y)([1,2,3],[[1,2,3],[1,2,3]])
        TypeError: unsupported operand type(s) for +: 'int' and 'list'

        >>> # using **numpy** functions on pytrees
        >>> import jax.numpy as jnp
        >>> bcmap(jnp.add)([1,2,3],[1,2,3])
        [DeviceArray(2, dtype=int32, weak_type=True),
        DeviceArray(4, dtype=int32, weak_type=True),
        DeviceArray(6, dtype=int32, weak_type=True)]
    """

    @ft.wraps(func)
    def wrapper(*args, **kwargs):
        if len(args) > 0:
            # positional arguments are passed the argument to be compare
            # the tree structure with is the first argument
            leaves0, treedef0 = jtu.tree_flatten(args[0], is_leaf=is_leaf)
            masked_args = [_non_partial]
            masked_kwargs = {}
            leaves = [leaves0]
            leaves_keys = []

            for arg in args[1:]:
                if jtu.tree_structure(arg) == treedef0:
                    masked_args += [_non_partial]
                    leaves += [treedef0.flatten_up_to(arg)]
                else:
                    masked_args += [arg]
        else:
            # only kwargs are passed the argument to be compare
            # the tree structure with is the first kwarg
            key0 = next(iter(kwargs))
            leaves0, treedef0 = jtu.tree_flatten(kwargs.pop(key0), is_leaf=is_leaf)
            masked_args = []
            masked_kwargs = {key0: _non_partial}
            leaves = [leaves0]
            leaves_keys = [key0]

        for key in kwargs:
            if jtu.tree_structure(kwargs[key]) == treedef0:
                masked_kwargs[key] = _non_partial
                leaves += [treedef0.flatten_up_to(kwargs[key])]
                leaves_keys += [key]
            else:
                masked_kwargs[key] = kwargs[key]

        func_ = _Partial(func, *masked_args, **masked_kwargs)

        if len(leaves_keys) == 0:
            # no kwargs leaves are present, so we can immediately zip
            return jtu.tree_unflatten(treedef0, [func_(*xs) for xs in zip(*leaves)])

        # kwargs leaves are present, so we need to zip them
        kwargnum = len(leaves) - len(leaves_keys)
        all_leaves = []
        for xs in zip(*leaves):
            xs_args, xs_kwargs = xs[:kwargnum], xs[kwargnum:]
            all_leaves += [func_(*xs_args, **dict(zip(leaves_keys, xs_kwargs)))]
        return jtu.tree_unflatten(treedef0, all_leaves)

    docs = f"Broadcasted version of {func.__name__}\n{func.__doc__}"
    wrapper.__doc__ = docs
    return wrapper
