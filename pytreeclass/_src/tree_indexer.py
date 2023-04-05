# this script defines methods used in indexing using `at` property
# and decorator using to enable masking and math operations

from __future__ import annotations

import copy
import functools as ft
import operator as op
from collections.abc import Callable
from contextlib import contextmanager
from math import ceil, floor, trunc
from typing import Any, NamedTuple, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from pytreeclass._src.tree_trace import tree_map_with_trace

PyTree = Any
_no_initializer = object()
_non_partial = object()
T = TypeVar("T")

# allow methods in mutable context to be called without raising an error
# this is done by registering the instance id in a set before entering the
# mutable context and removing it after exiting the context
_mutable_instance_registry: set[int] = set()


def _conditional_mutable_method(method: Callable) -> Callable:
    # decorator to allow for decorated methods to be called
    # within the mutable context and raise an error otherwise
    @ft.wraps(method)
    def wrapper(self, *a, **k):
        if id(self) in _mutable_instance_registry:
            return method(self, *a, **k)
        msg = f"Cannot call `{method.__name__}` on frozen instance."
        raise AttributeError(msg)

    return wrapper


@contextmanager
def _mutable_context(tree: PyTree, *, kopy: bool = False):
    tree = copy.copy(tree) if kopy else tree
    _mutable_instance_registry.add(id(tree))
    yield tree
    _mutable_instance_registry.discard(id(tree))


def _is_leaf_bool(node: Any) -> bool:
    if hasattr(node, "dtype"):
        return node.dtype == "bool"
    return isinstance(node, bool)


def _check_valid_mask_leaf(where: Any):
    if not _is_leaf_bool(where) and where is not None:
        raise TypeError(f"All tree leaves must be boolean.Found {(where)}")
    return where


def _true_tree(tree: PyTree, is_leaf) -> PyTree:
    return jtu.tree_map(lambda _: True, tree, is_leaf=is_leaf)


# tree indexing by boolean PyTree


def _get_at_mask(
    tree: PyTree, where: PyTree, is_leaf: Callable[[Any], bool] | None
) -> PyTree:
    def lhs_get(lhs: Any, where: Any):
        # check if where is a boolean leaf inside the `tree_map`
        # to avoid extrachecks in `tree_map`
        where = _check_valid_mask_leaf(where)
        if isinstance(lhs, (jax.Array, np.ndarray)):
            # return empty array instead of None if condition is not met
            # not `jittable` as size of array changes
            return lhs[jnp.where(where)]
        return lhs if where else None

    return jtu.tree_map(lhs_get, tree, where, is_leaf=is_leaf)


def _set_at_mask(
    tree: PyTree,
    where: PyTree,
    set_value: Any,
    is_leaf: Callable[[Any], bool] | None,
) -> PyTree:
    def lhs_set(lhs: Any, where: Any, set_value: Any):
        # check if where is a boolean leaf inside the `tree_map`
        # to avoid extrachecks in `tree_map`
        where = _check_valid_mask_leaf(where)
        if isinstance(lhs, (jax.Array, np.ndarray)):
            if jnp.isscalar(set_value):
                # apply scalar set_value to lhs array if condition is met
                return jnp.where(where, set_value, lhs)
            # set_value is not scalar
            return set_value if jnp.all(where) else lhs
        # lhs is not an array and set_value, so we apply the set_value to the
        # lhs if the condition is met
        return set_value if (where is True) else lhs

    if isinstance(set_value, type(tree)) and (
        jtu.tree_structure(tree, is_leaf=is_leaf)
        == jtu.tree_structure(set_value, is_leaf=is_leaf)
    ):
        # do not broadcast set_value if it is a pytree of same structure
        # for example tree.at[where].set(tree2) will set all tree leaves to tree2 leaves
        # if tree2 is a pytree of same structure as tree
        # instead of making each leaf of tree a copy of tree2
        # is design is similar to `numpy` design `Array.at[...].set(Array)`
        return jtu.tree_map(lhs_set, tree, where, set_value, is_leaf=is_leaf)

    # set_value is broadcasted to tree leaves
    # for example tree.at[where].set(1) will set all tree leaves to 1
    partial_lhs_set = lambda lhs, where: lhs_set(lhs, where, set_value)
    return jtu.tree_map(partial_lhs_set, tree, where, is_leaf=is_leaf)


def _apply_at_mask(
    tree: PyTree,
    where: PyTree,
    func: Callable[[Any], Any],
    is_leaf: Callable[[Any], bool] | None,
) -> PyTree:
    def lhs_apply(lhs: Any, where: bool):
        # check if where is a boolean leaf inside the `tree_map`
        # to avoid extrachecks in `tree_map`
        where, value = _check_valid_mask_leaf(where), func(lhs)
        if isinstance(lhs, (jax.Array, np.ndarray)):
            try:
                # lhs is an array with scalar output
                return jnp.where(where, value, lhs)
            except TypeError:
                # set_value is not `scalar` type
                return value if jnp.all(where) else lhs
        # lhs is not an array and value is not scalar
        return value if (where is True) else lhs

    return jtu.tree_map(lhs_apply, tree, where, is_leaf=is_leaf)


def _reduce_at_mask(
    tree: PyTree,
    where: PyTree,
    func: Callable[[Any, Any], Any],
    initializer: Any = _no_initializer,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    # note: current `jax`` implementation of `tree_reduce`
    # does not support `is_leaf` argument, but it is supported here
    tree = tree.at[where].get(is_leaf=is_leaf)
    if initializer is _no_initializer:
        return jtu.tree_reduce(func, tree)
    return jtu.tree_reduce(func, tree, initializer)


class _TreeAtMask(NamedTuple):
    tree: PyTree
    where: PyTree

    def get(self, *, is_leaf: Callable[[Any], bool] | None = None) -> PyTree:
        where = _true_tree(self.tree, is_leaf) if self.where is ... else self.where
        return _get_at_mask(self.tree, where, is_leaf)

    def set(
        self, set_value: Any, *, is_leaf: Callable[[Any], bool] | None = None
    ) -> PyTree:
        where = _true_tree(self.tree, is_leaf) if self.where is ... else self.where
        return _set_at_mask(self.tree, where, set_value, is_leaf)

    def apply(
        self,
        func: Callable[[Any], Any],
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ):
        where = _true_tree(self.tree, is_leaf) if self.where is ... else self.where
        return _apply_at_mask(self.tree, where, func, is_leaf)

    def reduce(
        self,
        func: Callable[[Any, Any], Any],
        initializer: Callable[[Any, Any], Any] | Any = _no_initializer,
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> Any:
        where = _true_tree(self.tree, is_leaf) if self.where is ... else self.where
        return _reduce_at_mask(self.tree, where, func, initializer, is_leaf)


def _at_mask(tree: PyTree, where: PyTree) -> PyTree:
    class TreeAtMask(_TreeAtMask):
        def __getitem__(self, rhs_where: str | int | PyTree):
            if isinstance(rhs_where, (str, int)):
                # promote `rhs` path to boolean mask
                mask = self.tree != self.tree
                mask = mask.at[rhs_where].set(True)
                rhs_where = mask

            rhs_where = self.where & rhs_where
            return TreeAtMask(tree=tree, where=rhs_where)

        def __getattr__(self, name):
            # support for nested `.at`
            if name == "at":
                # pass the current where condition to the next level
                return TreeAtMask(tree=tree, where=self.where)

            msg = f"{name} is not a valid attribute of {self}\n"
            msg += f"Did you mean to use .at[{name!r}]?"
            raise AttributeError(msg)

    return TreeAtMask(tree=tree, where=where)


# tree indexing by name or index  path


def _get_at_trace(
    tree: PyTree,
    where: tuple[int | str, ...],
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTree:
    # canonicalize `where` to `tuple` to conform to leaf trace type
    where = tuple(where)

    def lhs_get(trace, leaf):
        names, _, indices, __ = trace
        for i, item in enumerate(where):
            try:
                if (isinstance(item, int) and indices[i] != item) or (
                    isinstance(item, str) and names[i] != item
                ):
                    return None
            except IndexError:
                msg = f"Out of bounds path={where} for leaf={leaf}."
                msg += f" Path length={len(where)} is longer than "
                msg += f"leaf path levels={len(indices)-1}."
                raise IndexError(msg)
        return leaf

    return tree_map_with_trace(lhs_get, tree, is_leaf=is_leaf)


def _set_at_trace(
    tree: PyTree,
    where: tuple[int | str, ...],
    set_value: Any,
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTree:
    # canonicalize `where` to `tuple` to conform to leaf trace type
    where = tuple(where)

    def lhs_set(trace, leaf, set_value: Any):
        names, _, indices, __ = trace
        for i, item in enumerate(where):
            try:
                if (isinstance(item, int) and indices[i] != item) or (
                    isinstance(item, str) and names[i] != item
                ):
                    return leaf
            except IndexError:
                msg = f"Out of bounds path={where} for leaf={leaf}."
                msg += f" Path length={len(where)} is longer than "
                msg += f"leaf path levels={len(indices)-1}."
                raise IndexError(msg)
        # consider the same structure PyTree case tree.at[...].set(tree)
        return set_value

    # set_value is broadcasted to tree leaves
    # for example tree.at[where].set(1) will set all tree leaves to 1
    partial_lhs_set = lambda trace, lhs: lhs_set(trace, lhs, set_value)
    return tree_map_with_trace(partial_lhs_set, tree, is_leaf=is_leaf)


def _apply_at_trace(
    tree: PyTree,
    where: tuple[int | str, ...],
    func: Callable,
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTree:
    # canonicalize `where` to `tuple` to conform to leaf trace type
    where = tuple(where)

    def lhs_apply(trace, leaf):
        names, _, indices, __ = trace

        for i, item in enumerate(where):
            try:
                if (isinstance(item, int) and indices[i] != item) or (
                    isinstance(item, str) and names[i] != item
                ):
                    return leaf
            except IndexError:
                msg = f"Out of bounds path={where} for leaf={leaf}."
                msg += f" Path length={len(where)} is longer than "
                msg += f"leaf path levels={len(indices)-1}."
                raise IndexError(msg)
        return func(leaf)

    return tree_map_with_trace(lhs_apply, tree, is_leaf=is_leaf)


def _reduce_at_trace(
    tree: PyTree,
    where: tuple[str | int, ...],
    func: Callable[[Any, Any], Any],
    initializer: Any = _no_initializer,
    is_leaf: Callable[[Any], bool] | None = None,
) -> PyTree:
    # note: current `jax`` implementation of tree_reduce does
    # not support `is_leaf` argument using `tree_map_with_trace` with `is_leaf` argument
    # achieves the same result
    # canonicalize `where` to `tuple` to conform to leaf trace type
    where = tuple(where)

    def lhs_reduce(trace, leaf):
        names, _, indices, __ = trace

        for i, item in enumerate(where):
            try:
                if (isinstance(item, int) and indices[i] != item) or (
                    isinstance(item, str) and names[i] != item
                ):
                    return None
            except IndexError:
                msg = f"Out of bounds path={where} for leaf={leaf}."
                msg += f" Path length={len(where)} is more than "
                msg += f"leaf path levels={len(indices)-1}."
                raise IndexError(msg)
        return leaf

    tree = tree_map_with_trace(lhs_reduce, tree, is_leaf=is_leaf)
    if initializer is _no_initializer:
        return jtu.tree_reduce(func, tree)
    return jtu.tree_reduce(func, tree, initializer=initializer)


class _TreeAtPath(NamedTuple):
    tree: PyTree
    where: tuple[int | str]

    def get(self, *, is_leaf: Callable[[Any], bool] | None = None) -> PyTree:
        return _get_at_trace(self.tree, self.where, is_leaf)

    def set(
        self, set_value: Any, *, is_leaf: Callable[[Any], bool] | None = None
    ) -> PyTree:
        return _set_at_trace(self.tree, self.where, set_value, is_leaf)

    def apply(
        self,
        func: Callable[[Any], Any],
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> PyTree:
        return _apply_at_trace(self.tree, self.where, func, is_leaf)

    def reduce(
        self,
        func: Callable[[Any, Any], Any],
        *,
        initializer: Any = _no_initializer,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> Any:
        return _reduce_at_trace(self.tree, self.where, func, initializer, is_leaf)

    def __call__(self, *a, **k) -> tuple[Any, PyTree]:
        # return the output of the method called on the attribute
        # along with the new tree
        # this method first creates a copy of the tree,
        # next it unfreezes the tree then calls the method on the attribute
        # and finally freezes the tree again
        with _mutable_context(self.tree, kopy=True) as tree:
            # check if the attribute is a string
            (method_name,) = self.where
            if not isinstance(method_name, str):
                msg = "Expected method name to be a string, "
                msg += f"Found method_name=`{method_name}` "
                msg += f"of type={type(method_name).__name__}."
                raise TypeError(msg)

            method = getattr(tree, method_name)
            value = method(*a, **k)
        return value, tree


def _at_trace(tree: PyTree, where: tuple[str | int] | PyTree) -> PyTree:
    class TreeAtPath(_TreeAtPath):
        def __getitem__(self, rhs_where: tuple[str | int] | PyTree):
            # support for nested `.at``

            if isinstance(rhs_where, type(tree)):
                # promote `lhs` name path to boolean mask
                # and pass to `TreeAtMask`
                lhs_mask = self.tree != self.tree
                (lhs_where,) = self.where
                lhs_mask = lhs_mask.at[lhs_where].set(True)
                return _at_mask(tree=tree, where=lhs_mask & rhs_where)

            # case for name/index path rhs
            rhs_where = (*self.where, rhs_where)
            return _at_trace(tree=tree, where=rhs_where)

        def __getattr__(self, name):
            # support nested `.at``
            # for example `.at[A].at[B]` represents model.A.B
            if name == "at":
                # pass the current tree and the current path to the next `.at`
                return TreeAtPath(tree=tree, where=self.where)

            msg = f"{name} is not a valid attribute of {self}\n"
            msg += f"Did you mean to use .at[{name!r}]?"
            raise AttributeError(msg)

    return TreeAtPath(tree=tree, where=where)


def tree_indexer(tree: PyTree) -> PyTree:
    """Adds `.at` indexing abilities to a PyTree.

    Example:
        >>> import jax
        >>> import pytreeclass as pytc

        >>> @jax.tree_util.register_pytree_node_class
        ... class Tree:
        ...     def __init__(self, a, b):
        ...         self.a = a
        ...         self.b = b
        ...     def tree_flatten(self):
        ...         return (self.a, self.b), None
        ...     @classmethod
        ...     def tree_unflatten(cls, aux_data, children):
        ...         return cls(*children)
        ...     @property
        ...     def at(self):
        ...         return pytc.tree_indexer(self)
        ...     def __repr__(self) -> str:
        ...         return f"{self.__class__.__name__}(a={self.a}, b={self.b})"

        >>> # Register the `Tree` class trace function to support indexing
        >>> def test_trace_func(tree):
        ...     names = ("a", "b")
        ...     types = (type(tree.a), type(tree.b))
        ...     indices = (0, 1)
        ...     metadatas = (None, None)
        ...     return [*zip(names, types, indices, metadatas)]

        >>> pytc.register_pytree_node_trace(Tree, test_trace_func)

        >>> Tree(1, 2).at["a"].get()
        Tree(a=1, b=None)
    """

    class AtIndexer:
        def __getitem__(self, where):
            if isinstance(where, (str, int)):
                # indexing by name or index
                # name and index rules are defined using
                # `register_pytree_node_trace_func`
                return _at_trace(tree=tree, where=(where,))

            if isinstance(where, (type(tree), type(...))):
                # indexing by boolean pytree
                # Ellipsis as an alias for all elements
                # model.at[model == model ] <--> model.at[...]
                return _at_mask(tree=tree, where=where)

            raise NotImplementedError(
                f"Indexing with {type(where).__name__} is not implemented.\n"
                "Example of supported indexing:\n\n"
                "@pytc.treeclass\n"
                f"class {type(tree).__name__}:\n"
                "    ...\n\n"
                f">>> tree = {type(tree).__name__}(...)\n"
                ">>> # indexing by boolean pytree\n"
                ">>> tree.at[tree > 0].get()\n\n"
                ">>> # indexing by attribute name\n"
                ">>> tree.at[`attribute_name`].get()\n\n"
                ">>> # indexing by attribute index\n"
                ">>> tree.at[`attribute_index`].get()"
            )

    return AtIndexer()


class BroadcastablePartial(ft.partial):
    def __call__(self, *args, **keywords) -> Callable:
        # https://stackoverflow.com/a/7811270
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is _non_partial else arg for arg in self.args)  # type: ignore
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
        >>> import jax
        >>> import pytreeclass as pytc
        >>> import functools as ft

        >>> @ft.partial(pytc.treeclass, leafwise=True)
        ... class Test:
        ...    a: tuple[int] = (1,2,3)
        ...    b: tuple[int] = (4,5,6)
        ...    c: jax.Array = jnp.array([1,2,3])

        >>> tree = Test()

        >>> # 0 is broadcasted to all leaves of the pytree
        >>> print(pytc.bcmap(jnp.where)(tree>1, tree, 0))
        Test(a=(0, 2, 3), b=(4, 5, 6), c=[0 2 3])
        >>> print(pytc.bcmap(jnp.where)(tree>1, 0, tree))
        Test(a=(1, 0, 0), b=(0, 0, 0), c=[1 0 0])

        >>> # 1 is broadcasted to all leaves of the list pytree
        >>> pytc.bcmap(lambda x,y:x+y)([1,2,3],1)
        [2, 3, 4]

        >>> # trees are summed leaf-wise
        >>> pytc.bcmap(lambda x,y:x+y)([1,2,3],[1,2,3])
        [2, 4, 6]

        >>> # Non scalar second args case
        >>> try:
        ...     pytc.bcmap(lambda x,y:x+y)([1,2,3],[[1,2,3],[1,2,3]])
        ... except TypeError as e:
        ...     print(e)
        unsupported operand type(s) for +: 'int' and 'list'

        >>> # using **numpy** functions on pytrees
        >>> import jax.numpy as jnp
        >>> pytc.bcmap(jnp.add)([1,2,3],[1,2,3]) # doctest: +SKIP
        [2, 4, 6]
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
                if treedef0 == jtu.tree_structure(arg):
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
            if treedef0 == jtu.tree_structure(kwargs[key]):
                masked_kwargs[key] = _non_partial
                leaves += [treedef0.flatten_up_to(kwargs[key])]
                leaves_keys += [key]
            else:
                masked_kwargs[key] = kwargs[key]

        bfunc = BroadcastablePartial(func, *masked_args, **masked_kwargs)

        if len(leaves_keys) == 0:
            # no kwargs leaves are present, so we can immediately zip
            return jtu.tree_unflatten(treedef0, [bfunc(*xs) for xs in zip(*leaves)])

        # kwargs leaves are present, so we need to zip them
        kwargnum = len(leaves) - len(leaves_keys)
        all_leaves = []
        for xs in zip(*leaves):
            xs_args, xs_kwargs = xs[:kwargnum], xs[kwargnum:]
            all_leaves += [bfunc(*xs_args, **dict(zip(leaves_keys, xs_kwargs)))]
        return jtu.tree_unflatten(treedef0, all_leaves)

    docs = f"Broadcasted version of {func.__name__}\n{func.__doc__}"
    wrapper.__doc__ = docs
    return wrapper


def _unary_op(func):
    def wrapper(self):
        return jtu.tree_map(func, self)

    return ft.wraps(func)(wrapper)


def _binary_op(func):
    def wrapper(lhs, rhs=None):
        if isinstance(rhs, type(lhs)):
            return jtu.tree_map(func, lhs, rhs)
        return jtu.tree_map(lambda x: func(x, rhs), lhs)

    return ft.wraps(func)(wrapper)


def _swop(func):
    # swaping the arguments of a two-arg function
    return ft.wraps(func)(lambda lhs, rhs: func(rhs, lhs))


def _leafwise_transform(klass: type[T]) -> type[T]:
    # add leafwise transform methods to the class
    # that enable the user to apply a function to
    # all the leaves of the tree
    for key, method in (
        ("__abs__", _unary_op(abs)),
        ("__add__", _binary_op(op.add)),
        ("__and__", _binary_op(op.and_)),
        ("__ceil__", _unary_op(ceil)),
        ("__divmod__", _binary_op(divmod)),
        ("__eq__", _binary_op(op.eq)),
        ("__floor__", _unary_op(floor)),
        ("__floordiv__", _binary_op(op.floordiv)),
        ("__ge__", _binary_op(op.ge)),
        ("__gt__", _binary_op(op.gt)),
        ("__invert__", _unary_op(op.invert)),
        ("__le__", _binary_op(op.le)),
        ("__lshift__", _binary_op(op.lshift)),
        ("__lt__", _binary_op(op.lt)),
        ("__matmul__", _binary_op(op.matmul)),
        ("__mod__", _binary_op(op.mod)),
        ("__mul__", _binary_op(op.mul)),
        ("__ne__", _binary_op(op.ne)),
        ("__neg__", _unary_op(op.neg)),
        ("__or__", _binary_op(op.or_)),
        ("__pos__", _unary_op(op.pos)),
        ("__pow__", _binary_op(op.pow)),
        ("__radd__", _binary_op(_swop(op.add))),
        ("__rand__", _binary_op(_swop(op.and_))),
        ("__rdivmod__", _binary_op(_swop(divmod))),
        ("__rfloordiv__", _binary_op(_swop(op.floordiv))),
        ("__rlshift__", _binary_op(_swop(op.lshift))),
        ("__rmatmul__", _binary_op(_swop(op.matmul))),
        ("__rmod__", _binary_op(_swop(op.mod))),
        ("__rmul__", _binary_op(_swop(op.mul))),
        ("__ror__", _binary_op(_swop(op.or_))),
        ("__round__", _binary_op(round)),
        ("__rpow__", _binary_op(_swop(op.pow))),
        ("__rrshift__", _binary_op(_swop(op.rshift))),
        ("__rshift__", _binary_op(op.rshift)),
        ("__rsub__", _binary_op(_swop(op.sub))),
        ("__rtruediv__", _binary_op(_swop(op.truediv))),
        ("__rxor__", _binary_op(_swop(op.xor))),
        ("__sub__", _binary_op(op.sub)),
        ("__truediv__", _binary_op(op.truediv)),
        ("__trunc__", _unary_op(trunc)),
        ("__xor__", _binary_op(op.xor)),
    ):
        if key not in vars(klass):
            # do not override any user defined methods
            # this behavior similar is to `dataclasses.dataclass`
            setattr(klass, key, method)
    return klass


def _is_lhs_rhs_equal(lhs, rhs) -> bool | jax.Array:
    if hasattr(lhs, "shape") and hasattr(lhs, "dtype"):
        if hasattr(rhs, "shape") and hasattr(rhs, "dtype"):
            verdict = jnp.array_equal(lhs, rhs)
            try:
                return bool(verdict)
            except Exception:
                return verdict  # fail under `jit`
        return False
    return lhs == rhs


def is_tree_equal(*trees: Any) -> bool | jax.Array:
    """Return `True` if all pytrees are equal.

    Note:
        trees are compared using their leaves and treedefs.
        For `array` leaves `jnp.array_equal` is used, for other leaves
        method `__eq__` is used.

    Note:
        Under `jit` the return type is boolean `jax.Array` instead of python `bool`.
    """

    tree0, *rest = trees
    leaves0, treedef0 = jtu.tree_flatten(tree0)
    verdict = True

    for tree in rest:
        leaves, treedef = jtu.tree_flatten(tree)
        if (treedef != treedef0) or verdict is False:
            return False
        verdict = ft.reduce(op.and_, map(_is_lhs_rhs_equal, leaves0, leaves), verdict)
    return verdict


def tree_copy(tree: PyTree) -> PyTree:
    """Return a copy of the tree."""
    return jtu.tree_unflatten(*jtu.tree_flatten(tree)[::-1])  # type: ignore
