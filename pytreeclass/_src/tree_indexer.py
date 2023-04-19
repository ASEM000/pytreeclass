# this script defines methods used in indexing using `at` property
# and decorator using to enable masking and math operations

from __future__ import annotations

import copy
import functools as ft
import operator as op
from collections.abc import Callable
from contextlib import contextmanager
from math import ceil, floor, trunc
from typing import Any, NamedTuple, Sequence, TypeVar

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np

from pytreeclass._src.tree_trace import tree_map_with_trace

T = TypeVar("T")
PyTree = TypeVar("PyTree")
EllipsisType = type(Ellipsis)
TraceType = Any

_no_initializer = object()
_non_partial = object()

# allow methods in mutable context to be called without raising `AttributeError`
# the instances are registered  during initialization and using `at` property with `__call__
# this is done by registering the instance id in a set before entering the
# mutable context and removing it after exiting the context
_mutable_instance_registry: set[int] = set()


@contextmanager
def _mutable_context(tree: PyTree, *, kopy: bool = False):
    tree = copy.copy(tree) if kopy else tree
    _mutable_instance_registry.add(id(tree))
    yield tree
    _mutable_instance_registry.discard(id(tree))


# tree indexing by boolean PyTree


def _get_at_mask(
    tree: PyTree, where: PyTree, is_leaf: Callable[[Any], bool] | None
) -> PyTree:
    def leaf_get(leaf: Any, where: Any):
        # check if where is a boolean leaf inside the `tree_map`
        # to avoid extrachecks in `tree_map`
        if isinstance(leaf, (jax.Array, np.ndarray)) and jnp.ndim(leaf) > 0:
            # return empty array instead of None if condition is not met
            # not `jittable` as size of array changes
            return leaf[jnp.where(where)]
        return leaf if where else None

    return jtu.tree_map(leaf_get, tree, where, is_leaf=is_leaf)


def _set_at_mask(
    tree: PyTree,
    where: PyTree,
    set_value: Any,
    is_leaf: Callable[[Any], bool] | None,
) -> PyTree:
    def leaf_set(leaf: Any, where: Any, set_value: Any):
        # check if where is a boolean leaf inside the `tree_map`
        # to avoid extrachecks in `tree_map`
        if isinstance(leaf, (jax.Array, np.ndarray)):
            if jnp.isscalar(set_value):
                # apply scalar set_value to leaf array if condition is met
                return jnp.where(where, set_value, leaf)
            # set_value is not scalar
            return set_value if jnp.all(where) else leaf
        # leaf is not an array and set_value, so we apply the set_value to the
        # leaf if the condition is met
        return set_value if (where is True) else leaf

    if isinstance(set_value, type(tree)) and (
        jtu.tree_structure(tree, is_leaf=is_leaf)
        == jtu.tree_structure(set_value, is_leaf=is_leaf)
    ):
        # do not broadcast set_value if it is a pytree of same structure
        # for example tree.at[where].set(tree2) will set all tree leaves to tree2 leaves
        # if tree2 is a pytree of same structure as tree
        # instead of making each leaf of tree a copy of tree2
        # is design is similar to `numpy` design `Array.at[...].set(Array)`
        return jtu.tree_map(leaf_set, tree, where, set_value, is_leaf=is_leaf)

    # set_value is broadcasted to tree leaves
    # for example tree.at[where].set(1) will set all tree leaves to 1
    partial_leaf_set = lambda leaf, where: leaf_set(leaf, where, set_value)
    return jtu.tree_map(partial_leaf_set, tree, where, is_leaf=is_leaf)


def _apply_at_mask(
    tree: PyTree,
    where: PyTree,
    func: Callable[[Any], Any],
    is_leaf: Callable[[Any], bool] | None,
) -> PyTree:
    def leaf_apply(leaf: Any, where: bool):
        # check if where is a boolean leaf inside the `tree_map`
        # to avoid extrachecks in `tree_map`
        value = func(leaf)
        if isinstance(leaf, (jax.Array, np.ndarray)):
            try:
                # leaf is an array with scalar output
                return jnp.where(where, value, leaf)
            except TypeError:
                # set_value is not `scalar` type
                return value if jnp.all(where) else leaf
        # leaf is not an array and value is not scalar
        return value if (where is True) else leaf

    return jtu.tree_map(leaf_apply, tree, where, is_leaf=is_leaf)


def _reduce_at_mask(
    tree: PyTree,
    where: PyTree,
    func: Callable[[Any, Any], Any],
    initializer: Any = _no_initializer,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Any:
    tree = tree.at[where].get(is_leaf=is_leaf)
    if initializer is _no_initializer:
        return jtu.tree_reduce(func, tree)
    return jtu.tree_reduce(func, tree, initializer)


def _merge_where(
    tree: PyTree,
    where: tuple[int | str | PyTree | EllipsisType, ...],
    is_leaf: Callable[[Any], bool] | None = None,
):
    def merge_non_boolean_where(
        where: tuple[int | str | EllipsisType, ...],
        trace: Sequence[TraceType],
        leaf: Any,
    ):
        names, _, indices = trace
        is_array = isinstance(leaf, (jax.Array, np.ndarray))

        if len(where) > len(indices):
            return jnp.zeros_like(leaf, dtype=bool) if is_array else False
        for i, item in enumerate(where):
            if not (item is ... or indices[i] == item or names[i] == item):
                return jnp.zeros_like(leaf, dtype=bool) if is_array else False
        return jnp.ones_like(leaf, dtype=bool) if is_array else True

    def merge_boolean_where(*leaves):
        def is_leaf_bool(leaf: Any) -> bool:
            return (
                leaf.dtype == "bool"
                if hasattr(leaf, "dtype")
                else isinstance(leaf, bool)
            )

        verdict = True
        for leaf in leaves:
            if not is_leaf_bool(leaf):
                msg = f"Expected boolean leaf, found {type(leaf).__name__}."
                raise TypeError(msg)
            verdict &= leaf
        return verdict

    mask = None

    if non_boolean_where := [i for i in where if isinstance(i, (int, str, type(...)))]:
        func = ft.partial(merge_non_boolean_where, non_boolean_where)
        mask = tree_map_with_trace(func, tree, is_leaf=is_leaf)

    if boolean_where := [i for i in where if isinstance(i, type(tree))]:
        args = (mask, *boolean_where) if mask else boolean_where
        mask = jtu.tree_map(merge_boolean_where, *args)

    return mask


def _recursive_getattr(tree: Any, where: tuple[str, ...]):
    def step(tree: Any, where: tuple[str, ...]):
        if len(where) == 1:
            return getattr(tree, where[0])
        return step(getattr(tree, where[0]), where[1:])

    return step(tree, where)


class AtIndexer(NamedTuple):
    # base class for indexing with `.at`
    tree: PyTree
    where: tuple[str | int] | PyTree

    def __getitem__(self, where: str | int | PyTree | EllipsisType) -> AtIndexer:
        if isinstance(where, (type(self.tree), str, int, EllipsisType)):
            return AtIndexer(self.tree, (*self.where, where))

        raise NotImplementedError(
            f"Indexing with {type(where).__name__} is not implemented.\n"
            "Example of supported indexing:\n\n"
            f"class {type(self.tree).__name__}:(pytc.TreeClass)\n"
            "    ...\n\n"
            f">>> tree = {type(self.tree).__name__}(...)\n"
            ">>> # indexing by boolean pytree\n"
            ">>> tree.at[tree > 0].get()\n\n"
            ">>> # indexing by attribute name\n"
            ">>> tree.at[`attribute_name`].get()\n\n"
            ">>> # indexing by attribute index\n"
            ">>> tree.at[`attribute_index`].get()"
        )

    def get(self, *, is_leaf: Callable[[Any], bool] | None = None) -> PyTree:
        where = _merge_where(self.tree, self.where, is_leaf)
        return _get_at_mask(self.tree, where, is_leaf)

    def set(
        self,
        set_value: Any,
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> PyTree:
        where = _merge_where(self.tree, self.where, is_leaf)
        return _set_at_mask(self.tree, where, set_value, is_leaf)

    def apply(
        self,
        func: Callable[[Any], Any],
        *,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> PyTree:
        where = _merge_where(self.tree, self.where, is_leaf)
        return _apply_at_mask(self.tree, where, func, is_leaf)

    def reduce(
        self,
        func: Callable[[Any, Any], Any],
        *,
        initializer: Any = _no_initializer,
        is_leaf: Callable[[Any], bool] | None = None,
    ) -> Any:
        where = _merge_where(self.tree, self.where, is_leaf)
        return _reduce_at_mask(self.tree, where, func, initializer, is_leaf)

    def __getattr__(self, name: str) -> AtIndexer:
        # support nested `.at``
        # for example `.at[A].at[B]` represents model.A.B
        if name == "at":
            # pass the current tree and the current path to the next `.at`
            return AtIndexer(tree=self.tree, where=self.where)

        msg = f"{name} is not a valid attribute of {self}\n"
        msg += f"Did you mean to use .at[{name!r}]?"
        raise AttributeError(msg)

    def __call__(self, *a, **k) -> tuple[Any, PyTree]:
        with _mutable_context(self.tree, kopy=True) as tree:
            value = _recursive_getattr(tree, self.where)(*a, **k)
        return value, tree


def tree_indexer(tree: PyTree) -> AtIndexer:
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
        ...     indices = (None, None) # use None to indicate that the attribute is not indexable
        ...     return [*zip(names, types, indices)]

        >>> pytc.register_pytree_node_trace(Tree, test_trace_func)

        >>> Tree(1, 2).at["a"].get()
        Tree(a=1, b=None)
    """
    return AtIndexer(tree=tree, where=())


class BroadcastablePartial(ft.partial):
    def __call__(self, *args, **keywords) -> Callable:
        # https://stackoverflow.com/a/7811270
        keywords = {**self.keywords, **keywords}
        iargs = iter(args)
        args = (next(iargs) if arg is _non_partial else arg for arg in self.args)  # type: ignore
        return self.func(*args, *iargs, **keywords)


def bcmap(
    func: Callable[..., Any],
    *,
    is_leaf: Callable[[Any], bool] | None = None,
) -> Callable:
    """(map)s a function over pytrees leaves with automatic (b)road(c)asting for scalar arguments

    Args:
        func: the function to be mapped over the pytree
        is_leaf: a function that returns True if the argument is a leaf of the pytree

    Example:
        >>> import jax
        >>> import pytreeclass as pytc
        >>> import functools as ft

        >>> class Test(pytc.TreeClass, leafwise=True):
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
    def wrapper(leaf, rhs=None):
        if isinstance(rhs, type(leaf)):
            return jtu.tree_map(func, leaf, rhs)
        return jtu.tree_map(lambda x: func(x, rhs), leaf)

    return ft.wraps(func)(wrapper)


def _swop(func):
    # swaping the arguments of a two-arg function
    return ft.wraps(func)(lambda leaf, rhs: func(rhs, leaf))


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


def _is_leaf_rhs_equal(leaf, rhs) -> bool | jax.Array:
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        if hasattr(rhs, "shape") and hasattr(rhs, "dtype"):
            verdict = jnp.array_equal(leaf, rhs)
            try:
                return bool(verdict)
            except Exception:
                return verdict  # fail under `jit`
        return False
    return leaf == rhs


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
        verdict = ft.reduce(op.and_, map(_is_leaf_rhs_equal, leaves0, leaves), verdict)
    return verdict


def tree_copy(tree: T) -> T:
    """Return a copy of the tree."""
    return jtu.tree_unflatten(*jtu.tree_flatten(tree)[::-1])  # type: ignore
