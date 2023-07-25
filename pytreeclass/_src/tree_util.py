# Copyright 2023 PyTreeClass authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for pytrees."""

from __future__ import annotations

import dataclasses as dc
import functools as ft
import operator as op
from math import ceil, floor, trunc
from typing import Any, Callable, Hashable, Iterator, Sequence, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
from jax._src.tree_util import _registry, _registry_with_keypaths
from jax.util import unzip2

T = TypeVar("T")
PyTree = Any
# TODO: swich to EllipsisType for python 3.10
EllipsisType = TypeVar("EllipsisType")
KeyEntry = TypeVar("KeyEntry", bound=Hashable)
TypeEntry = TypeVar("TypeEntry", bound=type)
TraceEntry = Tuple[KeyEntry, TypeEntry]
KeyPath = Tuple[KeyEntry, ...]
TypePath = Tuple[TypeEntry, ...]
TraceType = Tuple[KeyPath, TypePath]
IsLeafType = Union[None, Callable[[Any], bool]]


@dc.dataclass(frozen=True)
class NamedSequenceKey(jtu.GetAttrKey, jtu.SequenceKey):
    # inherit from both `jtu.GetAttrKey` and `jtu.SequenceKey`
    # in case the user perform isinstance check to unpack the name/index
    # `TreeClass` is modeled as a `NamedTuple`` with both `name` and `idx` identifiers
    def __str__(self):
        return f".{self.name}"


def tree_hash(*trees: PyTree) -> int:
    leaves, treedef = jtu.tree_flatten(trees)
    return hash((*leaves, treedef))


def tree_copy(tree: T) -> T:
    """Return a copy of the tree."""
    return jtu.tree_unflatten(*jtu.tree_flatten(tree)[::-1])


def _is_leaf_rhs_equal(leaf, rhs) -> bool | jax.Array:
    if hasattr(leaf, "shape") and hasattr(leaf, "dtype"):
        if hasattr(rhs, "shape") and hasattr(rhs, "dtype"):
            if leaf.shape != rhs.shape:
                return False
            if leaf.dtype != rhs.dtype:
                return False
            try:
                return bool(verdict := jnp.all(leaf == rhs))
            except Exception:
                return verdict  # fail under `jit`
        return False
    return leaf == rhs


def is_tree_equal(*trees: Any) -> bool | jax.Array:
    """Return ``True`` if all pytrees are equal.

    Note:
        trees are compared using their leaves and treedefs.

    Note:
        Under ``jit`` the return type is boolean `jax.Array` instead of ``bool``.
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


class Partial:
    """jaxable Partial function with support for positional partial application.

    Example:
        >>> import pytreeclass as pytc
        >>> def f(a, b, c):
        ...     print(f"a: {a}, b: {b}, c: {c}")
        ...     return a + b + c

        >>> # positional arguments using `...` placeholder
        >>> f_a = pytc.Partial(f, ..., 2, 3)
        >>> f_a(1)
        a: 1, b: 2, c: 3
        6

        >>> # keyword arguments
        >>> f_b = pytc.Partial(f, b=2, c=3)
        >>> f_a(1)
        a: 1, b: 2, c: 3
        6

    Note:
        - The ``...`` is used to indicate a placeholder for positional arguments.
        - See: https://stackoverflow.com/a/7811270
        - `Partial` is used internally by `bcmap` which maps a function over pytrees
          leaves with automatic broadcasting for scalar arguments.
    """

    __slots__ = ["func", "args", "kwargs", "__weakref__"]  # type: ignore

    def __init__(self, func: Callable[..., Any], *args: Any, **kwargs: Any):
        """Initialize a `Partial` function.

        Args:
            func: The function to be partially applied.
            args: Positional arguments to be partially applied. use ``...`` as a
                placeholder for positional arguments.
            kwargs: Keyword arguments to be partially applied.
        """
        self.func = func
        self.args = args
        self.kwargs = kwargs

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        iargs = iter(args)
        args = (next(iargs) if arg is ... else arg for arg in self.args)  # type: ignore
        return self.func(*args, *iargs, **{**self.kwargs, **kwargs})

    def __repr__(self) -> str:
        return f"Partial({self.func}, {self.args}, {self.kwargs})"

    def __hash__(self) -> int:
        return tree_hash(self)

    def __eq__(self, other: Any) -> bool:
        return is_tree_equal(self, other)


jtu.register_pytree_node(
    nodetype=Partial,
    flatten_func=lambda x: ((x.func, x.args, x.kwargs), None),
    unflatten_func=lambda _, xs: Partial(*xs),
)


def bcmap(func: Callable, *, is_leaf: IsLeafType = None) -> Callable:
    """Map a function over pytree leaves with automatic broadcasting for scalar arguments.

    Args:
        func: the function to be mapped over the pytree
        is_leaf: a predicate function that returns True if the node is a leaf

    Example:
        >>> import jax
        >>> import pytreeclass as pytc
        >>> import functools as ft

        >>> @pytc.autoinit
        ... @pytc.leafwise
        ... class Test(pytc.TreeClass):
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
            masked_args = [...]
            masked_kwargs = {}
            leaves = [leaves0]
            leaves_keys = []

            for arg in args[1:]:
                if treedef0 == jtu.tree_structure(arg):
                    masked_args += [...]
                    leaves += [treedef0.flatten_up_to(arg)]
                else:
                    masked_args += [arg]
        else:
            # only kwargs are passed the argument to be compare
            # the tree structure with is the first kwarg
            key0 = next(iter(kwargs))
            leaves0, treedef0 = jtu.tree_flatten(kwargs.pop(key0), is_leaf=is_leaf)
            masked_args = []
            masked_kwargs = {key0: ...}
            leaves = [leaves0]
            leaves_keys = [key0]

        for key in kwargs:
            if treedef0 == jtu.tree_structure(kwargs[key]):
                masked_kwargs[key] = ...
                leaves += [treedef0.flatten_up_to(kwargs[key])]
                leaves_keys += [key]
            else:
                masked_kwargs[key] = kwargs[key]

        bfunc = Partial(func, *masked_args, **masked_kwargs)

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

    name = getattr(func, "__name__", func)

    docs = f"Broadcasted version of {name}\n{func.__doc__}"
    wrapper.__doc__ = docs
    return wrapper


def uop(func):
    def wrapper(self):
        return jtu.tree_map(func, self)

    return ft.wraps(func)(wrapper)


def bop(func):
    def wrapper(leaf, rhs=None):
        if isinstance(rhs, type(leaf)):
            return jtu.tree_map(func, leaf, rhs)
        return jtu.tree_map(lambda x: func(x, rhs), leaf)

    return ft.wraps(func)(wrapper)


def swop(func):
    # swaping the arguments of a two-arg function
    return ft.wraps(func)(lambda leaf, rhs: func(rhs, leaf))


def leafwise(klass: type[T]) -> type[T]:
    """A class decorator that adds leafwise operators to a class.

    Leafwise operators are operators that are applied to the leaves of a pytree.
    For example leafwise ``__add__`` is equivalent to:

    - ``jax.tree_map(lambda x: x + rhs, tree)`` if ``rhs`` is a scalar.
    - ``jax.tree_map(lambda x, y: x + y, tree, rhs)`` if ``rhs`` is a pytree
      with the same structure as ``tree``.

    Args:
        klass: The class to be decorated.

    Returns:
        The decorated class.

    Example:
        >>> # use ``numpy`` functions on :class:`TreeClass`` classes decorated with ``leafwise``
        >>> import pytreeclass as pytc
        >>> import jax.numpy as jnp
        >>> @pytc.leafwise
        ... @pytc.autoinit
        ... class Point(pytc.TreeClass):
        ...    x: float = 0.5
        ...    y: float = 1.0
        ...    description: str = "point coordinates"
        >>> # use :func:`tree_mask` to mask the non-inexact part of the tree
        >>> # i.e. mask the string leaf ``description`` to ``Point`` work
        >>> # with ``jax.numpy`` functions
        >>> co = pytc.tree_mask(Point())
        >>> print(pytc.bcmap(jnp.where)(co > 0.5, co, 1000))
        Point(x=1000.0, y=1.0, description=#point coordinates)

    Note:
        If a mathematically equivalent operator is already defined on the class,
        then it is not overridden.

    ==================      ============
    Method                  Operator
    ==================      ============
    ``__add__``              ``+``
    ``__and__``              ``&``
    ``__ceil__``             ``math.ceil``
    ``__divmod__``           ``divmod``
    ``__eq__``               ``==``
    ``__floor__``            ``math.floor``
    ``__floordiv__``         ``//``
    ``__ge__``               ``>=``
    ``__gt__``               ``>``
    ``__invert__``           ``~``
    ``__le__``               ``<=``
    ``__lshift__``           ``<<``
    ``__lt__``               ``<``
    ``__matmul__``           ``@``
    ``__mod__``              ``%``
    ``__mul__``              ``*``
    ``__ne__``               ``!=``
    ``__neg__``              ``-``
    ``__or__``               ``|``
    ``__pos__``              ``+``
    ``__pow__``              ``**``
    ``__round__``            ``round``
    ``__sub__``              ``-``
    ``__truediv__``          ``/``
    ``__trunc__``            ``math.trunc``
    ``__xor__``              ``^``
    ==================      ============
    """
    for key, method in (
        ("__abs__", uop(abs)),
        ("__add__", bop(op.add)),
        ("__and__", bop(op.and_)),
        ("__ceil__", uop(ceil)),
        ("__divmod__", bop(divmod)),
        ("__eq__", bop(op.eq)),
        ("__floor__", uop(floor)),
        ("__floordiv__", bop(op.floordiv)),
        ("__ge__", bop(op.ge)),
        ("__gt__", bop(op.gt)),
        ("__invert__", uop(op.invert)),
        ("__le__", bop(op.le)),
        ("__lshift__", bop(op.lshift)),
        ("__lt__", bop(op.lt)),
        ("__matmul__", bop(op.matmul)),
        ("__mod__", bop(op.mod)),
        ("__mul__", bop(op.mul)),
        ("__ne__", bop(op.ne)),
        ("__neg__", uop(op.neg)),
        ("__or__", bop(op.or_)),
        ("__pos__", uop(op.pos)),
        ("__pow__", bop(op.pow)),
        ("__radd__", bop(swop(op.add))),
        ("__rand__", bop(swop(op.and_))),
        ("__rdivmod__", bop(swop(divmod))),
        ("__rfloordiv__", bop(swop(op.floordiv))),
        ("__rlshift__", bop(swop(op.lshift))),
        ("__rmatmul__", bop(swop(op.matmul))),
        ("__rmod__", bop(swop(op.mod))),
        ("__rmul__", bop(swop(op.mul))),
        ("__ror__", bop(swop(op.or_))),
        ("__round__", bop(round)),
        ("__rpow__", bop(swop(op.pow))),
        ("__rrshift__", bop(swop(op.rshift))),
        ("__rshift__", bop(op.rshift)),
        ("__rsub__", bop(swop(op.sub))),
        ("__rtruediv__", bop(swop(op.truediv))),
        ("__rxor__", bop(swop(op.xor))),
        ("__sub__", bop(op.sub)),
        ("__truediv__", bop(op.truediv)),
        ("__trunc__", uop(trunc)),
        ("__xor__", bop(op.xor)),
    ):
        if key not in vars(klass):
            # do not override any user defined methods
            # this behavior similar is to `dataclasses.dataclass`
            setattr(klass, key, method)
    return klass


def flatten_one_trace_level(
    trace: TraceType,
    tree: PyTree,
    is_leaf: IsLeafType,
    is_trace_leaf: Callable[[TraceType], bool] | None,
):
    # the code style of `tree_{...} is heavilty influenced by `jax.tree_util`
    # https://github.com/google/jax/blob/main/jax/_src/tree_util.py
    # similar to jax corresponding key path API but adds `is_trace_leaf`
    # predicate and type path
    if (is_leaf and is_leaf(tree)) or (is_trace_leaf and is_trace_leaf(trace)):
        # is_leaf is a predicate function that determines whether a value
        # is a leaf is_trace_leaf is a predicate function that determines
        # whether a trace is a leaf
        yield trace, tree
        return

    if type(tree) in _registry_with_keypaths:
        keys_leaves, _ = _registry_with_keypaths[type(tree)].flatten_with_keys(tree)
        keys, leaves = unzip2(keys_leaves)

    elif isinstance(tree, tuple) and hasattr(tree, "_fields"):
        # this conforms to the `jax` convention for namedtuples
        leaves = (getattr(tree, field) for field in tree._fields)  # type: ignore
        # use `NamedSequenceKey` to index by name and index unlike `jax` handler
        keys = tuple(NamedSequenceKey(idx, key) for idx, key in enumerate(tree._fields))  # type: ignore

    elif type(tree) in _registry:
        # no named handler for this type in key path
        leaves, _ = _registry[type(tree)].to_iter(tree)
        keys = tuple(jtu.GetAttrKey(f"leaf_{i}") for i, _ in enumerate(leaves))

    else:
        yield trace, tree
        return

    for key, leaf in zip(keys, leaves):
        yield from flatten_one_trace_level(
            ((*trace[0], key), (*trace[1], type(leaf))),
            leaf,
            is_leaf,
            is_trace_leaf,
        )


def tree_leaves_with_trace(
    tree: PyTree,
    *,
    is_leaf: IsLeafType = None,
    is_trace_leaf: Callable[[TraceType], bool] | None = None,
) -> Sequence[tuple[TraceType, Any]]:
    r"""Similar to jax.tree_util.tree_leaves` but returns  object, leaf pairs.

    Args:
        tree: The tree to be flattened.
        is_leaf: A predicate function that determines whether a value is a leaf.
        is_trace_leaf: A predicate function that determines whether a trace is a leaf.

    Returns:
        A list of (trace, leaf) pairs.

    Example:
        >>> import pytreeclass as pytc
        >>> tree = [1, [2, [3]]]
        >>> traces, _ = zip(*pytc.tree_leaves_with_trace(tree))
    """
    return list(flatten_one_trace_level(((), ()), tree, is_leaf, is_trace_leaf))


def tree_flatten_with_trace(
    tree: PyTree,
    *,
    is_leaf: IsLeafType = None,
) -> tuple[Sequence[tuple[TraceType, Any]], jtu.PyTreeDef]:
    """Similar to jax.tree_util.tree_flatten` but returns key path, type path pairs.

    Args:
        tree: The tree to be flattened.
        is_leaf: A predicate function that determines whether a value is a leaf.

    Returns:
        A pair (leaves, treedef) where leaves is a list of (trace, leaf) pairs and
        treedef is a PyTreeDef object that can be used to reconstruct the tree.
    """
    treedef = jtu.tree_structure(tree, is_leaf=is_leaf)
    traces_leaves = tree_leaves_with_trace(tree, is_leaf=is_leaf)
    return traces_leaves, treedef


def tree_map_with_trace(
    func: Callable[..., Any],
    tree: Any,
    *rest: Any,
    is_leaf: IsLeafType = None,
) -> Any:
    # the code style of `tree_{...} is heavilty influenced by `jax.tree_util`
    # https://github.com/google/jax/blob/main/jax/_src/tree_util.py
    """Map a function over a pytree, with trace and type path.

    Similar to `jax.tree_util.tree_map_with_path` that accept a function
    that takes a two-item tuple for key path and type path.

    Args:
        func: A function that takes a trace and a leaf and returns a new leaf.
        tree: The tree to be mapped over.
        rest: Additional trees to be mapped over.
        is_leaf: A predicate function that determines whether a value is a leaf.

    Returns:
        A new tree with the same structure as tree.

    Example:
        >>> import jax.tree_util as jtu
        >>> import pytreeclass as pytc
        >>> tree = {"a": [1, 2], "b": 4, "c": [5, 6]}

        >>> # apply to "a" leaf
        >>> def map_func(trace, leaf):
        ...     names, _= trace
        ...     if jtu.DictKey("a") in names:
        ...         return leaf + 100
        ...     return leaf
        >>> pytc.tree_map_with_trace(map_func, tree)
        {'a': [101, 102], 'b': 4, 'c': [5, 6]}

        >>> # apply to any item with list in its type path
        >>> def map_func(trace, leaf):
        ...     _, types = trace
        ...     if list in types:
        ...         return leaf + 100
        ...     return leaf
        >>> pytc.tree_map_with_trace(map_func, tree)
        {'a': [101, 102], 'b': 4, 'c': [105, 106]}
    """
    traces_leaves, treedef = tree_flatten_with_trace(tree, is_leaf=is_leaf)
    traces_leaves = list(zip(*traces_leaves))
    traces_leaves += [treedef.flatten_up_to(r) for r in rest]
    return treedef.unflatten(func(*xs) for xs in zip(*traces_leaves))


class Node:
    # mainly used for visualization
    __slots__ = ["data", "parent", "children", "__weakref__"]

    def __init__(
        self,
        data: tuple[TraceEntry, Any],
        parent: Node | None = None,
    ):
        self.data = data
        self.parent = parent
        self.children: dict[TraceEntry, Node] = {}

    def add_child(self, child: Node) -> None:
        # add child node to this node and set
        # this node as the parent of the child
        if not isinstance(child, Node):
            raise TypeError(f"`child` must be a `Node`, got {type(child)}")
        ti, __ = child.data
        if ti not in self.children:
            # establish parent-child relationship
            child.parent = self
            self.children[ti] = child

    def __iter__(self) -> Iterator[Node]:
        # iterate over children nodes
        return iter(self.children.values())

    def __repr__(self) -> str:
        return f"Node(data={self.data})"

    def __contains__(self, key: TraceEntry) -> bool:
        return key in self.children


def construct_tree(
    tree: PyTree,
    is_leaf: IsLeafType = None,
    is_trace_leaf: IsLeafType = None,
) -> Node:
    # construct a tree with `Node` objects using `tree_leaves_with_trace`
    # to establish parent-child relationship between nodes

    traces_leaves = tree_leaves_with_trace(
        tree,
        is_leaf=is_leaf,
        is_trace_leaf=is_trace_leaf,
    )

    ti = (None, type(tree))
    vi = tree
    root = Node(data=(ti, vi))

    for trace, leaf in traces_leaves:
        keys, types = trace
        cur = root
        for i, ti in enumerate(zip(keys, types)):
            if ti in cur:
                # common parent node
                cur = cur.children[ti]
            else:
                # new path
                vi = leaf if i == len(keys) - 1 else None
                child = Node(data=(ti, vi))
                cur.add_child(child)
                cur = child
    return root
