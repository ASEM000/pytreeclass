from __future__ import annotations

import dataclasses as dc
import functools as ft
import hashlib
import operator as op
from math import ceil, floor, trunc
from typing import Any, Callable, Hashable, Iterator, Sequence, Tuple, TypeVar, Union

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax._src.tree_util import _registry, _registry_with_keypaths
from jax.util import unzip2

T = TypeVar("T")
PyTree = Any
EllipsisType = type(Ellipsis)
KeyEntry = TypeVar("KeyEntry", bound=Hashable)
TypeEntry = TypeVar("TypeEntry", bound=type)
TraceEntry = Tuple[KeyEntry, TypeEntry]
KeyPath = Tuple[KeyEntry, ...]
TypePath = Tuple[TypeEntry, ...]
TraceType = Tuple[KeyPath, TypePath]
IsLeafType = Union[type(None), Callable[[Any], bool]]


def _hash_node(node: Any) -> int:
    if isinstance(node, (jax.Array, np.ndarray)):
        return int(hashlib.sha256(np.array(node).tobytes()).hexdigest(), 16)
    if isinstance(node, set):
        return hash(frozenset(node))
    if isinstance(node, dict):
        return hash(frozenset(node.items()))
    if isinstance(node, list):
        return hash(tuple(node))
    return hash(node)


def tree_hash(*trees: PyTree) -> int:
    hashed = jtu.tree_map(_hash_node, jtu.tree_leaves(trees))
    return hash((*hashed, jtu.tree_structure(trees)))


class _ImmutableWrapper:
    __slots__ = ("__wrapped__", "__weakref__")

    def __init__(self, x: Any) -> None:
        object.__setattr__(self, "__wrapped__", getattr(x, "__wrapped__", x))

    def unwrap(self) -> Any:
        return getattr(self, "__wrapped__")

    def __setattr__(self, _, __) -> None:
        raise AttributeError("Cannot assign to frozen instance.")

    def __delattr__(self, _: str) -> None:
        raise AttributeError("Cannot delete from frozen instance.")


class _HashableWrapper(_ImmutableWrapper):
    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, _HashableWrapper):
            return False
        return tree_hash(self.unwrap()) == tree_hash(rhs.unwrap())

    def __hash__(self) -> int:
        return tree_hash(self.unwrap())


def _frozen_error(opname: str, tree):
    raise NotImplementedError(
        f"Cannot apply `{opname}` operation to a frozen object `{tree!r}`.\n"
        "Unfreeze the object first to apply operations to it\n"
        "Example:\n"
        ">>> import jax\n"
        ">>> import pytreeclass as pytc\n"
        ">>> tree = jax.tree_map(pytc.unfreeze, tree, is_leaf=pytc.is_frozen)"
    )


class _FrozenWrapper(_ImmutableWrapper):
    def __repr__(self):
        return f"#{self.unwrap()!r}"

    def __eq__(self, rhs: Any) -> bool:
        if not isinstance(rhs, _FrozenWrapper):
            return False
        return self.unwrap() == rhs.unwrap()

    def __hash__(self) -> int:
        return tree_hash(self.unwrap())

    # raise helpful error message when trying to interact with frozen object
    __add__ = __radd__ = __iadd__ = lambda x, _: _frozen_error("+", x)
    __sub__ = __rsub__ = __isub__ = lambda x, _: _frozen_error("-", x)
    __mul__ = __rmul__ = __imul__ = lambda x, _: _frozen_error("*", x)
    __matmul__ = __rmatmul__ = __imatmul__ = lambda x, _: _frozen_error("@", x)
    __truediv__ = __rtruediv__ = __itruediv__ = lambda x, _: _frozen_error("/", x)
    __floordiv__ = __rfloordiv__ = __ifloordiv__ = lambda x, _: _frozen_error("//", x)
    __mod__ = __rmod__ = __imod__ = lambda x, _: _frozen_error("%", x)
    __pow__ = __rpow__ = __ipow__ = lambda x, _: _frozen_error("**", x)
    __lshift__ = __rlshift__ = __ilshift__ = lambda x, _: _frozen_error("<<", x)
    __rshift__ = __rrshift__ = __irshift__ = lambda x, _: _frozen_error(">>", x)
    __and__ = __rand__ = __iand__ = lambda x, _: _frozen_error("and", x)
    __xor__ = __rxor__ = __ixor__ = lambda x, _: _frozen_error("xor", x)
    __or__ = __ror__ = __ior__ = lambda x, _: _frozen_error("or", x)
    __neg__ = __pos__ = __abs__ = __invert__ = lambda x: _frozen_error("unary op", x)
    __call__ = lambda x, *_, **__: _frozen_error("call", x)


jtu.register_pytree_node(
    nodetype=_FrozenWrapper,
    flatten_func=lambda tree: ((), _HashableWrapper(tree.unwrap())),
    unflatten_func=lambda treedef, _: _FrozenWrapper(treedef.unwrap()),
)


def freeze(wrapped: Any) -> _FrozenWrapper:
    r"""Freeze a value to avoid updating it by `jax` transformations.

    Example:
        >>> import jax
        >>> import pytreeclass as pytc
        >>> import jax.tree_util as jtu
        >>> # Usage with `jax.tree_util.tree_leaves`
        >>> # no leaves for a wrapped value
        >>> jtu.tree_leaves(pytc.freeze(2.))
        []

        >>> # retrieve the frozen wrapper value using `is_leaf=pytc.is_frozen`
        >>> jtu.tree_leaves(pytc.freeze(2.), is_leaf=pytc.is_frozen)
        [#2.0]

        >>> # Usage with `jax.tree_util.tree_map`
        >>> a= [1,2,3]
        >>> a[1] = pytc.freeze(a[1])
        >>> jtu.tree_map(lambda x:x+100, a)
        [101, #2, 103]

        >>> class Test(pytc.TreeClass):
        ...     a: float
        ...     @jax.value_and_grad
        ...     def __call__(self, x):
        ...         # unfreeze `a` to update it
        ...         self = jax.tree_map(pytc.unfreeze, self, is_leaf=pytc.is_frozen)
        ...         return x ** self.a

        >>> # without `freeze` wrapping `a`, `a` will be updated
        >>> value, grad = Test(a = 2.)(2.)
        >>> print(f"value: {value}\ngrad: {grad}")
        value: 4.0
        grad: Test(a=2.7725887)

        >>> # with `freeze` wrapping `a`, `a` will NOT be updated
        >>> value, grad = Test(a=pytc.freeze(2.))(2.)
        >>> print(f"value: {value}\ngrad: {grad}")
        value: 4.0
        grad: Test(a=#2.0)

        >>> # usage with `jax.tree_map` to freeze a tree
        >>> tree = Test(a = 2.)
        >>> frozen_tree = jax.tree_map(pytc.freeze, tree)
        >>> value, grad = frozen_tree(2.)
        >>> print(f"value: {value}\ngrad: {grad}")
        value: 4.0
        grad: Test(a=#2.0)
    """
    return _FrozenWrapper(wrapped)


def unfreeze(x: Any) -> Any:
    """Unfreeze `frozen` value, otherwise return the value itself.

    - use `is_leaf=pytc.is_frozen` with `jax.tree_map` to unfreeze a tree.**

    Example:
        >>> import pytreeclass as pytc
        >>> import jax
        >>> frozen_value = pytc.freeze(1)
        >>> pytc.unfreeze(frozen_value)
        1
        >>> # usage with `jax.tree_map`
        >>> frozen_tree = jax.tree_map(pytc.freeze, {"a": 1, "b": 2})
        >>> unfrozen_tree = jax.tree_map(pytc.unfreeze, frozen_tree, is_leaf=pytc.is_frozen)
        >>> unfrozen_tree
        {'a': 1, 'b': 2}
    """
    return x.unwrap() if isinstance(x, _FrozenWrapper) else x


def is_frozen(wrapped: Any) -> bool:
    """Returns True if the value is a frozen wrapper."""
    return isinstance(wrapped, _FrozenWrapper)


def is_nondiff(x: Any) -> bool:
    """
    Returns True if the node is a non-differentiable node, and False for if the
    node is of type float, complex number, or a numpy array of floats or
    complex numbers.

    Example:
        >>> import pytreeclass as pytc
        >>> import jax.numpy as jnp
        >>> pytc.is_nondiff(jnp.array(1))  # int array is non-diff type
        True
        >>> pytc.is_nondiff(jnp.array(1.))  # float array is diff type
        False
        >>> pytc.is_nondiff(1)  # int is non-diff type
        True
        >>> pytc.is_nondiff(1.)  # float is diff type
        False

    Note:
        This function is meant to be used with `jax.tree_map` to
        create a mask for non-differentiable nodes in a tree, that can be used
        to freeze the non-differentiable nodes before passing the tree to a
        `jax` transformation.
    """
    if hasattr(x, "dtype") and np.issubdtype(x.dtype, np.inexact):
        return False
    if isinstance(x, (float, complex)):
        return False
    return True


def tree_copy(tree: T) -> T:
    """Return a copy of the tree."""
    leaves, treedef = jtu.tree_flatten(tree)
    return jtu.tree_unflatten(treedef, leaves)


class Partial:
    """
    jaxable Partial function with support for positional partial application.

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
        - The `...` is used to indicate a placeholder for positional arguments.
        - See: https://stackoverflow.com/a/7811270
        - `Partial` is used internally by `bcmap` which maps a function over pytrees
            leaves with automatic broadcasting for scalar arguments.
    """

    __slots__ = ("func", "args", "kwargs", "__weakref__")  # type: ignore

    def __init__(
        self,
        func: Callable[..., Any],
        *args: Any,
        **kwargs: Any,
    ):
        """Initialize a `Partial` function.

        Args:
            func: The function to be partially applied.
            args: Positional arguments to be partially applied. use `...` as a
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


def bcmap(
    func: Callable[..., Any],
    *,
    is_leaf: IsLeafType = None,
) -> Callable:
    """
    (map)s a function over pytrees leaves with automatic (b)road(c)asting
    for scalar arguments

    Args:
        func: the function to be mapped over the pytree
        is_leaf: a predicate function that returns True if the node is a leaf

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

        # avoid circular import by importing Partial here
        from pytreeclass import Partial

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
        Under `jit` the return type is boolean `jax.Array` instead of `bool`.
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


@dc.dataclass(frozen=True)
class NamedSequenceKey:
    idx: int
    key: Hashable

    def __str__(self):
        return f".{self.key}"


def _generate_path_mask(
    tree: PyTree,
    where: tuple[int | str, ...],
    is_leaf: IsLeafType = None,
) -> PyTree:
    # generate a mask for `where` path in `tree`
    # where path is a tuple of indices or keys, for example
    # where=("a",) wil set all leaves of `tree` with key "a" to True and
    # all other leaves to False
    match = False

    def _unpack_entry(entry) -> tuple[Any, ...]:
        # define rule for indexing matching through `at` property
        # in `jax` internals uses `jtu.GetAttrKey` to index an attribute,
        # however its not ergonomic to use `tree.at[jtu.GetAttrKey("attr")]`
        # to index an attribute instead `tree.at['attr']` is more ergonomic
        if isinstance(entry, jtu.GetAttrKey):
            return (entry.name,)
        if isinstance(entry, jtu.SequenceKey):
            return (entry.idx,)
        if isinstance(entry, (jtu.DictKey, jtu.FlattenedIndexKey)):
            return (entry.key,)
        if isinstance(entry, NamedSequenceKey):
            return (entry.idx, entry.key)
        return (entry,)

    def map_func(path: TraceType, _: Any):
        keys, _ = path

        if len(where) > len(keys):
            # path is shorter than `where` path. for example
            # where=("a", "b") and the current path is ("a",) then
            # the current path is not a match
            return False

        for wi, key in zip(where, keys):
            if wi not in (..., *_unpack_entry(key)):
                return False

        nonlocal match

        return (match := True)

    mask = tree_map_with_trace(map_func, tree, is_leaf=is_leaf)

    if not match:
        raise LookupError(f"No leaf match is found for {where=} and {mask=}")

    return mask


def _combine_maybe_bool_masks(*masks: PyTree) -> PyTree:
    # combine boolean masks with `&` operator if all masks are boolean
    # otherwise raise an error
    def check_and_return_bool_leaf(leaf: Any) -> bool:
        if hasattr(leaf, "dtype"):
            if leaf.dtype == "bool":
                return leaf
            raise TypeError(f"Expected boolean array mask, got {leaf=}")

        if isinstance(leaf, bool):
            return leaf
        raise TypeError(f"Expected boolean mask, got {leaf=}")

    def map_func(*leaves):
        verdict = True
        for leaf in leaves:
            verdict &= check_and_return_bool_leaf(leaf)
        return verdict

    return jtu.tree_map(map_func, *masks)


def _resolve_where(
    tree: PyTree,
    where: tuple[int | str | PyTree | EllipsisType, ...],
    is_leaf: IsLeafType = None,
):
    mask = None

    if path := [i for i in where if isinstance(i, (int, str, type(...)))]:
        mask = _generate_path_mask(tree, path, is_leaf=is_leaf)

    if maybe_bool_masks := [i for i in where if isinstance(i, type(tree))]:
        all_masks = [mask, *maybe_bool_masks] if mask else maybe_bool_masks
        mask = _combine_maybe_bool_masks(*all_masks)

    return mask


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
    is_trace_leaf: Callable[[TraceEntry], bool] | None = None,
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
    r"""
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
    __slots__ = ("data", "parent", "children", "__weakref__")

    def __init__(self, data: tuple[Hashable, type, Any]):
        self.data = data
        self.parent = None
        self.children = {}

    def add_child(self, child: Node) -> None:
        # add child node to this node and set
        # this node as the parent of the child
        if not isinstance(child, Node):
            raise TypeError(f"`child` must be a `Node`, got {type(child)}")
        key, _, __ = child.data
        if key not in self.children:
            # establish parent-child relationship
            child.parent = self
            self.children[key] = child

    def __iter__(self) -> Iterator[Node]:
        # iterate over children nodes
        return iter(self.children.values())

    def __repr__(self) -> str:
        return f"Node(data={self.data})"


# disallow traversal to avoid infinite recursion
# in case of circular references
jtu.register_pytree_node(
    nodetype=Node,
    flatten_func=lambda tree: ((), tree),
    unflatten_func=lambda treedef, _: treedef[0],
)


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

    value = tree if len(traces_leaves) == 1 else None
    root = Node(data=(None, tree.__class__, value))

    for trace, leaf in traces_leaves:
        keys, types = trace
        cur = root
        for i, (key, type) in enumerate(zip(keys, types)):
            if key in cur.children:
                # common parent node
                cur = cur.children[key]
            else:
                # new path
                value = leaf if i == len(keys) - 1 else None
                child = Node(data=(key, type, value))
                cur.add_child(child)
                cur = child
    return root
