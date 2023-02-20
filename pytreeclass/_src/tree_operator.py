from __future__ import annotations

import functools as ft
import hashlib
import inspect
import math
import operator as op
from typing import Any, Callable

import jax.tree_util as jtu
import numpy as np

"""A wrapper around a tree that allows to use the tree leaves as if they were scalars."""

PyTree = Any
_empty = inspect.Parameter.empty


def _hash_node(node):
    if hasattr(node, "dtype") and hasattr(node, "shape"):
        return hashlib.sha256(np.array(node).tobytes()).hexdigest()
    if isinstance(node, set):
        return hash(frozenset(node))
    if isinstance(node, dict):
        return hash(frozenset(node.items()))
    if isinstance(node, list):
        return hash(tuple(node))
    return hash(node)


def _hash(tree):
    hashed = jtu.tree_map(_hash_node, jtu.tree_leaves(tree))
    return hash((*hashed, jtu.tree_structure(tree)))


def _copy(tree: PyTree) -> PyTree:
    """Return a copy of the tree"""
    return jtu.tree_unflatten(*jtu.tree_flatten(tree)[::-1])


@ft.lru_cache(maxsize=None)
def _transform_to_pos_or_kwd_func(func: Callable) -> Callable:
    """Function transformation to convert a function pos only and kwd only args to pos or kwd args.

    Args:
        func : the function to be transformed to a all keyword args function

    Raises:
        ValueError: in case the function has variable length args
        ValueError: in case the function has variable length keyword args

    Returns:
        Callable: transformed function with all args as keyword args

    Example:
        >>> @_transform_to_pos_or_kwd_func
        ... def func(a,b,/,*,c=1):
        ...     return a+b
        >>> func
        <function __main__.f(a, b, c=1)>  # removed / and * from the signature
    """

    sig = inspect.signature(func)
    params = list(sig.parameters.values())

    lhs, rhs = "lambda ", ""

    for i, param in enumerate(params):
        if param.kind == param.VAR_POSITIONAL or param.kind == param.VAR_KEYWORD:
            raise ValueError("Variable length argument is not supported")

        if param.kind != param.POSITIONAL_OR_KEYWORD:
            params[i] = param.replace(kind=param.POSITIONAL_OR_KEYWORD)

        lhs += f"{param.name}"
        lhs += f"={param.default}," if param.default is not _empty else ","
        rhs += f"{param.name}"
        rhs += f"={param.name}," if param.kind == param.KEYWORD_ONLY else ","

    new_func = eval(f"{lhs[:-1]}:func({rhs[:-1]})", {"func": func})
    new_func = ft.wraps(func)(new_func)
    new_func.__signature__ = sig.replace(parameters=params)
    return new_func


@ft.lru_cache(maxsize=None)
def bcmap(
    func: Callable[..., Any],
    *,
    is_leaf: Callable[[Any], bool] | None = None,
    broadcasted_argnames: tuple[str, ...] = None,
    broadcasted_argnums: tuple[int, ...] = None,
) -> Callable:

    """(map)s a function over pytrees leaves with automatic (b)road(c)asting for scalar arguments,
    or broadcast for chosen argnames or argnums.

    Args:
        func: the function to be mapped over the pytree
        is_leaf: a function that returns True if the argument is a leaf of the pytree
        broadcasted_argnames: the indices of the arguments that should be broadcasted.
        broadcasted_argnums: the names of the arguments that should be broadcasted

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
    # The **prime motivation** for this function is to allow the
    # use of numpy functions on pytrees by decorating the numpy function.
    # for example, the following codes are equivalent:
    # >>> jtu.tree_map(np.add, jnp.array([1,2,3]), jnp.array([1,2,3]))
    # >>> bcmap(np.add)(jnp.array([1,2,3]), jnp.array([1,2,3])
    # In case of all arguments are of the same structure, the function is equivalent to
    # `bcmap` <=> `ft.partial(ft.partial, jtu.tree_map)`

    signature = inspect.getfullargspec(func)
    sig_argnames = signature[0]

    # remove * and / from the signature to make it possible to use `functools.partial`
    kwd_or_pos_func = _transform_to_pos_or_kwd_func(func)

    @ft.wraps(kwd_or_pos_func)
    def wrapper(*args, **kwds):
        # automatically to handle the broadcasting of scalar arguments to pytree leaves
        broadcast_kwds, no_broadcast_kwds = dict(), dict()

        if broadcasted_argnums or broadcasted_argnames:
            # here we handle the case where the user provides which
            # argnum and argname to broadcast over

            if broadcasted_argnums:
                # the user provided positional args to broadcast
                for name, argnum in zip(sig_argnames, range(len(args))):
                    if argnum in broadcasted_argnums:
                        broadcast_kwds[name] = args[argnum]

            if broadcasted_argnames:
                # the user provided keyword args to broadcast
                for name in broadcasted_argnames:
                    broadcast_kwds[name] = kwds[name]

            # handle non-broadcasted arg
            for (key, value) in zip(sig_argnames, args):
                if key not in broadcast_kwds:
                    no_broadcast_kwds[key] = value

            for key in kwds:
                if key not in broadcast_kwds:
                    no_broadcast_kwds[key] = kwds[key]

            # fetch the first arg/kwd to broadcast against
            # this arg/kwd is the first non-broadcasted arg/kwd
            name0 = next(iter(no_broadcast_kwds))
            node0 = no_broadcast_kwds.pop(name0)
            leaves, treedef = jtu.tree_flatten(node0, is_leaf=is_leaf)

        else:
            # here we handle the case where the user does not provide which argnum and argname
            # to broadcast over, so we broadcast all the arguments that dont have same
            # same pytree structure as the first. this is the default behavior of `bcmap`
            # ex: [1,2,3], 1, [3,4] => 1 and [3,4] will be broadcasted (partialized) to [1,2,3]
            # this will throw an error later because [3,4] is not broadcastable to [1,2,3]
            # but we proceed with the assumption that the user knows what he is doing

            if len(args) == 0:
                # the user provided only keyword args
                # we fetch the first arg by its name from the keywords
                node0 = kwds.pop(sig_argnames[0])
                leaves, treedef = jtu.tree_flatten(node0, is_leaf=is_leaf)
                name0 = sig_argnames[0]

            else:
                # the user provided positional args
                leaves, treedef = jtu.tree_flatten(args[0], is_leaf=is_leaf)
                name0 = sig_argnames[0]

            # handle positional args except the first one
            # as we are comparing aginast the first arg
            for (key, value) in zip(sig_argnames[1:], args[1:]):
                if jtu.tree_structure(value) == treedef:
                    # similar pytree structure arguments are not broadcasted (i.e. not partialized)
                    no_broadcast_kwds[key] = value
                else:
                    # different pytree structure arguments are broadcasted
                    # might be a scalar or a pytree with a different structure
                    # in case of different structure, the function will fail
                    broadcast_kwds[key] = value

            # handle keyword-only args
            for key in kwds:
                if jtu.tree_structure(kwds[key]) == treedef:
                    no_broadcast_kwds[key] = kwds[key]
                else:
                    broadcast_kwds[key] = kwds[key]

        all_leaves = [leaves]
        all_leaves += [treedef.flatten_up_to(r) for r in no_broadcast_kwds.values()]

        # pass the leaves values to the function by argnames
        # without kwd_func we would have to pass the leaves as positional arguments
        # which would not work if a middle arg is needed to be broadcasted
        # this is why we need to transform the function to a keyword accepting function
        partial_func = ft.partial(kwd_or_pos_func, **broadcast_kwds)
        names = [name0] + list(no_broadcast_kwds.keys())
        flattened = (partial_func(**dict(zip(names, xs))) for xs in zip(*all_leaves))
        return treedef.unflatten(flattened)

    wrapper.__doc__ = f"(broadcasted function) {func.__doc__}"
    return wrapper


class _TreeOperator:
    """Base class for tree operators used

    Example:
        >>> import jax.tree_util as jtu
        >>> import dataclasses as dc
        >>> @jtu.register_pytree_node_class`
        ... @dc.dataclass
        ... class Tree(_TreeOperator):
        ...    a: int =1
        ...    def tree_flatten(self):
        ...        return (self.a,), None
        ...    @classmethod
        ...    def tree_unflatten(cls, _, children):
        ...        return cls(*children)

        >>> tree = Tree()
        >>> tree + 1
        Tree(a=2)
    """

    __abs__ = bcmap(op.abs)
    __add__ = bcmap(op.add)
    __and__ = bcmap(op.and_)
    __ceil__ = bcmap(math.ceil)
    __copy__ = _copy
    __divmod__ = bcmap(divmod)
    __eq__ = bcmap(op.eq)
    __floor__ = bcmap(math.floor)
    __floordiv__ = bcmap(op.floordiv)
    __ge__ = bcmap(op.ge)
    __gt__ = bcmap(op.gt)
    __inv__ = bcmap(op.inv)
    __invert__ = bcmap(op.invert)
    __le__ = bcmap(op.le)
    __lshift__ = bcmap(op.lshift)
    __lt__ = bcmap(op.lt)
    __matmul__ = bcmap(op.matmul)
    __mod__ = bcmap(op.mod)
    __mul__ = bcmap(op.mul)
    __ne__ = bcmap(op.ne)
    __neg__ = bcmap(op.neg)
    __or__ = bcmap(op.or_)
    __pos__ = bcmap(op.pos)
    __pow__ = bcmap(op.pow)
    __radd__ = bcmap(op.add)
    __rand__ = bcmap(op.and_)
    __rdivmod__ = bcmap(divmod)
    __rfloordiv__ = bcmap(op.floordiv)
    __rlshift__ = bcmap(op.lshift)
    __rmod__ = bcmap(op.mod)
    __rmul__ = bcmap(op.mul)
    __ror__ = bcmap(op.or_)
    __round__ = bcmap(round)
    __rpow__ = bcmap(op.pow)
    __rrshift__ = bcmap(op.rshift)
    __rshift__ = bcmap(op.rshift)
    __rsub__ = bcmap(op.sub)
    __rtruediv__ = bcmap(op.truediv)
    __rxor__ = bcmap(op.xor)
    __sub__ = bcmap(op.sub)
    __truediv__ = bcmap(op.truediv)
    __trunk__ = bcmap(math.trunc)
    __xor__ = bcmap(op.xor)
    __hash__ = _hash
