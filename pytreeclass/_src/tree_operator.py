from __future__ import annotations

import functools as ft
import hashlib
import inspect
import operator as op
from typing import Any, Callable

import jax.tree_util as jtu
import numpy as np

"""A wrapper around a tree that allows to use the tree leaves as if they were scalars."""

PyTree = Any


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
def _transform_to_kwds_func(func: Callable) -> Callable:
    """Convert a function to a all keyword accepting function to use `functools.partial` on any arg

    Args:
        func : the function to be transformed to a all keyword args function

    Raises:
        ValueError: in case the function has variable length args
        ValueError: in case the function has variable length keyword args

    Returns:
        Callable: transformed function with all args as keyword args

    Example:
        >>> @_transform_to_kwds_func
        ... def func(x,y=1,*,z=2):
        ...     return x+y+z
        >>> func
        <function <lambda>(x=None, y=1, z=2)>
    """
    # get the function signature
    args, _args, _kwds, args_default, _, kwds_default, _ = inspect.getfullargspec(func)

    # Its not possible to use *args or **kwds, because its not possible to
    # get their argnames and default values
    if _args is not None:
        raise ValueError("Variable length args are not supported")

    if _kwds is not None:
        raise ValueError("Variable length keyword args are not supported")

    del _args, _kwds

    # convert the args_default to a dict with the arg name as key
    # if defaults are not provided, use `None`
    args_default = args_default or ()
    args_default = (None,) * (len(args) - len(args_default)) + args_default
    args_default = dict(zip(args, args_default))

    kwds_default = kwds_default or {}

    del args

    # positional args encoded as keyword args
    lhs = ", ".join(f"{key}={args_default[key]}" for key in args_default)

    # keyword-only args are encoded as keyword args
    lhs += (("," + ", ".join(f"{k}={kwds_default[k]}" for k in kwds_default))) if kwds_default else ""  # fmt: skip

    # reference the keyword only args by their name inside function
    rhs = ", ".join(f"{key}" for key in args_default)
    rhs += ("," + ",".join(f"{k}={k}" for k in kwds_default)) if kwds_default else ""

    kwd_func = eval(f"lambda {lhs} : func({rhs})", {"func": func})

    # copy original function docstring and add a note about the keyword only args
    kwd_func.__doc__ = f"(keyworded transformed function) {func.__doc__}"
    return kwd_func


@ft.lru_cache(maxsize=None)
def bmap(
    func: Callable[..., Any], *, is_leaf: Callable[[Any], bool] | None = None
) -> Callable:

    """Maps a function over pytrees leaves with automatic broadcasting for scalar arguments.

    Example:
        >>> @pytc.treeclass
        ... class Test:
        ...    a: int = (1,2,3)
        ...    b: int = (4,5,6)
        ...    c: jnp.ndarray = jnp.array([1,2,3])

        >>> tree = Test()
        >>> # 0 is broadcasted to all leaves of the pytree

        >>> print(pytc.bmap(jnp.where)(tree>1, tree, 0))
        Test(a=(0,2,3), b=(4,5,6), c=[0 2 3])

        >>> print(pytc.bmap(jnp.where)(tree>1, 0, tree))
        Test(a=(1,0,0), b=(0,0,0), c=[1 0 0])

        >>> # 1 is broadcasted to all leaves of the list pytree
        >>> bmap(lambda x,y:x+y)([1,2,3],1)
        [2, 3, 4]

        >>> # trees are summed leaf-wise
        >>> bmap(lambda x,y:x+y)([1,2,3],[1,2,3])
        [2, 4, 6]

        >>> # Non scalar second args case
        >>> bmap(lambda x,y:x+y)([1,2,3],[[1,2,3],[1,2,3]])
        TypeError: unsupported operand type(s) for +: 'int' and 'list'

        >>> # using **numpy** functions on pytrees
        >>> import jax.numpy as jnp
        >>> bmap(jnp.add)([1,2,3],[1,2,3])
        [DeviceArray(2, dtype=int32, weak_type=True),
        DeviceArray(4, dtype=int32, weak_type=True),
        DeviceArray(6, dtype=int32, weak_type=True)]
    """
    # The **prime motivation** for this function is to allow the
    # use of numpy functions on pytrees by decorating the numpy function.
    # for example, the following codes are equivalent:
    # >>> jtu.tree_map(np.add, jnp.array([1,2,3]), jnp.array([1,2,3]))
    # >>> bmap(np.add)(jnp.array([1,2,3]), jnp.array([1,2,3])
    # In case of all arguments are of the same structure, the function is equivalent to
    # `bmap` <=> `ft.partial(ft.partial, jtu.tree_map)`

    signature = inspect.getfullargspec(func)
    arg_names = signature[0]

    # transform the function to a keyword accepting function to
    # make it possible to use `functools.partial`
    kwd_func = _transform_to_kwds_func(func)

    @ft.wraps(func)
    def wrapper(*args, **kwds):

        if len(args) == 0:
            # the user provided only keyword args, then we fetch the first arg
            # by its name from the keywords
            leaves, treedef = jtu.tree_flatten(kwds[arg_names[0]], is_leaf=is_leaf)
            del kwds[arg_names[0]]

        else:
            # the user provided positional args
            leaves, treedef = jtu.tree_flatten(args[0], is_leaf=is_leaf)

        partial_args = dict()
        non_partial_args = dict()

        # handle positional args except the first one
        # as we are comparing aginast the first arg
        for (key, value) in zip(arg_names[1:], args[1:]):
            if jtu.tree_structure(value) == treedef:
                # similar pytree structure arguments are not broadcasted
                non_partial_args[key] = value
            else:
                # different pytree structure arguments are broadcasted
                # might be a scalar or a pytree with a different structure
                # in case of different structure, the function will fail
                partial_args[key] = value

        # handle keyword-only args
        for (key, value) in kwds.items():
            if jtu.tree_structure(value) == treedef:
                non_partial_args[key] = kwds[key]
            else:
                partial_args[key] = kwds[key]

        all_leaves = [leaves]
        all_leaves += [treedef.flatten_up_to(r) for r in non_partial_args.values()]

        # pass the leaves values to the function by argnames
        partial_func = ft.partial(kwd_func, **partial_args)
        argnames = [arg_names[0]] + list(non_partial_args.keys())
        flattened = (partial_func(**dict(zip(argnames, xs))) for xs in zip(*all_leaves))
        return treedef.unflatten(flattened)

    wrapper.__doc__ = f"(broadcasted function) {func.__doc__}"

    return wrapper


class _TreeOperator:
    """Base class for tree operators used

    Example:
        >>> import jax.tree_util as jtu
        >>> import dataclasses as dc
        >>> @jtu.register_pytree_node_class
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

    __copy__ = _copy
    __hash__ = _hash
    __abs__ = bmap(op.abs)
    __add__ = bmap(op.add)
    __radd__ = bmap(op.add)
    __and__ = bmap(op.and_)
    __rand__ = bmap(op.and_)
    __eq__ = bmap(op.eq)
    __floordiv__ = bmap(op.floordiv)
    __ge__ = bmap(op.ge)
    __gt__ = bmap(op.gt)
    __inv__ = bmap(op.inv)
    __invert__ = bmap(op.invert)
    __le__ = bmap(op.le)
    __lshift__ = bmap(op.lshift)
    __lt__ = bmap(op.lt)
    __matmul__ = bmap(op.matmul)
    __mod__ = bmap(op.mod)
    __mul__ = bmap(op.mul)
    __rmul__ = bmap(op.mul)
    __ne__ = bmap(op.ne)
    __neg__ = bmap(op.neg)
    __not__ = bmap(op.not_)
    __or__ = bmap(op.or_)
    __pos__ = bmap(op.pos)
    __pow__ = bmap(op.pow)
    __rshift__ = bmap(op.rshift)
    __sub__ = bmap(op.sub)
    __rsub__ = bmap(op.sub)
    __truediv__ = bmap(op.truediv)
    __xor__ = bmap(op.xor)
