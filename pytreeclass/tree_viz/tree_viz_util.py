from __future__ import annotations

import dataclasses as dc
import math
import sys
from collections import defaultdict
from typing import Any, NamedTuple

import numpy as np
from jax._src.tree_util import _registry

from pytreeclass._src.tree_freeze import is_frozen

PyTree = Any


def _format_width(string, width=60):
    """strip newline/tab characters if less than max width"""
    children_length = len(string) - string.count("\n") - string.count("\t")
    if children_length > width:
        return string
    return string.replace("\n", "").replace("\t", "")


def _format_size(node_size, newline=False):
    """return formatted size from inexact(exact) complex number

    Examples:
        >>> _format_size(1024)
        '1.00KB'
        >>> _format_size(1024**2)
        '1.00MB'
    """
    mark = "\n" if newline else ""
    order_kw = ["B", "KB", "MB", "GB", "TB", "PB", "EB", "ZB", "YB"]

    if isinstance(node_size, complex):
        # define order of magnitude
        real_size_order = int(math.log(max(node_size.real, 1), 1024))
        imag_size_order = int(math.log(max(node_size.imag, 1), 1024))
        fmt = f"{(node_size.real)/(1024**real_size_order):.2f}{order_kw[real_size_order]}{mark}"
        fmt += f"({(node_size.imag)/(1024**imag_size_order):.2f}{order_kw[imag_size_order]})"
        return fmt

    if isinstance(node_size, (float, int)):
        size_order = int(math.log(node_size, 1024)) if node_size > 0 else 0
        return f"{(node_size)/(1024**size_order):.2f}{order_kw[size_order]}"

    raise TypeError(f"node_size must be int or float, got {type(node_size)}")


def _format_count(node_count, newline=False):
    """return formatted count from inexact(exact) complex number

    Examples:
        >>> _format_count(1024)
        '1,024'

        >>> _format_count(1024**2)
        '1,048,576'
    """

    mark = "\n" if newline else ""

    if isinstance(node_count, complex):
        return f"{int(node_count.real):,}{mark}({int(node_count.imag):,})"

    if isinstance(node_count, (float, int)):
        return f"{int(node_count):,}"

    raise TypeError(f"node_count must be int or float, got {type(node_count)}")


def is_children_frozen(tree):
    """assert if a dataclass is static"""
    if dc.is_dataclass(tree):
        fields = dc.fields(tree)
        if len(fields) > 0:
            if all(is_frozen(f) for f in fields):
                return True
            if all(is_frozen(getattr(tree, f.name)) for f in fields):
                return True

    return False


def _mermaid_marker(field: dc.Field, node: Any, default: str = "--") -> str:
    """return the suitable marker given the field and node item

    Args:
        field (Field): field item of the pytree node
        node (Any): node item
        default (str, optional): default marker. Defaults to "".

    Returns:
        str: marker character.
    """
    if is_frozen(field) or is_children_frozen(node):
        return "--x"
    return default


def _marker(field: dc.Field, node: Any, default: str = "") -> str:
    """return the suitable marker given the field and node item"""
    # if all your children are frozen, the you are frozen
    if is_frozen(field) or is_children_frozen(node):
        return "#"

    return default


class NodeInfo(NamedTuple):
    node: Any
    path: str
    frozen: bool = False
    repr: bool = True
    count: complex = complex(0, 0)
    size: complex = complex(0, 0)
    stats: dict[str, int] | None = None


def _get_type(node: Any) -> str:
    if hasattr(node, "dtype"):
        return (
            str(node.dtype)
            .replace("float", "f")
            .replace("int", "i")
            .replace("complex", "c")
        )
    return node.__class__.__name__


def tree_trace(tree: PyTree, depth=float("inf")) -> list[NodeInfo]:
    """trace the tree and return a list of NodeInfo objects
    Returns:
        list of NodeInfo objects containing
            node: the node itself
            path: the path to the node.
            frozen: the marker for the node in case of a dataclass node with a marker field
            repr: whether to use repr or str for the node in case of a dataclass node
            count: the number of parameters in the node in form of (inexact, exact) complex number
            size: the size of the parameters in the node in form of (inexact, exact) complex number

    Example:
        >>> for leaf in pytc.tree_viz.utils.tree_trace((1,2,3)):
        ...    print(leaf)
        NodeInfo(node=1, path=['[0]'], frozen=False, repr=True, count=1j, size=28j)
        NodeInfo(node=2, path=['[1]'], frozen=False, repr=True, count=1j, size=28j)
        NodeInfo(node=3, path=['[2]'], frozen=False, repr=True, count=1j, size=28j)
    """
    # unlike `jax.tree_util.tree_leaves`, this function can flatten tree at a certain depth
    # for instance, a depth of 0 will return the tree itself, and a depth of 1 will return the
    # the direct discedents of the tree. depth of `inf` is equivalent to `jax.tree_util.tree_leaves`
    # the returned list is a list of NodeInfo objects, which contains the node itself, the path to
    # the node, and some stats about the path size and number of leaves.
    # this is done on two steps: the first is to reach to the desired depth using recurisve calls
    # defined by `yield_one_level_leaves`, and the second is to calculate the stats of the nodes using
    # `yield_stats`, which picks up the yield leaves from the first step and recurse to the max tree depth.

    def yield_one_level_leaves(info: NodeInfo, depth: int) -> list[NodeInfo]:
        if is_frozen(info.node):
            info = info._replace(node=info.node.unwrap(), frozen=True)
            yield from yield_one_level_leaves(info, depth)

        elif _registry.get(type(info.node)):
            # the tree is JAX pytree and has a flatten handler
            handler = _registry.get(type(info.node))
            leaves = list(handler.to_iter(info.node)[0])

            # check for python datastructure that has named paths first
            # otherwise assign `leaf_` for each leaf.
            if isinstance(info.node, tuple) and hasattr(info.node, "_fields"):
                names = (f"{f}" for f in info.node._fields)
                reprs = (info.repr,) * len(info.node)  # inherit repr from parent
            elif isinstance(info.node, dict):
                names = (f"{key}" for key in info.node)
                reprs = (info.repr,) * len(info.node)
            elif isinstance(info.node, (list, tuple)):
                names = (f"[{i}]" for i in range(len(info.node)))
                reprs = (info.repr,) * len(info.node)  # inherit repr from parent
            elif dc.is_dataclass(info.node):
                names = (f"{f.name}" for f in dc.fields(info.node))
                reprs = (info.repr and f.repr for f in dc.fields(info.node))
            else:
                names = (f"leaf_{i}" for i in range(len(leaves)))
                reprs = (info.repr,) * len(leaves)

            for name, repr, leaf in zip(names, reprs, leaves):
                sub_info = NodeInfo(leaf, info.path + [name], info.frozen, repr)
                yield from (recurse_step(sub_info, depth - 1))

        else:
            # the node is not a JAX pytree
            yield info

    def yield_subtree_stats(info: NodeInfo) -> NodeInfo:
        # calcuate some states of a single subtree defined by the `NodeInfo` objects
        # for each subtree, we will calculate the types distribution and their size
        stats = defaultdict(lambda: 0)
        count = size = 0

        for sub_info in recurse_step(info, float("inf")):
            type = _get_type(sub_info.node)
            node = sub_info.node
            _count = int(np.array(node.shape).prod()) if hasattr(node, "shape") else 1
            _count = complex(0, _count) if sub_info.frozen else _count
            stats[type] += int(_count.imag + _count.real)
            count += _count

            _size = node.nbytes if hasattr(node, "nbytes") else sys.getsizeof(node)
            _size = complex(0, _size) if sub_info.frozen else _size
            size += _size

        return info._replace(count=count, size=size, stats=stats)

    def recurse_step(info: NodeInfo, depth):
        yield from yield_one_level_leaves(info, depth) if depth > 0 else [info]

    info = NodeInfo(node=tree, path=[], frozen=False, repr=True)
    return list(map(yield_subtree_stats, recurse_step(info, depth)))
