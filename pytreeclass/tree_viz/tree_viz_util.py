from __future__ import annotations

import dataclasses as dc
import math
import sys
from typing import Any, NamedTuple

import numpy as np

from pytreeclass._src.tree_freeze import _FrozenWrapper

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

    elif isinstance(node_size, (float, int)):
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
    elif isinstance(node_count, (float, int)):
        return f"{int(node_count):,}"

    raise TypeError(f"node_count must be int or float, got {type(node_count)}")


def is_children_frozen(tree):
    """assert if a dataclass is static"""
    if dc.is_dataclass(tree):
        fields = dc.fields(tree)
        if len(fields) > 0:
            if all(isinstance(f, _FrozenWrapper) for f in fields):
                return True
            if all(isinstance(getattr(tree, f.name), _FrozenWrapper) for f in fields):
                return True

    return False


def _mermaid_marker(field_item: dc.Field, node: Any, default: str = "--") -> str:
    """return the suitable marker given the field and node item

    Args:
        field_item (Field): field item of the pytree node
        node (Any): node item
        default (str, optional): default marker. Defaults to "".

    Returns:
        str: marker character.
    """
    # for now, we only have two markers '*' for non-diff and '#' for frozen
    if isinstance(field_item, _FrozenWrapper) or is_children_frozen(node):
        return "--x"
    return default


def _marker(field_item: dc.Field, node: Any, default: str = "") -> str:
    """return the suitable marker given the field and node item"""
    # '*' for non-diff

    if isinstance(field_item, _FrozenWrapper) or is_children_frozen(node):
        return "#"

    return default


# def _sequential_tree_shape_eval(tree, array):
#     """Evaluate shape propagation of assumed sequential modules"""
#     dyanmic, static = dcu._dataclass_structure(tree)

#     # all dynamic/static leaves
#     all_leaves = (*dyanmic.values(), *static.values())
#     leaves = [leaf for leaf in all_leaves if dc.is_dataclass(leaf)]

#     shape = [jax.eval_shape(lambda x: x, array)]
#     for leave in leaves:
#         shape += [jax.eval_shape(leave, shape[-1])]
#     return shape


class NodeInfo(NamedTuple):
    node: Any
    path: str
    frozen: bool = False
    repr: bool = True
    count: int = 0
    size: int = 0


def tree_trace(tree: PyTree, depth=float("inf")) -> list[NodeInfo]:
    """trace and flatten a a PyTree to a list of `NodeInfo` objects

    Args:
        tree : the PyTree to be traced
        depth : the depth to be traced. Defaults to float("inf") for trace till the leaf.

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
        NodeInfo(node=1, path='[0]', frozen=False, repr=True, count=1j, size=28j)
        NodeInfo(node=2, path='[1]', frozen=False, repr=True, count=1j, size=28j)
        NodeInfo(node=3, path='[2]', frozen=False, repr=True, count=1j, size=28j)
    """

    def container_flatten(info: NodeInfo, depth: int):
        for i, item in enumerate(info.node):
            frozen = info.frozen or isinstance(item, _FrozenWrapper)
            sub_info = NodeInfo(item, f"{info.path}[{i}]", frozen, info.repr)
            yield from tree_flatten_recurse(sub_info, depth - 1)

    def dict_flatten(info: NodeInfo, depth: int):
        for key, item in info.node.items():
            frozen = info.frozen or isinstance(item, _FrozenWrapper)
            sub_info = NodeInfo(item, f"{info.path}[{key}]", frozen, info.repr)
            yield from tree_flatten_recurse(sub_info, depth - 1)

    def dcls_flatten(info: NodeInfo, depth: int):
        for field in dc.fields(info.node):
            node = getattr(info.node, field.name)
            path = f"{info.path}" + ("." if len(info.path) > 0 else "") + field.name
            frozen = info.frozen or isinstance(field, _FrozenWrapper)
            frozen = frozen or isinstance(node, _FrozenWrapper)
            sub_info = NodeInfo(node, path, frozen, info.repr)
            yield from tree_flatten_recurse(sub_info, depth - 1)

    def leaf_count_and_size_flatten(info: NodeInfo) -> NodeInfo:
        """count and size of the leaf node in the form of (inexact, exact) complex number"""
        count, size = complex(0, 0), complex(0, 0)

        for sub_info in tree_flatten_recurse(info, float("inf")):
            node = sub_info.node
            if hasattr(node, "shape") and hasattr(node, "dtype"):
                if np.issubdtype(node.dtype, np.inexact):
                    # inexact(trainable) array
                    count += complex(int(np.array(node.shape).prod()), 0)
                    size += complex(int(node.nbytes), 0)
                else:
                    # non in-exact paramter
                    count += complex(0, int(np.array(node.shape).prod()))
                    size += complex(0, int(node.nbytes))

            elif isinstance(node, (float, complex)):
                # inexact non-array (array_like)
                count += complex(1, 0)
                size += complex(sys.getsizeof(node), 0)

            else:
                count += complex(0, 1)
                size += complex(0, sys.getsizeof(node))

        return info._replace(count=count, size=size)

    def tree_flatten_recurse(info: NodeInfo, depth):
        if depth < 1:
            yield info

        elif isinstance(info.node, tuple) and hasattr(info.node, "_fields"):
            yield from dict_flatten(info._asdict(), depth)

        elif isinstance(info.node, (list, tuple)):
            yield from container_flatten(info, depth)

        elif isinstance(info.node, dict):
            yield from dict_flatten(info, depth)

        elif dc.is_dataclass(info.node):
            yield from dcls_flatten(info, depth)

        else:
            yield info

    info = NodeInfo(node=tree, path="", frozen=False, repr=True)
    return list(map(leaf_count_and_size_flatten, tree_flatten_recurse(info, depth)))
