from __future__ import annotations

import inspect
import math
from typing import Any, Callable

import numpy as np


def _format_width(string, width=60):
    """strip newline/tab characters if less than max width"""
    children_length = len(string) - string.count("\n") - string.count("\t")
    if children_length > width:
        return string
    return string.replace("\n", "").replace("\t", "")


def _numpy_repr(node: np.ndarray) -> str:
    """Replace np.ndarray repr with short hand notation for type and shape

    Args:
        node: numpy array

    Returns:
        str: short hand notation for type and shape

    Example:
        >>> _numpy_repr(np.ones((2,3)))
        'f32[2,3]'
    """
    shape = (
        f"{node.shape}".replace(",", "")
        .replace("(", "[")
        .replace(")", "]")
        .replace(" ", ",")
    )

    if issubclass(node.dtype.type, np.integer):
        dtype = f"{node.dtype}".replace("int", "i")
    elif issubclass(node.dtype.type, np.floating):
        dtype = f"{node.dtype}".replace("float", "f")
    elif issubclass(node.dtype.type, np.complexfloating):
        dtype = f"{node.dtype}".replace("complex", "c")
    else:
        dtype = f"{node.dtype}"

    return f"{dtype}{shape}"


def _numpy_extended_repr(node: np.ndarray) -> str:
    """Adds information about min, max, mean, and std to _numpy_repr
    This function is inspired by https://github.com/xl0/lovely-jax

    Args:
        node: numpy array

    Returns:
        str: short hand notation for type and shape

    Example:
        >>> _numpy_extended_repr(np.ones((2,3)))
        f64[2,3]<∈[1.00,1.00],μ≈1.00,σ≈0.00>
    """
    shape_dtype = _numpy_repr(node)
    if issubclass(node.dtype.type, np.number):
        low, high = np.min(node), np.max(node)
        interval = "(" if math.isinf(low) else "["
        std, mean = np.std(node), np.mean(node)
        interval += (
            f"{low},{high}"
            if issubclass(node.dtype.type, np.integer)
            else f"{low:.2f},{high:.2f}"
        )
        interval += ")" if math.isinf(high) else "]"
        interval = interval.replace("inf", "∞")
        return f"{shape_dtype}<∈{interval},μ≈{mean:.2f},σ≈{std:.2f}>"

    return shape_dtype


def _func_repr(func: Callable) -> str:
    """Pretty print function

    Args:
        func (Callable): function to be printed

    Returns:
        str: pretty printed function

    Example:
        >>> def example(a: int, b=1, *c, d, e=2, **f) -> str:
            ...
        >>> _func_repr(example)
        "example(a,b,*c,d,e,**f)"
    """
    args, varargs, varkw, _, kwonlyargs, _, _ = inspect.getfullargspec(func)
    args = (",".join(args)) if len(args) > 0 else ""
    varargs = ("*" + varargs) if varargs is not None else ""
    kwonlyargs = (",".join(kwonlyargs)) if len(kwonlyargs) > 0 else ""
    varkw = ("**" + varkw) if varkw is not None else ""
    name = "Lambda" if (func.__name__ == "<lambda>") else func.__name__

    fmt = f"{name}("
    fmt += ",".join(item for item in [args, varargs, kwonlyargs, varkw] if item != "")
    fmt += ")"
    return fmt


def _list_repr(node: list, depth: int) -> str:
    fmt = (f"{_format_width(_format_node_repr(v,depth=depth+1))}" for v in node)
    fmt = (",\n" + "\t" * (depth + 1)).join(fmt)
    fmt = "[\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + "]"
    return _format_width(fmt)


def _list_str(node: list, depth: int) -> str:
    fmt = (f"{_format_width(_format_node_str(v,depth=depth+1))}" for v in node)
    fmt = (",\n" + "\t" * (depth + 1)).join(fmt)
    fmt = "[\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + "]"
    return _format_width(fmt)


def _tuple_repr(node: tuple, depth: int) -> str:
    fmt = (f"{_format_width(_format_node_repr(v,depth=depth+1))}" for v in node)
    fmt = (",\n" + "\t" * (depth + 1)).join(fmt)
    fmt = "(\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + ")"
    return _format_width(fmt)


def _tuple_str(node: tuple, depth: int) -> str:
    fmt = (f"{_format_width(_format_node_str(v,depth=depth+1))}" for v in node)
    fmt = (",\n" + "\t" * (depth + 1)).join(fmt)
    fmt = "(\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + ")"
    return _format_width(fmt)


def _set_repr(node: set, depth: int) -> str:
    fmt = (f"{_format_width(_format_node_repr(v,depth=depth+1))}" for v in node)
    fmt = (",\n" + "\t" * (depth + 1)).join(fmt)
    fmt = "{\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + "}"
    return _format_width(fmt)


def _set_str(node: set, depth: int) -> str:
    fmt = (f"{_format_width(_format_node_str(v,depth=depth+1))}" for v in node)
    fmt = (",\n" + "\t" * (depth + 1)).join(fmt)
    fmt = "{\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + "}"
    return _format_width(fmt)


def _dict_repr(node: dict, depth: int) -> str:
    fmt = (f"{k}:{_format_node_repr(v,depth=depth+1)}" for k, v in node.items())
    fmt = (",\n" + "\t" * (depth + 1)).join(fmt)
    fmt = "{\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + "}"
    return _format_width(fmt)


def _dict_str(node: dict, depth: int) -> str:
    fmt = (
        f"{k}:{_format_node_str(v,depth=depth+1)}"
        if "\n" not in f"{v!s}"
        else f"{k}:"
        + "\n"
        + "\t" * (depth + 1)
        + f"{_format_width(_format_node_str(v,depth=depth+1))}"
        for k, v in node.items()
    )

    fmt = (",\n" + "\t" * (depth + 1)).join(fmt)
    fmt = "{\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + "}"
    return _format_width(fmt)


def _format_node_repr(node: Any, depth: int = 0, stats: bool = False) -> str:
    """pretty printer for a node

    Args:
        node: node to be printed
        depth: indentation depth. Defaults to 0.
        stats: print stats of arrays. Defaults to False.

    Returns:
        str: pretty printed node

    Examples:

        >>> print(_format_node_repr(dict(a=1, b=2, c=3, d= "x"*5), depth=0))
        {a:1,b:2,c:3,d:'xxxxx'}

        >>> print(_format_node_repr(dict(a=1, b=2, c=3, d= "x"*40), depth=0))
        {
            a:1,
            b:2,
            c:3,
            d:'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        }
    """

    if isinstance(node, (Callable)):
        if hasattr(node, "func"):
            return f"Partial({_func_repr(node.func)})"
        return _func_repr(node)

    elif hasattr(node, "shape") and hasattr(node, "dtype"):
        return _numpy_extended_repr(node) if stats else _numpy_repr(node)

    elif isinstance(node, list):
        return _list_repr(node, depth)

    elif isinstance(node, tuple):
        return _tuple_repr(node, depth)

    elif isinstance(node, set):
        return _set_repr(node, depth)

    elif isinstance(node, dict):
        return _dict_repr(node, depth)

    return ("\n" + "\t" * (depth)).join(f"{node!r}".split("\n"))


def _format_node_str(node, depth: int = 0):
    """
    Pretty printer for a node, differs from `_format_node_repr` in that
    it calls !s instead of !r

    Args:
        node (Any): node to be printed
        depth (int, optional): indentation depth. Defaults to 0.

    Returns:
        str: pretty printed node

    Examples:

        >>> print(_format_node_repr(dict(a=1, b=2, c=3, d= "x"*5), depth=0))
        {a:1,b:2,c:3,d:'xxxxx'}

        >>> print(_format_node_repr(dict(a=1, b=2, c=3, d= "x"*40), depth=0))
        {
            a:1,
            b:2,
            c:3,
            d:'xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
        }
    """

    if isinstance(node, Callable):
        if hasattr(node, "func"):
            return f"Partial({_func_repr(node.func)})"
        return _func_repr(node)

    elif isinstance(node, list):
        return _list_str(node, depth)

    elif isinstance(node, tuple):
        return _tuple_str(node, depth)

    elif isinstance(node, set):
        return _set_str(node, depth)

    elif isinstance(node, dict):
        return _dict_str(node, depth)

    return ("\n" + "\t" * (depth)).join(f"{node!s}".split("\n"))
