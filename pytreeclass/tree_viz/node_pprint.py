from __future__ import annotations

import dataclasses as dc
import functools as ft
import inspect
import math
from types import FunctionType
from typing import Any, Callable

import numpy as np
from jax._src.custom_derivatives import custom_jvp
from jaxlib.xla_extension import CompiledFunction

from pytreeclass.tree_viz.tree_viz_util import _format_width


def _numpy_pprint(node: np.ndarray, kind: str = "repr") -> str:
    """Replace np.ndarray repr with short hand notation for type and shape

    Example:
        >>> _numpy_pprint(np.ones((2,3)))
        'f64[2,3]∈[1.0,1.0]'
    """
    shape = (
        f"{node.shape}".replace(",", "")
        .replace("(", "[")
        .replace(")", "]")
        .replace(" ", ",")
    )
    if issubclass(node.dtype.type, np.integer):
        base = f"{node.dtype}".replace("int", "i") + shape
    elif issubclass(node.dtype.type, np.floating):
        base = f"{node.dtype}".replace("float", "f") + shape
    elif issubclass(node.dtype.type, np.complexfloating):
        base = f"{node.dtype}".replace("complex", "c") + shape
    else:
        base = f"{node.dtype}" + shape

    # Extended repr for numpy array, with extended information
    # this part of function is inspired by
    # lovely-jax https://github.com/xl0/lovely-jax

    if issubclass(node.dtype.type, np.number):
        # get min, max, mean, std of node
        low, high = np.min(node), np.max(node)
        # add brackets to indicate closed/open interval
        interval = "(" if math.isinf(low) else "["
        # if issubclass(node.dtype.type, np.integer):
        # if integer, round to nearest integer
        interval += (
            f"{low},{high}"
            if issubclass(node.dtype.type, np.integer)
            else f"{low:.1f},{high:.1f}"
        )

        mean, std = np.mean(node), np.std(node)
        # add brackets to indicate closed/open interval
        interval += ")" if math.isinf(high) else "]"
        # replace inf with infinity symbol
        interval = interval.replace("inf", "∞")
        # return extended repr
        return f"{base} ∈{interval} μ(σ)={mean:.1f}({std:.1f})"

    if kind == "repr":
        return base
    raise ValueError(f"kind must be 'repr' got {kind}")


def _func_pprint(func: Callable) -> str:
    """Pretty print function

    Example:
        >>> def example(a: int, b=1, *c, d, e=2, **f) -> str:
            ...
        >>> _func_pprint(example)
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


def _node_pprint(node: Any, depth: int = 0, kind: str = "str") -> str:
    if isinstance(node, (FunctionType, custom_jvp)):
        return _func_pprint(node)

    if isinstance(node, CompiledFunction):
        # special case for jitted functions
        return f"jit({_func_pprint(node.__wrapped__)})"

    if isinstance(node, ft.partial):
        # applies for partial functions including `jtu.Partial`
        return f"Partial({_func_pprint(node.func)})"

    if hasattr(node, "shape") and hasattr(node, "dtype") and kind == "repr":
        # works for numpy arrays, jax arrays
        return _numpy_pprint(node, kind)

    if hasattr(node, "_fields") and hasattr(node, "_asdict"):
        return f"namedtuple({_dict_pprint(node._asdict(), depth, kind=kind)})"

    if dc.is_dataclass(node):
        return f"dataclass({_dict_pprint(node.asdict(node), depth, kind=kind)})"

    if isinstance(node, list):
        return _list_pprint(node, depth, kind=kind)

    if isinstance(node, tuple):
        return _tuple_pprint(node, depth, kind=kind)

    if isinstance(node, set):
        return _set_pprint(node, depth, kind=kind)

    if isinstance(node, dict):
        return _dict_pprint(node, depth, kind=kind)

    if kind == "repr":
        fmt = f"{node!r}"
    elif kind in ["str"]:
        fmt = f"{node!s}"
    else:
        raise ValueError(f"kind must be 'repr', 'str' or 'extended_repr', got {kind}")

    return ("\n" + "\t" * (depth)).join(fmt.split("\n"))


_printer_map = {
    "repr": lambda node, depth: _node_pprint(node, depth, kind="repr"),
    "str": lambda node, depth: _node_pprint(node, depth, kind="str"),
}


def _list_pprint(node: list, depth: int, kind: str = "repr") -> str:
    """Pretty print a list"""
    printer = _printer_map[kind]
    fmt = (f"{_format_width(printer(v,depth=depth+1))}" for v in node)
    fmt = (",\n" + "\t" * (depth + 1)).join(fmt)
    fmt = "[\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + "]"
    return _format_width(fmt)


def _tuple_pprint(node: list, depth: int, kind: str = "repr") -> str:
    """Pretty print a list"""
    printer = _printer_map[kind]
    fmt = (f"{_format_width(printer(v,depth=depth+1))}" for v in node)
    fmt = (",\n" + "\t" * (depth + 1)).join(fmt)
    fmt = "(\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + ")"
    return _format_width(fmt)


def _set_pprint(node: list, depth: int, kind: str = "repr") -> str:
    """Pretty print a list"""
    printer = _printer_map[kind]
    fmt = (f"{_format_width(printer(v,depth=depth+1))}" for v in node)
    fmt = (",\n" + "\t" * (depth + 1)).join(fmt)
    fmt = "{\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + "}"
    return _format_width(fmt)


def _dict_pprint(node: dict, depth: int, kind: str = "repr") -> str:
    printer = _printer_map[kind]
    fmt = (
        f"{k}:{printer(v,depth=depth+1)}"
        if "\n" not in f"{v!s}"
        else f"{k}:"
        + "\n"
        + "\t" * (depth + 1)
        + f"{_format_width(printer(v,depth=depth+1))}"
        for k, v in node.items()
    )

    fmt = (",\n" + "\t" * (depth + 1)).join(fmt)
    fmt = "{\n" + "\t" * (depth + 1) + (fmt) + "\n" + "\t" * (depth) + "}"
    return _format_width(fmt)
