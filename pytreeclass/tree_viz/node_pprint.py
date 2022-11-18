from __future__ import annotations

import inspect
import math
from dataclasses import MISSING, Field
from types import FunctionType
from typing import Any, Callable

import jax
import jax.numpy as jnp
import jax.tree_util as jtu
import numpy as np
from jax._src.custom_derivatives import custom_jvp
from jaxlib.xla_extension import CompiledFunction


def _format_width(string, width=60):
    """strip newline/tab characters if less than max width"""
    children_length = len(string) - string.count("\n") - string.count("\t")
    if children_length > width:
        return string
    return string.replace("\n", "").replace("\t", "")


def _jax_numpy_repr(node: jnp.ndarray | np.ndarray) -> str:
    """Replace jnp.ndarray repr with short hand notation for type and shape

    Args:
        node (jnp.ndarray): jax numpy array

    Returns:
        str: short hand notation for type and shape

    Example:
        >>> _jax_numpy_repr(jnp.ones((2,3)))
        'f32[2,3]'
    """
    shape = (
        f"{node.shape}".replace(",", "")
        .replace("(", "[")
        .replace(")", "]")
        .replace(" ", ",")
    )

    if issubclass(node.dtype.type, jnp.integer):
        dtype = f"{node.dtype}".replace("int", "i")
    elif issubclass(node.dtype.type, jnp.floating):
        dtype = f"{node.dtype}".replace("float", "f")
    elif issubclass(node.dtype.type, jnp.complexfloating):
        dtype = f"{node.dtype}".replace("complex", "c")
    else:
        dtype = f"{node.dtype}"

    return f"{dtype}{shape}"


def _jax_numpy_extended_repr(node: jnp.ndarray | np.ndarray) -> str:
    """Replace jnp.ndarray repr with short hand notation for type and shape
    Adds information about min, max, mean, and std

    Args:
        node (jnp.ndarray): jax numpy array

    Returns:
        str: short hand notation for type and shape

    Example:
        >>> _jax_numpy_extended_repr(jnp.ones((2,3)))
        ''f32[2,3]∈[1.00,1.00]<μ=1.00,σ=0.00>''
    """
    shape = (
        f"{node.shape}".replace(",", "")
        .replace("(", "[")
        .replace(")", "]")
        .replace(" ", ",")
    )

    if issubclass(node.dtype.type, jnp.number):
        low, high = jnp.min(node), jnp.max(node)
        interval = "(" if math.isinf(low) else "["
        std, mean = jnp.std(node), jnp.mean(node)

        if issubclass(node.dtype.type, jnp.integer):
            dtype = f"{node.dtype}".replace("int", "i")
            interval += f"{low},{high}"
        elif issubclass(node.dtype.type, jnp.floating):
            dtype = f"{node.dtype}".replace("float", "f")
            interval += f"{low:.2f},{high:.2f}"
        elif issubclass(node.dtype.type, jnp.complexfloating):
            dtype = f"{node.dtype}".replace("complex", "c")
            interval += f"{low:.2f},{high:.2f}"

        interval += ")" if math.isinf(high) else "]"
        interval = interval.replace("inf", "∞")
        return f"{dtype}{shape}∈{interval}<μ={mean:.2f},σ={std:.2f}>"

    dtype = f"{node.dtype}"
    return f"{dtype}{shape}"


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
    string = (",\n" + "\t" * (depth + 1)).join(
        f"{_format_width(_format_node_repr(v,depth=depth+1))}" for v in node
    )

    fmt = "[\n" + "\t" * (depth + 1) + (string) + "\n" + "\t" * (depth) + "]"
    return _format_width(fmt)


def _list_str(node: list, depth: int) -> str:
    string = (",\n" + "\t" * (depth + 1)).join(
        f"{_format_width(_format_node_str(v,depth=depth+1))}" for v in node
    )

    fmt = "[\n" + "\t" * (depth + 1) + (string) + "\n" + "\t" * (depth) + "]"
    return _format_width(fmt)


def _tuple_repr(node: tuple, depth: int) -> str:
    string = (",\n" + "\t" * (depth + 1)).join(
        f"{_format_width(_format_node_repr(v,depth=depth+1))}" for v in node
    )
    fmt = "(\n" + "\t" * (depth + 1) + (string) + "\n" + "\t" * (depth) + ")"
    return _format_width(fmt)


def _tuple_str(node: tuple, depth: int) -> str:
    string = (",\n" + "\t" * (depth + 1)).join(
        f"{_format_width(_format_node_str(v,depth=depth+1))}" for v in node
    )

    fmt = "(\n" + "\t" * (depth + 1) + (string) + "\n" + "\t" * (depth) + ")"
    return _format_width(fmt)


def _set_repr(node: set, depth: int) -> str:
    string = (",\n" + "\t" * (depth + 1)).join(
        f"{_format_width(_format_node_repr(v,depth=depth+1))}" for v in node
    )
    fmt = "{\n" + "\t" * (depth + 1) + (string) + "\n" + "\t" * (depth) + "}"
    return _format_width(fmt)


def _set_str(node: set, depth: int) -> str:
    string = (",\n" + "\t" * (depth + 1)).join(
        f"{_format_width(_format_node_str(v,depth=depth+1))}" for v in node
    )
    fmt = "{\n" + "\t" * (depth + 1) + (string) + "\n" + "\t" * (depth) + "}"
    return _format_width(fmt)


def _dict_repr(node: dict, depth: int) -> str:
    string = (",\n" + "\t" * (depth + 1)).join(
        f"{k}:{_format_node_repr(v,depth=depth+1)}" for k, v in node.items()
    )
    fmt = "{\n" + "\t" * (depth + 1) + (string) + "\n" + "\t" * (depth) + "}"
    return _format_width(fmt)


def _dict_str(node: dict, depth: int) -> str:
    string = (",\n" + "\t" * (depth + 1)).join(
        f"{k}:{_format_node_str(v,depth=depth+1)}"
        if "\n" not in f"{v!s}"
        else f"{k}:"
        + "\n"
        + "\t" * (depth + 1)
        + f"{_format_width(_format_node_str(v,depth=depth+1))}"
        for k, v in node.items()
    )
    fmt = "{\n" + "\t" * (depth + 1) + (string) + "\n" + "\t" * (depth) + "}"
    return _format_width(fmt)


def _field_repr(node: Field, depth: int) -> str:
    attrs = [f"name={node.name}"]
    attrs += [f"type={node.type}"]
    attrs += [f"default={node.default}"] if node.default is not MISSING else []
    attrs += [f"default_factory={node.default_factory}"] if node.default_factory is not MISSING else []  # fmt: skip
    attrs += [f"init={node.init}"]
    attrs += [f"repr={node.repr}"]
    attrs += [f"hash={node.hash}"] if node.hash is not None else []
    attrs += [f"compare={node.compare}"]
    attrs += [f"metadata={node.metadata}"] if node.metadata != {} else []
    string = (',').join(f"{_format_width(_format_node_repr(v,depth=depth+1))}" for v in attrs)  # fmt: skip
    return _format_width(f"Field({string})")


def _format_node_repr(node: Any, depth: int = 0) -> str:
    """pretty printer for a node

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

    if isinstance(node, (CompiledFunction, custom_jvp, FunctionType)):
        return _func_repr(node)

    elif isinstance(node, jtu.Partial):
        return f"Partial({_func_repr(node.func)})"

    elif isinstance(node, (np.ndarray, jnp.ndarray, jax.ShapeDtypeStruct)):
        return _jax_numpy_repr(node)

    elif isinstance(node, list):
        return _list_repr(node, depth)

    elif isinstance(node, tuple):
        return _tuple_repr(node, depth)

    elif isinstance(node, set):
        return _set_repr(node, depth)

    elif isinstance(node, dict):
        return _dict_repr(node, depth)

    elif isinstance(node, Field):
        return _field_repr(node, depth)

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

    if isinstance(node, (CompiledFunction, custom_jvp, FunctionType)):
        return _func_repr(node)

    elif isinstance(node, jtu.Partial):
        return f"Partial({_func_repr(node.func)})"

    elif isinstance(node, list):
        return _list_str(node, depth)

    elif isinstance(node, tuple):
        return _tuple_str(node, depth)

    elif isinstance(node, set):
        return _set_str(node, depth)

    elif isinstance(node, dict):
        return _dict_str(node, depth)

    elif isinstance(node, Field):
        return _field_repr(node, depth)

    return ("\n" + "\t" * (depth)).join(f"{node!s}".split("\n"))


def _format_node_extended_repr(node: Any, depth: int = 0) -> str:
    if isinstance(node, (jnp.ndarray, np.ndarray)):
        return _jax_numpy_extended_repr(node)
    return _format_node_repr(node, depth)


def _format_node_diagram(node, *a, **k):
    if isinstance(node, (CompiledFunction, custom_jvp, FunctionType)):
        return _func_repr(node)

    elif isinstance(node, jtu.Partial):
        return f"Partial({_func_repr(node.func)})"

    elif isinstance(node, (jnp.ndarray, jax.ShapeDtypeStruct)):
        return _jax_numpy_repr(node)

    return f"{node!r}"
