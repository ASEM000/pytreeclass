from __future__ import annotations

import inspect
from dataclasses import Field
from types import FunctionType
from typing import Any, Callable

import jax
import jax.numpy as jnp
from jax._src.custom_derivatives import custom_jvp
from jaxlib.xla_extension import CompiledFunction


def _format_width(string, width=50):
    """strip newline/tab characters if less than max width"""
    children_length = len(string) - string.count("\n") - string.count("\t")
    return (
        string
        if children_length > width
        else string.replace("\n", "").replace("\t", "")
    )


def _jax_numpy_repr(node: jnp.ndarray) -> str:
    """Replace jnp.ndarray repr with short hand notation for type and shape

    Args:
        node (jnp.ndarray): jax numpy array

    Returns:
        str: short hand notation for type and shape

    Example:
        >>> _jax_numpy_repr(jnp.ones((2,3)))
        'f32[2,3]'
    """
    replace_tuple = (
        ("int", "i"),
        ("float", "f"),
        ("complex", "c"),
        (",)", ")"),
        ("(", "["),
        (")", "]"),
        (" ", ""),
    )

    formatted_string = f"{node.dtype}{jnp.shape(node)!r}"

    for lhs, rhs in replace_tuple:
        formatted_string = formatted_string.replace(lhs, rhs)
    return formatted_string


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
    return (
        f"{name}("
        + ",".join(item for item in [args, varargs, kwonlyargs, varkw] if item != "")
        + ")"
    )


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

    elif isinstance(node, (jnp.ndarray, jax.ShapeDtypeStruct)):
        return _jax_numpy_repr(node)

    elif isinstance(node, list):
        # increase depth for each item in list
        # moreover, '_format_width' is done on each item repr
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{_format_width(_format_node_repr(v,depth=depth+1))}" for v in node
        )
        return _format_width(
            "[\n" + "\t" * (depth + 1) + (string) + "\n" + "\t" * (depth) + "]"
        )

    elif isinstance(node, tuple):
        # increase depth by 1 for each item in the tuple
        # moreover, `_format_width` is done on each item repr
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{_format_width(_format_node_repr(v,depth=depth+1))}" for v in node
        )
        return _format_width(
            "(\n" + "\t" * (depth + 1) + (string) + "\n" + "\t" * (depth) + ")"
        )

    elif isinstance(node, set):
        # increase depth by 1 for each item in the set
        # moreover, `_format_width` is done on each item repr
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{_format_width(_format_node_repr(v,depth=depth+1))}" for v in node
        )
        return _format_width(
            "{\n" + "\t" * (depth + 1) + (string) + "\n" + "\t" * (depth) + "}"
        )

    elif isinstance(node, dict):
        # increase depth by 1 for each item in the dict
        # moreover, `_format_width` is done on each item repr
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{k}:{_format_node_repr(v,depth=depth+1)}"
            if "\n" not in f"{v!s}"
            else f"{k}:"
            + "\n"
            + "\t" * (depth + 1)
            + f"{_format_width(_format_node_repr(v,depth=depth+1))}"
            for k, v in node.items()
        )
        return _format_width(
            "{\n" + "\t" * (depth + 1) + (string) + "\n" + "\t" * (depth) + "}"
        )

    elif isinstance(node, Field):
        attrs = [
            f"name={node.name}",
            f"type={node.type}",
            f"default={node.default}",
            f"default_factory={node.default_factory}",
            f"init={node.init}",
            f"repr={node.repr}",
            f"hash={node.hash}",
            f"compare={node.compare}",
            f"metadata={node.metadata}",
        ]
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{_format_width(_format_node_repr(v,depth=depth+1))}" for v in attrs
        )
        return _format_width(
            "Field(\n" + "\t" * (depth + 1) + (string) + "\n" + "\t" * (depth) + ")"
        )

    else:
        return ("\n" + "\t" * (depth)).join(f"{node!r}".split("\n"))


def _format_node_str(node, depth):
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

    elif isinstance(node, list):
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{_format_width(_format_node_str(v,depth=depth+1))}" for v in node
        )
        return _format_width(
            "[\n" + "\t" * (depth + 1) + (string) + "\n" + "\t" * (depth) + "]"
        )

    elif isinstance(node, tuple):
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{_format_width(_format_node_str(v,depth=depth+1))}" for v in node
        )
        return _format_width(
            "(\n" + "\t" * (depth + 1) + (string) + "\n" + "\t" * (depth) + ")"
        )

    elif isinstance(node, set):
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{_format_width(_format_node_str(v,depth=depth+1))}" for v in node
        )
        return _format_width(
            "{\n" + "\t" * (depth + 1) + (string) + "\n" + "\t" * (depth) + "}"
        )

    elif isinstance(node, dict):
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{k}:{_format_node_str(v,depth=depth+1)}"
            if "\n" not in f"{v!s}"
            else f"{k}:"
            + "\n"
            + "\t" * (depth + 1)
            + f"{_format_width(_format_node_str(v,depth=depth+1))}"
            for k, v in node.items()
        )
        return _format_width(
            "{\n" + "\t" * (depth + 1) + (string) + "\n" + "\t" * (depth) + "}"
        )

    elif isinstance(node, Field):
        attrs = [
            f"name={node.name}",
            f"type={node.type}",
            f"default={node.default}",
            f"default_factory={node.default_factory}",
            f"init={node.init}",
            f"repr={node.repr}",
            f"hash={node.hash}",
            f"compare={node.compare}",
            f"metadata={node.metadata}",
        ]
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{_format_width(_format_node_repr(v,depth=depth+1))}" for v in attrs
        )
        return _format_width(
            "Field(\n" + "\t" * (depth + 1) + (string) + "\n" + "\t" * (depth) + ")"
        )

    else:
        return ("\n" + "\t" * (depth)).join(f"{node!s}".split("\n"))


def _format_node_diagram(node, *args, **kwargs):
    if isinstance(node, (CompiledFunction, custom_jvp, FunctionType)):
        return _func_repr(node)

    elif isinstance(node, (jnp.ndarray, jax.ShapeDtypeStruct)):
        return _jax_numpy_repr(node)

    else:
        return f"{node!r}"
