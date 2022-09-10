from __future__ import annotations

import inspect
from types import FunctionType

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
    """Replace jnp.ndarray repr with short hand notation for type and shape"""
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


def _func_repr(func):
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


def _format_node_repr(node, depth):
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

    else:
        return ("\n" + "\t" * (depth)).join(f"{node!r}".split("\n"))


def _format_node_str(node, depth):
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

    else:
        return ("\n" + "\t" * (depth)).join(f"{node!s}".split("\n"))


def _format_node_diagram(node, *args, **kwargs):
    if isinstance(node, (CompiledFunction, custom_jvp, FunctionType)):
        return _func_repr(node)

    elif isinstance(node, (jnp.ndarray, jax.ShapeDtypeStruct)):
        return _jax_numpy_repr(node)

    else:
        return f"{node!r}"
