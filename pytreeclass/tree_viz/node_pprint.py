from __future__ import annotations

import inspect
from types import FunctionType

import jax
import jax.numpy as jnp
import jaxlib

from pytreeclass.src.dispatch import dispatch


def _format_width(string, width=50):
    """strip newline/tab characters if less than max width"""
    stripped_string = string.replace("\n", "").replace("\t", "")
    children_length = len(stripped_string)
    return string if children_length > width else stripped_string


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
    @dispatch(argnum=0)
    def __format_node_repr(node, depth):
        return ("\n" + "\t" * (depth)).join(f"{node!r}".split("\n"))

    @__format_node_repr.register(jaxlib.xla_extension.CompiledFunction)
    @__format_node_repr.register(jax._src.custom_derivatives.custom_jvp)
    @__format_node_repr.register(FunctionType)
    def _(node, *args, **kwargs):
        return _func_repr(node)

    @__format_node_repr.register(jnp.ndarray)
    @__format_node_repr.register(jax.ShapeDtypeStruct)
    def _(node, *args, **kwargs):
        return _jax_numpy_repr(node)

    @__format_node_repr.register(list)
    def _(node, depth):
        # increase depth for each item in list
        # moreover, '_format_width' is done on each item repr
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{_format_width(_format_node_repr(v,depth=depth+1))}" for v in node
        )
        return "[\n" + "\t" * (depth + 1) + string + "\n" + "\t" * (depth) + "]"

    @__format_node_repr.register(tuple)
    def _(node, depth):
        # increase depth by 1 for each item in the tuple
        # moreover, `_format_width` is done on each item repr
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{_format_width(_format_node_repr(v,depth=depth+1))}" for v in node
        )
        return "(\n" + "\t" * (depth + 1) + string + "\n" + "\t" * (depth) + ")"

    @__format_node_repr.register(set)
    def _(node, depth):
        # increase depth by 1 for each item in the set
        # moreover, `_format_width` is done on each item repr
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{_format_width(_format_node_repr(v,depth=depth+1))}" for v in node
        )
        return "{\n" + "\t" * (depth + 1) + string + "\n" + "\t" * (depth) + "}"

    @__format_node_repr.register(dict)
    def _(node, depth):
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
        return "{\n" + "\t" * (depth + 1) + string + "\n" + "\t" * (depth) + "}"

    return __format_node_repr(node, depth)


def _format_node_str(node, depth):
    @dispatch(argnum=0)
    def __format_node_str(node, depth):
        return ("\n" + "\t" * (depth)).join(f"{node!s}".split("\n"))

    @__format_node_str.register(jaxlib.xla_extension.CompiledFunction)
    @__format_node_str.register(jax._src.custom_derivatives.custom_jvp)
    @__format_node_str.register(FunctionType)
    def _(node, *args, **kwargs):
        return _func_repr(node)

    @__format_node_str.register(list)
    def _(node, depth):
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{_format_width(_format_node_str(v,depth=depth+1))}" for v in node
        )
        return "[\n" + "\t" * (depth + 1) + string + "\n" + "\t" * (depth) + "]"

    @__format_node_str.register(tuple)
    def _(node, depth):
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{_format_width(_format_node_str(v,depth=depth+1))}" for v in node
        )
        return "(\n" + "\t" * (depth + 1) + string + "\n" + "\t" * (depth) + ")"

    @__format_node_str.register(set)
    def _(node, depth):
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{_format_width(_format_node_str(v,depth=depth+1))}" for v in node
        )
        return "{\n" + "\t" * (depth + 1) + string + "\n" + "\t" * (depth) + "}"

    @__format_node_str.register(dict)
    def _(node, depth):
        string = (",\n" + "\t" * (depth + 1)).join(
            f"{k}:{_format_node_str(v,depth=depth+1)}"
            if "\n" not in f"{v!s}"
            else f"{k}:"
            + "\n"
            + "\t" * (depth + 1)
            + f"{_format_width(_format_node_str(v,depth=depth+1))}"
            for k, v in node.items()
        )
        return "{\n" + "\t" * (depth + 1) + string + "\n" + "\t" * (depth) + "}"

    return _format_width(__format_node_str(node, depth))


def _format_node_diagram(node, *args, **kwargs):
    @dispatch(argnum=0)
    def __format_node_diagram(node, *args, **kwargs):
        return f"{node!r}"

    @__format_node_diagram.register(jaxlib.xla_extension.CompiledFunction)
    @__format_node_diagram.register(jax._src.custom_derivatives.custom_jvp)
    @__format_node_diagram.register(FunctionType)
    def _(node, *args, **kwargs):
        return _func_repr(node)

    @__format_node_diagram.register(jnp.ndarray)
    @__format_node_diagram.register(jax.ShapeDtypeStruct)
    def _(node, *args, **kwargs):
        return _jax_numpy_repr(node)

    return __format_node_diagram(node, *args, **kwargs)
