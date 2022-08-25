from dataclasses import field

from pytreeclass.src.tree_viz_util import _format_node_repr, _format_node_str


class ImmutableInstanceError(Exception):
    pass


class static_value:
    def __init__(self, value):
        self.value = value

    def __repr__(self):
        return f"*{_format_node_repr(self.value,0)}"

    def __str__(self):
        return f"*{_format_node_str(self.value,0)}"


class mutableContext:
    """Allow mutable behvior within this context"""

    def __init__(self, instance):
        assert hasattr(
            instance, "__treeclass_structure__"
        ), "instance must be a treeclass"
        self.instance = instance

    def __enter__(self):
        object.__setattr__(self.instance, "__immutable_treeclass__", False)

    def __exit__(self, type_, value, traceback):
        object.__setattr__(self.instance, "__immutable_treeclass__", True)


def static_field(**kwargs):
    """ignore from pytree computations"""
    return field(**{**kwargs, **{"metadata": {"static": True}}})
