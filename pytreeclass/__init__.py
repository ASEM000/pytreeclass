from pytreeclass._src.tree_decorator import TreeClass, field
from pytreeclass._src.tree_freeze import freeze, is_frozen, is_nondiff, unfreeze
from pytreeclass._src.tree_indexer import bcmap, is_tree_equal, tree_indexer
from pytreeclass._src.tree_pprint import (
    tree_diagram,
    tree_indent,
    tree_mermaid,
    tree_repr,
    tree_repr_with_trace,
    tree_str,
    tree_summary,
)
from pytreeclass._src.tree_trace import (
    tree_flatten_with_trace,
    tree_leaves_with_trace,
    tree_map_with_trace,
)

__all__ = (
    # general utils
    "TreeClass",
    "is_tree_equal",
    "field",
    # pprint utils
    "tree_diagram",
    "tree_mermaid",
    "tree_repr",
    "tree_str",
    "tree_indent",
    "tree_summary",
    "tree_trace_summary",
    # freeze/unfreeze utils
    "is_nondiff",
    "is_frozen",
    "freeze",
    "unfreeze",
    # masking and indexing utils
    "bcmap",
    "tree_indexer",
    "tree_map_with_trace",
    "tree_leaves_with_trace",
    "tree_flatten_with_trace",
    "tree_repr_with_trace",
)

__version__ = "0.3.1"
