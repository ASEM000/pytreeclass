from pytreeclass._src.code_build import field, fields
from pytreeclass._src.tree_base import AtIndexer, TreeClass
from pytreeclass._src.tree_mask import (
    freeze,
    is_frozen,
    is_nondiff,
    tree_mask,
    tree_unmask,
    unfreeze,
)
from pytreeclass._src.tree_pprint import (
    pp_dispatcher,
    tree_diagram,
    tree_indent,
    tree_mermaid,
    tree_repr,
    tree_repr_with_trace,
    tree_str,
    tree_summary,
)
from pytreeclass._src.tree_util import (
    BaseKey,
    Partial,
    bcmap,
    is_tree_equal,
    tree_flatten_with_trace,
    tree_leaves_with_trace,
    tree_map_with_trace,
)

__all__ = (
    # general utils
    "TreeClass",
    "is_tree_equal",
    "field",
    "fields",
    # pprint utils
    "tree_diagram",
    "tree_mermaid",
    "tree_repr",
    "tree_str",
    "tree_indent",
    "tree_summary",
    "pp_dispatcher",
    # freeze/unfreeze utils
    "is_nondiff",
    "is_frozen",
    "freeze",
    "unfreeze",
    "tree_mask",
    "tree_unmask",
    # masking and indexing utils
    "bcmap",
    "AtIndexer",
    "tree_map_with_trace",
    "tree_leaves_with_trace",
    "tree_flatten_with_trace",
    "tree_repr_with_trace",
    "Partial",
    "BaseKey",
)

__version__ = "0.4.0"

AtIndexer.__module__ = "pytreeclass"
TreeClass.__module__ = "pytreeclass"
Partial.__module__ = "pytreeclass"
