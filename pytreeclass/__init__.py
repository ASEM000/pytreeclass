from pytreeclass._src.tree_base import is_tree_equal, treeclass
from pytreeclass._src.tree_decorator import field, fields, is_treeclass
from pytreeclass._src.tree_freeze import (
    FrozenWrapper,
    ImmutableWrapper,
    freeze,
    is_frozen,
    is_nondiff,
    unfreeze,
)
from pytreeclass._src.tree_indexer import bcmap, tree_indexer
from pytreeclass._src.tree_pprint import (
    tree_diagram,
    tree_mermaid,
    tree_repr,
    tree_str,
    tree_summary,
)
from pytreeclass._src.tree_trace import (
    register_pytree_node_trace,
    tree_flatten_with_trace,
    tree_leaves_with_trace,
    tree_map_with_trace,
)

__all__ = (
    # general utils
    "treeclass",
    "is_treeclass",
    "field",
    "fields",
    "is_tree_equal",
    # pprint utils
    "tree_diagram",
    "tree_mermaid",
    "tree_repr",
    "tree_str",
    "tree_summary",
    # freeze/unfreeze utils
    "is_nondiff",
    "is_frozen",
    "freeze",
    "unfreeze",
    "FrozenWrapper",
    "ImmutableWrapper",
    # masking and indexing utils
    "bcmap",
    "tree_indexer",
    "register_pytree_node_trace",
    "tree_map_with_trace",
    "tree_leaves_with_trace",
    "tree_flatten_with_trace",
)

__version__ = "0.2.0b12"
