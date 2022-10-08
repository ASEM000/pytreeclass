import functools as ft

from pytreeclass import tree_util, tree_viz
from pytreeclass._src.tree_util import (
    filter_nondiff,
    tree_freeze,
    tree_unfreeze,
    unfilter_nondiff,
)
from pytreeclass.treeclass import (
    field,
    fields,
    is_frozen_field,
    is_nondiff_field,
    is_treeclass,
    is_treeclass_equal,
    is_treeclass_frozen,
    is_treeclass_leaf,
    is_treeclass_leaf_bool,
    is_treeclass_non_leaf,
    is_treeclass_nondiff,
    treeclass,
)

nondiff_field = ft.partial(field, nondiff=True)
frozen_field = ft.partial(field, frozen=True)
static_field = nondiff_field


__all__ = (
    "treeclass",
    "tree_viz",
    "tree_util",
    "tree_freeze",
    "tree_unfreeze",
    "filter_nondiff",
    "unfilter_nondiff",
    "field",
    "is_frozen_field",
    "is_nondiff_field",
    "fields",
    "is_treeclass",
    "is_treeclass_equal",
    "is_treeclass_frozen",
    "is_treeclass_leaf_bool",
    "is_treeclass_leaf",
    "is_treeclass_non_leaf",
    "is_treeclass_nondiff",
    "nondiff_field",
    "frozen_field",
    "static_field",
)

__version__ = "0.1.10"
