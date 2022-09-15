from pytreeclass import tree_util, tree_viz
from pytreeclass._src.tree_util import nondiff_field
from pytreeclass.treeclass import treeclass

static_field = nondiff_field

__all__ = (
    "treeclass",
    "static_field",
    "tree_viz",
    "tree_util",
    "nondiff_field",
    "static_field",
)

__version__ = "0.1.6"
