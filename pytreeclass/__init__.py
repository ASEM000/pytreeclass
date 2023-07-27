# Copyright 2023 PyTreeClass authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from pytreeclass._src.code_build import autoinit, field, fields
from pytreeclass._src.tree_base import TreeClass
from pytreeclass._src.tree_index import AtIndexer, BaseKey
from pytreeclass._src.tree_mask import (
    freeze,
    is_frozen,
    is_nondiff,
    tree_mask,
    tree_unmask,
    unfreeze,
)
from pytreeclass._src.tree_pprint import (
    tree_diagram,
    tree_graph,
    tree_mermaid,
    tree_repr,
    tree_repr_with_trace,
    tree_str,
    tree_summary,
)
from pytreeclass._src.tree_util import (
    Partial,
    bcmap,
    is_tree_equal,
    leafwise,
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
    "autoinit",
    # pprint utils
    "tree_diagram",
    "tree_graph",
    "tree_mermaid",
    "tree_repr",
    "tree_str",
    "tree_summary",
    # masking utils
    "is_nondiff",
    "is_frozen",
    "freeze",
    "unfreeze",
    "tree_unmask",
    "tree_mask",
    # indexing utils
    "AtIndexer",
    "BaseKey",
    # tree utils
    "bcmap",
    "tree_map_with_trace",
    "tree_leaves_with_trace",
    "tree_flatten_with_trace",
    "tree_repr_with_trace",
    "Partial",
    "leafwise",
)

__version__ = "0.5.0post0"

AtIndexer.__module__ = "pytreeclass"
TreeClass.__module__ = "pytreeclass"
Partial.__module__ = "pytreeclass"
