# Copyright 2023 pytreeclass authors
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

"""Backend tools for pytreeclass."""

from __future__ import annotations

import dataclasses as dc
import importlib

# TODO: add optree support
BACKENDS = ["jax"]

if importlib.util.find_spec("jax"):
    # jax backend
    import jax

    @dc.dataclass(frozen=True)
    class NamedSequenceKey(jax.tree_util.GetAttrKey, jax.tree_util.SequenceKey):
        def __str__(self):
            return f".{self.name}"

    class TreeUtil:
        # tree utils
        tree_map = jax.tree_util.tree_map
        tree_leaves = jax.tree_util.tree_leaves
        tree_flatten = jax.tree_util.tree_flatten
        tree_unflatten = jax.tree_util.tree_unflatten
        tree_structure = jax.tree_util.tree_structure
        tree_reduce = jax.tree_util.tree_reduce
        tree_map_with_path = jax.tree_util.tree_map_with_path
        tree_flatten_with_path = jax.tree_util.tree_flatten_with_path
        # registeration utils
        register_pytree_node = jax.tree_util.register_pytree_node
        register_pytree_with_keys = jax.tree_util.register_pytree_with_keys
        # path keys
        GetAttrKey = jax.tree_util.GetAttrKey
        SequenceKey = jax.tree_util.SequenceKey
        DictKey = jax.tree_util.DictKey
        NamedSequenceKey = NamedSequenceKey
        keystr = jax.tree_util.keystr

    class Backend:
        # array utils
        ndarray = jax.Array
        numpy = jax.numpy
        # tree utils
        tree_util = TreeUtil

else:
    raise ImportError(f"None of the {BACKENDS=} are installed.")
