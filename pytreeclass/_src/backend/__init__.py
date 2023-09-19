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

from __future__ import annotations

import functools as ft
import os
from importlib.util import find_spec

backend = os.environ.get("PYTREECLASS_BACKEND", "default").lower()


@ft.lrucache(maxsize=1)
def is_available(backend):
    return find_spec(backend) is not None


if backend == "default":
    # do backend promotion if possible
    if is_available("jax"):
        backend = "jax"
    elif is_available("torch"):
        backend = "torch"
    elif is_available("numpy"):
        backend = "numpy"
    else:
        # no backend is available
        if not is_available("optree"):
            raise ImportError("No backend is available. Please install `optree`.")

        from pytreeclass._src.backend.arraylib.noarray import NoArray
        from pytreeclass._src.backend.treelib.optree import OpTreeTreeLib

        arraylib = NoArray()
        treelib = OpTreeTreeLib()


if backend == "jax":
    if not is_available("jax"):
        raise ImportError("`jax` backend requires `jax` to be installed.")

    from pytreeclass._src.backend.arraylib.jax import JaxArray
    from pytreeclass._src.backend.treelib.jax import JaxTreeLib

    arraylib = JaxArray()
    treelib = JaxTreeLib()

elif backend == "numpy":
    if not is_available("optree"):
        raise ImportError("`numpy` backend requires `optree` to be installed.")

    if not is_available("numpy"):
        raise ImportError("`numpy` backend requires `numpy` to be installed.")

    from pytreeclass._src.backend.arraylib.numpy import NumpyArray
    from pytreeclass._src.backend.treelib.optree import OpTreeTreeLib

    arraylib = NumpyArray()
    treelib = OpTreeTreeLib()

elif backend == "torch":
    if not is_available("torch"):
        raise ImportError("`torch` backend requires `torch` to be installed.")
    if not is_available("optree"):
        raise ImportError("`torch` backend requires `optree` to be installed.")

    from pytreeclass._src.backend.arraylib.torch import TorchArray
    from pytreeclass._src.backend.treelib.optree import OpTreeTreeLib

    arraylib = TorchArray()
    treelib = OpTreeTreeLib()