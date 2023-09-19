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

import os
from importlib.util import find_spec

backend = os.environ.get("PYTREECLASS_BACKEND", "jax").lower()


if backend == "jax":
    if find_spec("jax") is None:
        import logging

        logging.info("[PYTREECLASS]: switching to `numpy` backend.")
        backend = "default"
    else:
        from pytreeclass._src.backend.arraylib.jax import JaxArray
        from pytreeclass._src.backend.treelib.jax import JaxTreeLib

        arraylib = JaxArray()
        treelib = JaxTreeLib()


if backend == "numpy":
    if find_spec("numpy") is None:
        raise ImportError("`numpy` backend requires `numpy` to be installed.")
    if find_spec("optree") is None:
        raise ImportError("`numpy` backend requires `optree` to be installed.")

    from pytreeclass._src.backend.arraylib.numpy import NumpyArray
    from pytreeclass._src.backend.treelib.optree import OpTreeTreeLib

    arraylib = NumpyArray()
    treelib = OpTreeTreeLib()

elif backend == "torch":
    if find_spec("torch") is None:
        raise ImportError("`torch` backend requires `torch` to be installed.")
    if find_spec("optree") is None:
        raise ImportError("`torch` backend requires `optree` to be installed.")

    from pytreeclass._src.backend.arraylib.torch import TorchArray
    from pytreeclass._src.backend.treelib.optree import OpTreeTreeLib

    arraylib = TorchArray()
    treelib = OpTreeTreeLib()

elif backend == "default":
    if find_spec("optree") is None:
        raise ImportError("`default` backend requires `optree` to be installed.")

    from pytreeclass._src.backend.arraylib.noarray import NoArray
    from pytreeclass._src.backend.treelib.optree import OpTreeTreeLib

    arraylib = NoArray()
    treelib = OpTreeTreeLib()
