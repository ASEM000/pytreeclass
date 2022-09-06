from __future__ import annotations

from typing import Any

import pytreeclass as pytc
from pytreeclass.src.dispatch import dispatch


@pytc.treeclass
class Container:
    """A container for list/tuple/set/dict that flatten its items
    and register each item as a dataclass field

    Example:
        >>> a = Container([1,2,3])
        >>> a
        Container([0]=1,[1]=2,[2]=3) # each item is registered as a dataclass field

        >>> print(a.tree_diagram())
        Container
            ├── [0]=1
            ├── [1]=2
            └── [2]=3
    """

    keys: tuple[str] = pytc.static_field(repr=False)
    container_type: type = pytc.static_field(repr=False)

    # the motivation for this class definition is to expose
    # the items of a list/tuple/dict/set as dataclass fields
    # to enable field specific operations on the items.

    @dispatch(argnum=1)
    def __init__(self, values: list[Any] | tuple[Any] | dict[str, Any] | set[Any]):
        # raise an error if the input is not a list/tuple/dict/set
        raise NotImplementedError("values must be a list, tuple, set or dict")

    @__init__.register(list)
    def _(self, values: list[Any]):
        self.container_type = list
        self.keys = tuple(f"[{i}]" for i, _ in enumerate(values))
        for i, (key, value) in enumerate(zip(self.keys, values)):
            self.param(value, name=key)

    @__init__.register(tuple)
    def _(self, values: tuple[Any]):
        self.container_type = tuple
        self.keys = tuple(f"({i})" for i, _ in enumerate(values))
        for i, (key, value) in enumerate(zip(self.keys, values)):
            self.param(value, name=key)

    @__init__.register(set)
    def _(self, values: set[Any]):
        self.container_type = set
        self.keys = tuple("{" + f"{i}" + "}" for i, _ in enumerate(values))
        for i, (key, value) in enumerate(zip(self.keys, values)):
            self.param(value, name=key)

    @__init__.register(dict)
    def _(self, values: dict[str, Any]):
        self.container_type = dict
        for key, value in values.items():
            self.param(value, name=key)
        self.keys = tuple(values.keys())

    @dispatch(argnum=1)
    def __getitem__(self, key):
        raise NotImplementedError("key must be an int or a str")

    @__getitem__.register(str)
    def _(self, key: str):
        # return the item with the given key
        return getattr(self, key)

    @__getitem__.register(int)
    def _(self, index: int):
        # return the item with the given index
        return getattr(self, self.keys[index])

    def items(self):
        """return the items of the container in its input form"""
        if self.container_type is dict:
            return {key: getattr(self, key) for key in self.keys}
        elif self.container_type is list:
            return [getattr(self, key) for key in self.keys]
        elif self.container_type is tuple:
            return tuple([getattr(self, key) for key in self.keys])
        elif self.container_type is set:
            return set([getattr(self, key) for key in self.keys])
