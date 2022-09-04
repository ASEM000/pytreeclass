from __future__ import annotations

from typing import Any

import pytreeclass as pytc
from pytreeclass.src.decorator_util import dispatch


@pytc.treeclass
class Container:
    """A container for list/tuple/set/dict that flatten its items
    and register each item as a dataclass field

    Example:
        >>> a = Container([1,2,3])
        >>> a
        Container(node_0=1,node_1=2,node_2=3) # each item is registered as a dataclass field

        >>> print(a.tree_diagram())
        Container
            ├── node_0=1
            ├── node_1=2
            └── node_2=3
    """

    keys: tuple[str] = pytc.static_field(repr=False)
    # container_type: type = pytc.static_field(repr=False)

    # the motivation for this class definition is to expose
    # the items of a list/tuple/dict/set as dataclass fields
    # to enable field specific operations on the items.

    @dispatch(argnum=1)
    def __init__(self, values: list[Any] | tuple[Any] | dict[str, Any] | set[Any]):
        # raise an error if the input is not a list/tuple/dict/set
        raise NotImplementedError("values must be a list, tuple, set or dict")

    @__init__.register(list)
    @__init__.register(tuple)
    @__init__.register(set)
    def _(
        self,
        values: list[Any] | tuple[Any] | set[Any],
        *,
        keys: str | tuple[str] = None,
    ):
        """flatten the list/tuple/set and register each item as a dataclass field

        Args:
            values (list[Any] | tuple[Any] | set[Any]): _description_
            keys (str | tuple[str], optional): Name tuple for each value. Defaults to None.

        Returns:
            TypeError: treeclass wrapped class.
        """
        # this dispatch offers a name argument for the container children
        if keys is None:
            keys = tuple(f"node_{i}" for i, _ in enumerate(values))
        elif isinstance(keys, str):
            keys = tuple(f"{keys}_{i}" for i, _ in enumerate(values))
        elif isinstance(keys, (tuple, list)):
            assert len(keys) == len(values), "name must have the same length as values"
        else:
            raise TypeError("keys argument is not understood.")

        self.keys = keys
        # self.container_type = type(values)
        for i, (key, value) in enumerate(zip(keys, values)):
            self.param(value, name=key)

    @__init__.register(dict)
    def _(self, values: dict[str, Any]):
        """flatten the dict and register each item as a dataclass field

        Args:
            values (dict[str, Any]): dict items to be registered as dataclass fields
        """
        # self.container_type = dict
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

    # def items(self):
    #     """ return the items of the container in its input form"""
    #     @dispatch(argnum=0)
    #     def _items(x):
    #         return getattr(self, self.keys[0])

    #     @_items.register(list)
    #     def _(x):
    #         return [getattr(self, key) for key in self.keys]

    #     @_items.register(tuple)
    #     def _(x):
    #         return tuple([getattr(self, key) for key in self.keys])

    #     @_items.register(set)
    #     def _(x):
    #         return set([getattr(self, key) for key in self.keys])

    #     @_items.register(dict)
    #     def _(x):
    #         return {key: getattr(self, key) for key in self.keys}

    #     return _items(self.container_type)
