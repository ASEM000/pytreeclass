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
        names: str | tuple[str] = None,
    ):
        """flatten the list/tuple/set and register each item as a dataclass field

        Args:
            values (list[Any] | tuple[Any] | set[Any]): _description_
            names (str | tuple[str], optional): Name tuple for each value. Defaults to None.

        Returns:
            TypeError: treeclass wrapped class.
        """
        # this dispatch offers a name argument for the container children
        if names is None:
            names = ("node",) * len(values)
        elif isinstance(names, str):
            names = (names,) * len(values)
        elif isinstance(names, (tuple, list)):
            assert len(names) == len(values), "name must have the same length as values"
        else:
            raise TypeError("names argument is not understood.")

        self.keys = ()
        for i, (key, value) in enumerate(zip(names, values)):
            self.keys += (f"{key}_{i}",)
            self.param(value, name=self.keys[-1])

    @__init__.register(dict)
    def _(self, values: dict[str, Any]):
        """flatten the dict and register each item as a dataclass field

        Args:
            values (dict[str, Any]): dict items to be registered as dataclass fields
        """
        for key, value in values.items():
            self.param(value, name=key)

        self.keys = tuple(values.keys())

    @dispatch(argnum=1)
    def __getitem__(self, key):
        raise NotImplementedError("key must be an int or a str")

    @__getitem__.register(str)
    def _(self, key: str):
        # return the item with the given key
        return self.__dict__[key]

    @__getitem__.register(int)
    def _(self, index: int):
        # return the item with the given index
        return self.__dict__[self.keys[index]]
