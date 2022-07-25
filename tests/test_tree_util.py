from pytreeclass import treeclass
from pytreeclass.src.tree_util import is_treeclass, is_treeclass_leaf


@treeclass
class Test:
    a: int = 10


a = Test()
b = (1, "s", 1.0, [2, 3])


@treeclass
class Test2:
    a: int = 1
    b: Test = Test()


def test_is_treeclass():
    assert is_treeclass(a) is True
    assert all(is_treeclass(bi) for bi in b) is False


def test_is_treeclass_leaf():
    assert is_treeclass_leaf(a) is True
    assert all(is_treeclass_leaf(bi) for bi in b) is False
    assert is_treeclass_leaf(Test2()) is False
    assert is_treeclass_leaf(Test2().b) is True
