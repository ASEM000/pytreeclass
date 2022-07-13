from pytreeclass import treeclass
from pytreeclass.src.utils import (
    is_treeclass,
    is_treeclass_leaf,
    leaves_param_count_and_size,
    leaves_param_format,
    node_class_name,
)


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


def test_param_op():
    assert node_class_name(Test()) == "Test"

    T = Test()

    dynamic_leaves = [leave.tree_fields[0] for leave in T.treeclass_leaves]

    leaves_name = [node_class_name(leaf) for leaf in T.treeclass_leaves]
    params_count, params_size = zip(*leaves_param_count_and_size(dynamic_leaves))
    params_repr = leaves_param_format(dynamic_leaves)

    assert leaves_name == ["Test"]
    assert params_count == (complex(0, 1),)
    assert params_repr == [{"a": "10"}]
    assert params_size == (complex(0, 28),)
