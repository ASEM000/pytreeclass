import pytest

from pytreeclass.src.container import Container


def test_container():

    a = Container([1, 2, 3])
    assert f"{a!r}" == f"{a!s}" == "Container([0]=1,[1]=2,[2]=3)"
    assert len(a.__undeclared_fields__) == 3
    assert (
        f"{a.tree_diagram()}"
        == "Container\n    ├── [0]=1\n    ├── [1]=2\n    └── [2]=3   "
    )
    assert a[0] == 1
    assert a["[1]"] == 2

    b = Container((1, 2, 3))
    assert f"{b!r}" == f"{b!s}" == "Container((0)=1,(1)=2,(2)=3)"
    assert len(b.__undeclared_fields__) == 3
    assert (
        f"{b.tree_diagram()}"
        == "Container\n    ├── (0)=1\n    ├── (1)=2\n    └── (2)=3   "
    )
    assert b[0] == 1
    assert b["(1)"] == 2

    c = Container({"a": 1, "b": 2, "c": 3})
    assert f"{c!r}" == f"{c!s}" == "Container(a=1,b=2,c=3)"
    assert len(c.__undeclared_fields__) == 3
    assert f"{c.tree_diagram()}" == "Container\n    ├── a=1\n    ├── b=2\n    └── c=3 "

    assert c["a"] == 1
    assert c["b"] == 2
    assert c["c"] == 3
    assert c[0] == 1
    assert c[1] == 2
    assert c[2] == 3

    d = Container({1, 2, 3})
    assert f"{d!r}" == f"{d!s}" == "Container({0}=1,{1}=2,{2}=3)"
    assert len(d.__undeclared_fields__) == 3
    assert (
        f"{d.tree_diagram()}"
        == "Container\n    ├── {0}=1\n    ├── {1}=2\n    └── {2}=3   "
    )

    assert d[0] == 1
    assert d["{1}"] == 2


def test_container_error():

    with pytest.raises(NotImplementedError):
        Container(1)

    with pytest.raises(NotImplementedError):
        Container("abc")


def test_items():

    a = Container([1, 2, 3])
    assert a.items() == [1, 2, 3]

    b = Container({"a": 1, "b": 2, "c": 3})
    assert b.items() == {"a": 1, "b": 2, "c": 3}

    c = Container({1, 2, 3})
    assert c.items() == {1, 2, 3}

    d = Container((1, 2, 3))
    assert d.items() == (1, 2, 3)
