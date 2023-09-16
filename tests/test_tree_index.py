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


import re
from collections import namedtuple
from typing import NamedTuple

import pytest

from pytreeclass._src.backend import backend
from pytreeclass._src.backend import numpy as np
from pytreeclass._src.backend import tree_util as tu
from pytreeclass._src.code_build import autoinit
from pytreeclass._src.tree_base import (
    TreeClass,
    add_mutable_entry,
    discard_mutable_entry,
)
from pytreeclass._src.tree_index import AtIndexer, BaseKey
from pytreeclass._src.tree_util import is_tree_equal, leafwise


@leafwise
class ClassTree(TreeClass):
    """Tree class for testing."""

    def __init__(self, a: int, b: dict, e: int):
        """Initialize."""
        self.a = a
        self.b = b
        self.e = e


@leafwise
class ClassSubTree(TreeClass):
    """Tree class for testing."""

    def __init__(self, c: int, d: int):
        """Initialize."""
        self.c = c
        self.d = d


# key
tree1 = dict(a=1, b=dict(c=2, d=3), e=4)
tree2 = ClassTree(1, dict(c=2, d=3), 4)
tree3 = ClassTree(1, ClassSubTree(2, 3), 4)

# index
tree4 = [1, [2, 3], 4]
tree5 = (1, (2, 3), 4)
tree6 = [1, ClassSubTree(2, 3), 4]

# mixed
tree7 = dict(a=1, b=[2, 3], c=4)
tree8 = dict(a=1, b=ClassSubTree(c=2, d=3), e=4)

# by mask
tree9 = ClassTree(1, dict(c=2, d=3), np.array([4, 5, 6]))

_X = 1_000

default_int = np.int32 if backend == "jax" else np.int64


@pytest.mark.parametrize(
    ["tree", "expected", "where"],
    [
        # by name
        [tree1, dict(a=None, b=dict(c=2, d=None), e=None), ("b", "c")],
        [tree2, ClassTree(None, dict(c=2, d=None), None), ("b", "c")],
        [tree3, ClassTree(None, ClassSubTree(2, None), None), ("b", "c")],
        # by index
        [tree3, ClassTree(None, ClassSubTree(c=2, d=None), None), (1, 0)],
        [tree4, [None, [2, None], None], (1, 0)],
        [tree5, (None, (2, None), None), (1, 0)],
        [tree6, [None, ClassSubTree(2, None), None], (1, 0)],
        # mixed
        [tree7, dict(a=None, b=[2, None], c=None), ("b", 0)],
        [tree8, dict(a=None, b=ClassSubTree(c=2, d=None), e=None), ("b", 0)],
        # by regex
        [tree1, dict(a=None, b=dict(c=2, d=None), e=None), ("b", re.compile("c"))],
        [tree2, ClassTree(None, dict(c=2, d=None), None), ("b", re.compile("c"))],
        [tree3, ClassTree(None, ClassSubTree(2, None), None), ("b", re.compile("c"))],
        # by boolean mask
        [tree9, ClassTree(None, dict(c=None, d=3), np.array([4, 5, 6])), (tree9 > 2,)],
        [tree9, ClassTree(None, dict(c=None, d=None), np.array([5, 6])), (tree9 > 4,)],
        [tree9, tree9, (tree9 == tree9,)],
        [
            tree9,
            ClassTree(None, dict(c=None, d=None), np.array([], dtype=default_int)),
            (tree9 != tree9,),
        ],
        # by ellipsis
        [tree1, tree1, (...,)],
        [tree2, tree2, (...,)],
        [tree3, tree3, (...,)],
        [tree4, tree4, (...,)],
        [tree5, tree5, (...,)],
        [tree6, tree6, (...,)],
        [tree7, tree7, (...,)],
        [tree8, tree8, (...,)],
        [tree9, tree9, (...,)],
    ],
)
def test_indexer_get(tree, expected, where):
    indexer = AtIndexer(tree, where=where)
    assert is_tree_equal(indexer.get(), expected)
    assert is_tree_equal(indexer.get(is_parallel=True), expected)


@pytest.mark.parametrize(
    ["tree", "expected", "where", "set_value"],
    [
        # by name
        [tree1, dict(a=1, b=dict(c=_X, d=3), e=4), ("b", "c"), _X],
        [tree2, ClassTree(1, dict(c=_X, d=3), 4), ("b", "c"), _X],
        [tree3, ClassTree(1, ClassSubTree(_X, 3), 4), ("b", "c"), _X],
        # by index
        [tree3, ClassTree(1, ClassSubTree(_X, 3), 4), (1, 0), _X],
        [tree4, [1, [_X, 3], 4], (1, 0), _X],
        [tree5, (1, (_X, 3), 4), (1, 0), _X],
        [tree6, [1, ClassSubTree(_X, 3), 4], (1, 0), _X],
        # mixed
        [tree7, dict(a=1, b=[2, _X], c=4), ("b", 1), _X],
        [tree8, dict(a=1, b=ClassSubTree(c=2, d=_X), e=4), ("b", 1), _X],
        # by regex
        [tree1, dict(a=1, b=dict(c=_X, d=3), e=4), ("b", re.compile("c")), _X],
        [tree2, ClassTree(1, dict(c=_X, d=3), 4), ("b", re.compile("c")), _X],
        [tree3, ClassTree(1, ClassSubTree(_X, 3), 4), ("b", re.compile("c")), _X],
        # by boolean mask
        [
            tree9,
            ClassTree(1, dict(c=2, d=_X), np.array([_X, _X, _X])),
            (tree9 > 2,),
            _X,
        ],
        [tree9, ClassTree(1, dict(c=2, d=3), np.array([4, _X, _X])), (tree9 > 4,), _X],
        [
            tree9,
            ClassTree(_X, dict(c=_X, d=_X), np.array([_X, _X, _X])),
            (tree9 == tree9,),
            _X,
        ],
        [tree9, tree9, (tree9 != tree9,), _X],
        # by ellipsis
        [
            tree1,
            dict(a=_X, b=dict(c=_X, d=_X), e=_X),
            (...,),
            dict(a=_X, b=dict(c=_X, d=_X), e=_X),
        ],
        [tree2, ClassTree(_X, dict(c=_X, d=_X), _X), (...,), _X],
        [tree3, ClassTree(_X, ClassSubTree(_X, _X), _X), (...,), _X],
        [tree4, [_X, [_X, _X], _X], (...,), _X],
        [tree5, (_X, (_X, _X), _X), (...,), _X],
        [tree6, [_X, ClassSubTree(_X, _X), _X], (...,), _X],
        [tree7, dict(a=_X, b=[_X, _X], c=_X), (...,), _X],
        [tree8, dict(a=_X, b=ClassSubTree(c=_X, d=_X), e=_X), (...,), _X],
        [
            tree9,
            ClassTree(_X, dict(c=_X, d=_X), _X),
            (...,),
            ClassTree(_X, dict(c=_X, d=_X), _X),  # broadcastable option
        ],
    ],
)
def test_indexer_set(tree, expected, where, set_value):
    indexer = AtIndexer(tree, where=where)
    assert is_tree_equal(indexer.set(set_value), expected)
    assert is_tree_equal(indexer.set(set_value, is_parallel=True), expected)


@pytest.mark.parametrize(
    ["tree", "expected", "where"],
    [
        # by name
        [tree1, dict(a=1, b=dict(c=_X, d=3), e=4), ("b", "c")],
        [tree2, ClassTree(1, dict(c=_X, d=3), 4), ("b", "c")],
        [tree3, ClassTree(1, ClassSubTree(_X, 3), 4), ("b", "c")],
        # by index
        [tree4, [1, [_X, 3], 4], (1, 0)],
        [tree5, (1, (_X, 3), 4), (1, 0)],
        [tree6, [1, ClassSubTree(_X, 3), 4], (1, 0)],
        # mixed
        [tree7, dict(a=1, b=[2, _X], c=4), ("b", 1)],
        [tree8, dict(a=1, b=ClassSubTree(c=2, d=_X), e=4), ("b", 1)],
        # by regex
        [tree1, dict(a=1, b=dict(c=_X, d=3), e=4), ("b", re.compile("c"))],
        [tree2, ClassTree(1, dict(c=_X, d=3), 4), ("b", re.compile("c"))],
        [tree3, ClassTree(1, ClassSubTree(_X, 3), 4), ("b", re.compile("c"))],
        # by boolean mask
        [tree9, ClassTree(1, dict(c=2, d=_X), np.array([_X, _X, _X])), (tree9 > 2,)],
        [tree9, ClassTree(1, dict(c=2, d=3), np.array([4, _X, _X])), (tree9 > 4,)],
        [
            tree9,
            ClassTree(_X, dict(c=_X, d=_X), np.array([_X, _X, _X])),
            (tree9 == tree9,),
        ],
        [tree9, tree9, (tree9 != tree9,)],
        # by ellipsis
        [tree1, dict(a=_X, b=dict(c=_X, d=_X), e=_X), (...,)],
        [tree2, ClassTree(_X, dict(c=_X, d=_X), _X), (...,)],
        [tree3, ClassTree(_X, ClassSubTree(_X, _X), _X), (...,)],
        [tree4, [_X, [_X, _X], _X], (...,)],
        [tree5, (_X, (_X, _X), _X), (...,)],
        [tree6, [_X, ClassSubTree(_X, _X), _X], (...,)],
        [tree7, dict(a=_X, b=[_X, _X], c=_X), (...,)],
        [tree8, dict(a=_X, b=ClassSubTree(c=_X, d=_X), e=_X), (...,)],
        [tree9, ClassTree(_X, dict(c=_X, d=_X), _X), (...,)],
    ],
)
def test_indexer_apply(tree, expected, where):
    indexer = AtIndexer(tree, where=where)
    assert is_tree_equal(indexer.apply(lambda _: _X), expected)
    assert is_tree_equal(
        indexer.apply(lambda _: _X, is_parallel=True),
        expected,
    )


@pytest.mark.parametrize(
    ["tree", "expected", "where"],
    [
        # by name
        [tree1, 5, ("b", ("c", "d"))],
        [tree2, 5, ("b", ("c", "d"))],
        [tree3, 5, ("b", ("c", "d"))],
        # by index
        [tree4, 5, (1, (0, 1))],
        [tree5, 5, (1, (0, 1))],
        # mixed
        [tree7, 5, ("b", (0, 1))],
        # by regex
        [tree1, 5, ("b", re.compile("c|d"))],
        [tree2, 5, ("b", re.compile("c|d"))],
        [tree3, 5, ("b", re.compile("c|d"))],
        # by boolean mask
        [tree9, 3 + np.array([4, 5, 6]), (tree9 > 2,)],
        [tree9, np.array([5, 6]), (tree9 > 4,)],
        [tree9, 1 + 2 + 3 + np.array([4, 5, 6]), (tree9 == tree9,)],
        [
            tree9,
            0 + np.array([], dtype=default_int),
            (tree9 != tree9,),
        ],
        # by ellipsis
        [tree1, 1 + 2 + 3 + 4, (...,)],
        [tree2, 1 + 2 + 3 + 4, (...,)],
        [tree3, 1 + 2 + 3 + 4, (...,)],
        [tree4, 1 + 2 + 3 + 4, (...,)],
        [tree5, 1 + 2 + 3 + 4, (...,)],
        [tree6, 1 + 2 + 3 + 4, (...,)],
        [tree7, 1 + 2 + 3 + 4, (...,)],
        [tree8, 1 + 2 + 3 + 4, (...,)],
        [tree9, 1 + 2 + 3 + np.array([4, 5, 6]), (...,)],
    ],
)
def test_indexer_reduce(tree, expected, where):
    indexer = AtIndexer(tree, where=where)
    assert is_tree_equal(
        indexer.reduce(lambda x, y: x + y, initializer=0),
        expected,
    )


@pytest.mark.parametrize(
    ["tree", "expected", "where"],
    [
        # by name
        [tree1, (dict(a=1, b=dict(c=2, d=5), e=4), 3), ("b", ("c", "d"))],
        [tree2, (ClassTree(1, dict(c=2, d=5), 4), 3), ("b", ("c", "d"))],
        [tree3, (ClassTree(1, ClassSubTree(2, 5), 4), 3), ("b", ("c", "d"))],
        # by index
        [tree4, ([1, [2, 5], 4], 3), (1, (0, 1))],
        [tree5, ((1, (2, 5), 4), 3), (1, (0, 1))],
        # mixed
        [tree7, (dict(a=1, b=[2, 5], c=4), 3), ("b", (0, 1))],
        # [tree8, (dict(a=1, b=ClassSubTree(c=2, d=5), e=4), 3), ("b", (0, 1))],
        # by regex
        [tree1, (dict(a=1, b=dict(c=2, d=5), e=4), 3), ("b", re.compile("c|d"))],
        [tree2, (ClassTree(1, dict(c=2, d=5), 4), 3), ("b", re.compile("c|d"))],
        [tree3, (ClassTree(1, ClassSubTree(2, 5), 4), 3), ("b", re.compile("c|d"))],
    ],
)
def test_indexer_scan(tree, expected, where):
    indexer = AtIndexer(tree, where=where)
    assert is_tree_equal(
        indexer.scan(lambda x, s: (x + s, x), state=0),
        expected,
    )


def test_method_call():
    @leafwise
    @autoinit
    class Tree(TreeClass):
        a: int = 1

        def increment(self):
            self.a += 1

        def show(self):
            return 1

    t = Tree()

    @autoinit
    class Tree2(TreeClass):
        b: Tree = Tree()

    assert is_tree_equal(t.at["increment"]()[1], Tree(2))
    assert is_tree_equal(Tree2().at["b"]["show"]()[0], 1)

    with pytest.raises(AttributeError):
        t.at["bla"]()

    with pytest.raises(TypeError):
        t.at["a"]()

    @leafwise
    @autoinit
    class A(TreeClass):
        a: int

        def __call__(self, x):
            self.a += x
            return x

    a = A(1)
    _, b = a.at["__call__"](2)

    assert tu.tree_flatten(a)[0] == [1]
    assert tu.tree_flatten(b)[0] == [3]

    with pytest.raises(TypeError):
        a.at[0](1)


def test_call_context():
    @autoinit
    class L2(TreeClass):
        a: int = 1

        def delete(self, name):
            del self.a

    t = L2()

    add_mutable_entry(t)
    t.delete("a")
    discard_mutable_entry(t)

    with pytest.raises(AttributeError):
        t.delete("a")


@pytest.mark.parametrize("where", [(None,), ("a", [1]), (0, [1])])
def test_unsupported_where(where):
    t = namedtuple("a", ["x", "y"])(1, 2)
    with pytest.raises(NotImplementedError):
        AtIndexer(t, where=where).get()


def test_custom_key():
    class NameTypeContainer(NamedTuple):
        name: str
        type: type

    class Tree:
        def __init__(self, a, b) -> None:
            self.a = a
            self.b = b

        @property
        def at(self):
            return AtIndexer(self)

    if backend == "jax":
        import jax.tree_util as jtu

        def tree_flatten(tree):
            return (tree.a, tree.b), None

        def tree_unflatten(aux_data, children):
            return Tree(*children)

        def tree_flatten_with_keys(tree):
            ak = (NameTypeContainer("a", type(tree.a)), tree.a)
            bk = (NameTypeContainer("b", type(tree.b)), tree.b)
            return (ak, bk), None

        jtu.register_pytree_with_keys(
            nodetype=Tree,
            flatten_func=tree_flatten,
            flatten_with_keys=tree_flatten_with_keys,
            unflatten_func=tree_unflatten,
        )
    elif backend == "numpy":
        import optree as ot

        def tree_flatten(tree):
            ka = NameTypeContainer("a", type(tree.a))
            kb = NameTypeContainer("b", type(tree.b))
            return (tree.a, tree.b), None, (ka, kb)

        def tree_unflatten(aux_data, children):
            return Tree(*children)

        ot.register_pytree_node(
            Tree,
            flatten_func=tree_flatten,
            unflatten_func=tree_unflatten,
            namespace="PYTREECLASS",
        )

    tree = Tree(1, 2)

    class MatchNameType(BaseKey):
        def __init__(self, name, type):
            self.name = name
            self.type = type

        def __eq__(self, other):
            if isinstance(other, NameTypeContainer):
                return other == (self.name, self.type)
            return False

    assert tu.tree_flatten(tree.at[MatchNameType("a", int)].get())[0] == [1]


def test_repr_str():
    @autoinit
    class Tree(TreeClass):
        a: int = 1
        b: int = 2

    t = Tree()

    assert repr(t.at["a"]) == "TreeClassIndexer(tree=Tree(a=1, b=2), where=('a',))"
    assert str(t.at["a"]) == "TreeClassIndexer(tree=Tree(a=1, b=2), where=('a',))"
    assert repr(t.at[...]) == "TreeClassIndexer(tree=Tree(a=1, b=2), where=(Ellipsis,))"
