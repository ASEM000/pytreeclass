import pytest

import pytreeclass as pytc
from pytreeclass.src.misc import ImmutableInstanceError, mutableContext


def test_mutable_context():
    @pytc.treeclass
    class Test:
        x: int

        def __init__(self, x):
            self.x = x

    t = Test(1)

    with mutableContext(t):
        t.a = 12

    assert getattr(t, "a") == 12

    with pytest.raises(ImmutableInstanceError):
        t.a = 100

    with pytest.raises(AssertionError):
        with mutableContext(1):
            pass
