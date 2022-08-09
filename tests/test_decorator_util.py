import pytest

from pytreeclass.src.decorator_util import cached_property, dispatch


def test_cached_property():
    class test:
        def __init__(self, counter):
            self.counter = counter

        @cached_property
        def counter_cached_property(self):
            return self.counter

        @property
        def counter_property(self):
            return self.counter

    a = test(counter=10)
    assert a.counter_property == 10
    assert a.counter_cached_property == 10
    a.counter += 1
    assert a.counter_property == 11
    assert a.counter_cached_property == 10


def test_dispatch():
    @dispatch(argnum=0)
    def func(x):
        raise ValueError("unknown")

    @func.register(int)
    def _(x):
        return x + 1

    @func.register(float)
    def _(x):
        return x + 100

    assert func(1) == 2
    assert func(1.0) == 101.0

    with pytest.raises(ValueError):
        func("s")

    @dispatch(argnum=1)
    def func(x, y):
        raise ValueError("unknown")

    @func.register(int)
    def _(x, y):
        return y + 1

    @func.register(float)
    def _(x, y):
        return y + 100

    assert func("a", 1) == 2
    assert func(None, 1.0) == 101.0

    with pytest.raises(ValueError):
        func(None, "s")

    # dispatch by keyword argument
    @dispatch(argnum="name")
    def func(x, y, *, name):
        raise ValueError("unknown")

    @func.register(int)
    def _(x, y, *, name):
        return "int"

    @func.register(str)
    def _(x, y, *, name):
        return "str"

    assert func(1, 2, name=1) == "int"
    assert func(1, 1, name="s") == "str"

    with pytest.raises(ValueError):
        func(1, 1, name=1.0)


def test_singledispatchmethod():
    class test:
        @dispatch(argnum=1)
        def plus(self, x):
            raise ValueError("unknown")

        @plus.register(int)
        def _(self, x):
            return x + 1

        @plus.register(complex)
        @plus.register(float)
        def _(self, x):
            return x + 100.0

    A = test()

    assert A.plus(1) == 2
    assert A.plus(1.0) == 101.0
    assert A.plus(complex(1.0)) == complex(101.0)

    with pytest.raises(ValueError):
        A.plus("s")

    with pytest.raises(ValueError):

        class test:
            @dispatch(argnum=1.0)
            def plus(self, x): ...

        t = test()
        t.plus("s")
        t.plus(1)

    @dispatch
    def fn(x):
        raise ValueError("unknown")

    @fn.register(int)
    def _(x):
        return 1

    assert fn(3) == 1

    with pytest.raises(ValueError):
        fn(3.0)
