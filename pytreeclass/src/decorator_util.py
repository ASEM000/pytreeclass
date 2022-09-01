from __future__ import annotations

import functools as ft
import inspect

""" Porting utils """


class cached_property:
    def __init__(self, func):
        self.name = func.__name__
        self.func = func

    def __get__(self, instance, owner):
        attr = self.func(instance)
        object.__setattr__(instance, self.name, attr)
        return attr


def dispatch(*args, **kwargs):
    """ft.singledispatch with option of choosing the position of the dispatched argument
    or keyword argument.

    For multiple dispatch use the following pattern

    >>> @dispatch(argnum=0)
    ... def a(x,y): ...

    >>> @a.register(int)
    ... @dispatch(argnum=1)
    ... def b(x,y) : ...

    >>> @b.register(int)
    ... def _(x,y):
    ...     return "int,int"
    """

    def dispatch_wrapper(func, argnum: int | str = 0):
        """singledispatch by arg position/kw arg name"""
        dispatcher = ft.singledispatch(func)

        def wrapper(*ar, **kw):
            if isinstance(argnum, int):
                # based on https://stackoverflow.com/a/24602374/10879163
                return dispatcher.dispatch(ar[argnum].__class__)(*ar, **kw)

            elif isinstance(argnum, str):
                # dispatch by keyword argument
                return dispatcher.dispatch(kw[argnum].__class__)(*ar, **kw)

            else:
                raise ValueError("argnum must be int or str")

        wrapper.register = dispatcher.register
        ft.update_wrapper(wrapper, func)
        return wrapper

    if len(args) == 1 and inspect.isfunction(args[0]):
        # @dispatch
        # def f(..):
        return dispatch_wrapper(args[0], 0)

    elif len(args) == 0 and len(kwargs) > 0:
        # @dispatch(argnum=...)
        # def f(..):
        argnum = kwargs.get("argnum", 0)
        return ft.partial(dispatch_wrapper, argnum=argnum)