import functools
import inspect

""" Porting utils """


class cached_property:
    def __init__(self, func):
        self.name = func.__name__
        self.func = func

    def __get__(self, instance, owner):
        attr = self.func(instance)
        setattr(instance, self.name, attr)
        return attr


def dispatch(*args, **kwargs):
    # based on https://stackoverflow.com/a/24602374/10879163

    def dispatch_wrapper(func, argnum=1):
        """singledispatch by arg position"""
        dispatcher = functools.singledispatch(func)

        def wrapper(*ar, **kw):
            return dispatcher.dispatch(ar[argnum].__class__)(*ar, **kw)

        wrapper.register = dispatcher.register
        functools.update_wrapper(wrapper, func)
        return wrapper

    if len(args) > 0 and inspect.isfunction(args[0]):
        return dispatch_wrapper(args[0], 1)

    elif len(args) == 0 and len(kwargs) > 0:
        argnum = kwargs["argnum"] if "argnum" in kwargs else 1
        return functools.partial(dispatch_wrapper, argnum=argnum)


singledispatchmethod = functools.partial(dispatch, argnum=1)
