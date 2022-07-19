from __future__ import annotations

import copy

import jax
import jax.numpy as jnp

from pytreeclass.src.decorator_util import dispatch
from pytreeclass.src.tree_util import is_treeclass_leaf_bool


def node_setter(lhs, where, set_value):
    # do not change non-chosen values
    # assert isinstance(where, bool)

    if isinstance(lhs, jnp.ndarray):
        return jnp.where(where, set_value, lhs)
    else:
        return set_value if where else lhs


def node_getter(lhs, where):
    # not jittable as size can changes
    # does not change pytreestructure ,
    # but changes array sizes if fill_value=None

    if isinstance(lhs, jnp.ndarray):
        return lhs[where]
    else:
        # set None to non-chosen non-array values
        return lhs if where else None


def param_indexing_getter(model, *where: tuple[str, ...]):
    modelCopy = copy.copy(model)

    for field in model.__dataclass_fields__.values():
        value = modelCopy.__dict__[field.name]
        excluded_by_type = isinstance(value, str)
        excluded_by_meta = ("static" in field.metadata) and field.metadata["static"] is True  # fmt: skip
        excluded = excluded_by_type or excluded_by_meta
        if field.name not in where and not excluded:
            modelCopy.__dict__[field.name] = None

    return modelCopy


def param_indexing_setter(model, set_value, *where: tuple[str]):
    @dispatch(argnum=1)
    def _param_indexing_setter(model, set_value, *where: tuple[str]):
        raise NotImplementedError(f"Invalid set_value type = {type(set_value)}.")

    @_param_indexing_setter.register(float)
    @_param_indexing_setter.register(int)
    @_param_indexing_setter.register(complex)
    @_param_indexing_setter.register(jnp.ndarray)
    def set_scalar(model, set_value, *where: tuple[str]):
        modelCopy = model
        for field in model.__dataclass_fields__.values():
            value = modelCopy.__dict__[field.name]

            excluded_by_type = isinstance(value, str)
            excluded_by_meta = ("static" in field.metadata) and field.metadata["static"] is True  # fmt: skip
            excluded = excluded_by_meta or excluded_by_type
            if field.name in where and not excluded:
                modelCopy.__dict__[field.name] = node_setter(value, True, set_value)
        return modelCopy

    # @_param_indexing_setter.register(type(model))
    # def set_model(model, set_value, *where: tuple[str]):
    #     raise NotImplemented("Not yet implemented.")

    return _param_indexing_setter(model, set_value, *where)


def boolean_indexing_getter(model, where):

    lhs_leaves, lhs_treedef = model.flatten_leaves
    where_leaves, where_treedef = jax.tree_flatten(where)
    lhs_leaves = [
        node_getter(lhs_leaf, where_leaf)
        for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)
    ]

    return jax.tree_unflatten(lhs_treedef, lhs_leaves)


def boolean_indexing_setter(model, set_value, where):

    lhs_leaves, lhs_treedef = jax.tree_flatten(model)
    where_leaves, rhs_treedef = jax.tree_flatten(where)
    lhs_leaves = [
        node_setter(lhs_leaf, where_leaf, set_value=set_value,)
        for lhs_leaf, where_leaf in zip(lhs_leaves, where_leaves)  # fmt: skip
    ]

    return jax.tree_unflatten(lhs_treedef, lhs_leaves)


class treeIndexer:
    @property
    def at(self):
        class indexer:
            @dispatch(argnum=1)
            def __getitem__(inner_self, *args):
                raise NotImplementedError(
                    f"indexing with type{(tuple(type(arg) for arg in args))} is not implemented."
                )

            @__getitem__.register(str)
            @__getitem__.register(tuple)
            def __param_getitiem__(inner_self, *args):
                # indexing by param name
                flatten_args = jax.tree_leaves(args)
                if not all(isinstance(arg, str) for arg in flatten_args):
                    raise ValueError("Invalid indexing argument")

                class getterSetterIndexer:
                    def get(getter_setter_self):
                        # select by param name
                        return param_indexing_getter(self, *flatten_args)

                    def set(getter_setter_self, set_value):
                        # select by param name
                        return param_indexing_setter(
                            copy.copy(self), set_value, *flatten_args
                        )

                return getterSetterIndexer()

            @__getitem__.register(type(self))
            def __model_getitiem__(inner_self, arg):
                # indexing by model

                if not all(
                    is_treeclass_leaf_bool(leaf) for leaf in jax.tree_leaves(arg)
                ):
                    raise ValueError("model leaves argument must be boolean.")

                class getterSetterIndexer:
                    def get(getter_setter_self):
                        # select by class boolean x[x>1]
                        return boolean_indexing_getter(self, arg)

                    def set(getter_setter_self, set_value):
                        # select by class boolean
                        return boolean_indexing_setter(self, set_value, arg)

                return getterSetterIndexer()

        return indexer()

    def __getitem__(self, *args):
        # alias for .at[].get()
        return self.at.__getitem__(*args).get()
