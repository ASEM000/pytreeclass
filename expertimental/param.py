# def param(
#     tree, node: Any, *, name: str, static: bool = False, repr: bool = True
# ) -> Any:
#     """Add and return a parameter to the treeclass in a compact way.

#     Note:
#         If the node is already defined (checks by name) then it will be returned
#         Useful if node definition

#     Args:
#         node (Any): Any node to be added to the treeclass
#         name (str): Name of the node
#         static (bool, optional): Whether to exclude from tree leaves. Defaults to False.
#         repr (bool, optional): whether to show in repr/str/tree_viz . Defaults to True.


#     Example:
#         @pytc.treeclass
#         class StackedLinear:

#         def __init__(self,key):
#             self.keys = jax.random.split(key,3)

#         def __call__(self,x):
#             x = self.param(... ,name="l1")(x)
#             return x
#     """
#     if not is_treeclass(tree):
#         raise TypeError(f"param can only be applied to treeclass. Found {type(tree)}")

#     if hasattr(tree, name) and (name in tree.__undeclared_fields__):
#         return getattr(tree, name)

#     # create field
#     field_value = field(repr=repr, metadata={"static": static, "param": True})

#     object.__setattr__(field_value, "name", name)
#     object.__setattr__(field_value, "type", type(node))

#     # register it to class
#     tree.__undeclared_fields__.update({name: field_value})
#     object.__setattr__(tree, name, node)

#     return getattr(tree, name)
