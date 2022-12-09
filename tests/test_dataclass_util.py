# import dataclasses

# from pytreeclass._src.dataclass_util import (
#     # is_dataclass_leaf,
#     # is_dataclass_non_leaf,
#     # is_leaf_bool
# )

# # def test_field():
# #     f = field(nondiff=True)
# #     # assert f.metadata == {"static": "nondiff"}
# #     # ff = _field_copy(f)
# #     # assert hash(f) != hash(ff)


# def test_is_dataclass():
#     # @dataclasses.dataclass
#     # class Test:
#     #     a: int = field(default=1, nondiff=True)

#     # assert is_dataclass_fields_nondiff(Test()) is True

#     # @dataclasses.dataclass
#     # class Test:
#     #     a: int = dataclasses.field(default=1, metadata={"static": "frozen"})

#     # assert is_dataclass_fields_frozen(Test()) is True

#     @dataclasses.dataclass
#     class Test:
#         a: bool = True

#     # assert is_leaf_bool(Test().a) is True
#     # assert is_dataclass_leaf(Test()) is True

#     @dataclasses.dataclass
#     class Test2:
#         a: Test = Test()

#     # assert is_dataclass_leaf(Test2()) is False
#     # assert is_dataclass_non_leaf(Test2()) is True
