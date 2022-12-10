import pytest

import pytreeclass as pytc


def test_field():

    with pytest.raises(ValueError):
        pytc.field(default=1, default_factory=lambda: 1)

    assert pytc.field(default=1).default == 1
