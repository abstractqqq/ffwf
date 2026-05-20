import math
import os

import polars as pl
import pytest

import ffwf as fw
import ffwf.polars as plfw


def test_negative_zero_handling_pl(tmp_path):
    # 1. Prepare FWF data with negative zeros
    # int col: "-0", float col: "-0.0"
    content = b"-0-0.0\n"
    path = str(tmp_path / "neg_zero.fwf")
    with open(path, "wb") as f:
        f.write(content)

    specs = [fw.FieldSpec("i", 0, 2, "int"), fw.FieldSpec("f", 2, 4, "float")]

    # 2. Read back
    df = plfw.read_fwf_pl(path, specs)

    # Integers: -0 is just 0
    assert df["i"][0] == 0

    # Floats: -0.0 should have the sign bit preserved
    f_val = df["f"][0]
    assert f_val == 0.0
    # math.copysign is a reliable way to check the sign bit of 0.0
    assert math.copysign(1.0, f_val) == -1.0


def test_write_negative_zero_pl(tmp_path):
    path = str(tmp_path / "write_neg_zero.fwf")

    # Create DataFrame with negative zero
    df = pl.DataFrame({"f": [math.copysign(0.0, -1.0)]})
    assert math.copysign(1.0, df["f"][0]) == -1.0

    # Spec width 4 is enough for "-0.0" (standard polars string repr)
    specs = [fw.FieldSpec("f", 0, 4, "float")]

    plfw.write_fwf_pl(df, path, specs=specs, decimals=1)

    with open(path, "rb") as f:
        line = f.read()
        # Standard polars cast(String) for -0.0 is "-0.0"
        assert line == b"-0.0\n"


def test_write_negative_zero_validation_fail_pl(tmp_path):
    path = str(tmp_path / "fail_neg_zero.fwf")
    df = pl.DataFrame({"f": [math.copysign(0.0, -1.0)]})

    # Width 3 is NOT enough for "-0.0"
    specs = [fw.FieldSpec("f", 0, 3, "float")]

    with pytest.raises(
        ValueError, match=r"has data longer \(4\) than specified length \(3\)"
    ):
        plfw.write_fwf_pl(df, path, specs=specs, decimals=1)


if __name__ == "__main__":
    pytest.main([__file__])
