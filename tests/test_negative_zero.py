import math
import os

import polars as pl
import pytest

import polars_fwf as pfwf


def test_negative_zero_handling(tmp_path):
    # 1. Prepare FWF data with negative zeros
    # int col: "-0", float col: "-0.0"
    content = "-0-0.0\n"
    path = str(tmp_path / "neg_zero.fwf")
    with open(path, "w") as f:
        f.write(content)

    specs = [pfwf.FieldSpec("i", 0, 2, "int"), pfwf.FieldSpec("f", 2, 4, "float")]

    # 2. Read back
    df = pfwf.read_fwf(path, specs)

    # Integers: -0 is just 0
    assert df["i"][0] == 0

    # Floats: -0.0 should have the sign bit preserved
    f_val = df["f"][0]
    assert f_val == 0.0
    # math.copysign is a reliable way to check the sign bit of 0.0
    assert math.copysign(1.0, f_val) == -1.0


def test_write_negative_zero(tmp_path):
    path = str(tmp_path / "write_neg_zero.fwf")

    # Create DataFrame with negative zero
    df = pl.DataFrame({"f": [math.copysign(0.0, -1.0)]})
    assert math.copysign(1.0, df["f"][0]) == -1.0

    # Spec width 4 is enough for "-0.0" (standard polars string repr)
    specs = [pfwf.FieldSpec("f", 0, 4, "float")]

    pfwf.write_fwf(df, path, specs=specs, max_decimals=1)

    with open(path, "r") as f:
        line = f.read()
        # Standard polars cast(String) for -0.0 is "-0.0"
        assert line == "-0.0\n"


def test_write_negative_zero_validation_fail(tmp_path):
    path = str(tmp_path / "fail_neg_zero.fwf")
    df = pl.DataFrame({"f": [math.copysign(0.0, -1.0)]})

    # Width 3 is NOT enough for "-0.0"
    specs = [pfwf.FieldSpec("f", 0, 3, "float")]

    with pytest.raises(
        ValueError, match=r"has data longer \(4\) than specified length \(3\)"
    ):
        pfwf.write_fwf(df, path, specs=specs, max_decimals=1)


if __name__ == "__main__":
    pytest.main([__file__])
