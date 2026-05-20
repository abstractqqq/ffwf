import os

import polars as pl

import ffwf as fw
import ffwf.polars as plfw


def test_nan_handling_pl():
    specs = [
        fw.FieldSpec("val", 0, 10, fw.DType.F64),
    ]
    path = "data/nan_test.fwf"
    os.makedirs("data", exist_ok=True)

    lines = [
        b"      1.23\n",
        b"       NaN\n",
        b"       nan\n",
        b"       NAN\n",
        b"      4.56\n",
        b"       inf\n",
        b"       INF\n",
        b"      NULL\n",
    ]
    with open(path, "wb") as f:
        f.writelines(lines)

    print(f"Reading {path} with F64 spec...")
    df = plfw.read_fwf_pl(path, specs)
    print("DataFrame:")
    print(df)

    # Check if NaN/inf/NULL are handled
    assert df["val"][0] == 1.23
    assert pl.Series([df["val"][1]]).is_nan()[0]
    assert pl.Series([df["val"][2]]).is_nan()[0]
    assert pl.Series([df["val"][3]]).is_nan()[0]
    assert df["val"][4] == 4.56
    assert df["val"][5] == float("inf")
    assert df["val"][6] == float("inf")
    assert df["val"][7] is None


if __name__ == "__main__":
    test_nan_handling_pl()
