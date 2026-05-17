import polars as pl

import fwf


def test_native_polars():
    specs = [
        fwf.FieldSpec("id", 0, 5, fwf.DType.I32),
        fwf.FieldSpec("val", 5, 10, fwf.DType.F64),
        fwf.FieldSpec("tag", 15, 5, fwf.DType.String),
    ]

    # Now calling directly via pl namespace
    df = pl.read_fwf("data/test_data.fwf", specs, 21)

    print("DataFrame via pl.read_fwf:")
    print(df)

    assert df.shape == (3, 3)
    assert df["id"][0] == 1
    print("Native Polars plugin test passed!")


if __name__ == "__main__":
    test_native_polars()
