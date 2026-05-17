import polars as pl

import polars_fwf as pfwf


def test_native_polars():
    specs = [
        pfwf.FieldSpec("id", 0, 5, pfwf.DType.I32),
        pfwf.FieldSpec("val", 5, 10, pfwf.DType.F64),
        pfwf.FieldSpec("tag", 15, 5, pfwf.DType.String),
    ]

    # Now calling directly via pl namespace
    df = pl.read_fwf("data/test_data.fwf", specs)

    print("DataFrame via pl.read_fwf:")
    print(df)

    assert df.shape == (3, 3)
    assert df["id"][0] == 1
    print("Native Polars plugin test passed!")


if __name__ == "__main__":
    test_native_polars()
