import os

import polars as pl

import polars_fwf as pfwf


def test_nan_handling():
    specs = [
        pfwf.FieldSpec("val", 0, 10, pfwf.DType.F64),
    ]
    path = "data/nan_test.fwf"
    os.makedirs("data", exist_ok=True)

    with open(path, "wb") as f:
        f.write(b"      1.23\n")
        f.write(b"       NaN\n")
        f.write(b"      4.56\n")
        f.write(b"       inf\n")
        f.write(b"      NULL\n")

    print(f"Reading {path} with F64 spec...")
    df = pfwf.read_fwf(path, specs)
    print("DataFrame:")
    print(df)

    # Check if NaN/inf/NULL are handled
    # Our current Rust impl uses lexical_core::parse, which might not handle 'NaN' strings by default
    # unless configured. If it fails, it should push Null per our ErrorStrategy.


if __name__ == "__main__":
    test_nan_handling()
