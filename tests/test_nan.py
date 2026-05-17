import os

import polars as pl

import fwf


def test_nan_handling():
    specs = [
        fwf.FieldSpec("val", 0, 10, fwf.DType.F64),
    ]
    path = "data/nan_test.fwf"
    os.makedirs("data", exist_ok=True)

    with open(path, "w") as f:
        f.write("      1.23\n")
        f.write("       NaN\n")
        f.write("      4.56\n")
        f.write("       inf\n")
        f.write("      NULL\n")

    print(f"Reading {path} with F64 spec...")
    df = fwf.read_fwf(path, specs)
    print("DataFrame:")
    print(df)

    # Check if NaN/inf/NULL are handled
    # Our current Rust impl uses lexical_core::parse, which might not handle 'NaN' strings by default
    # unless configured. If it fails, it should push Null per our ErrorStrategy.


if __name__ == "__main__":
    test_nan_handling()
