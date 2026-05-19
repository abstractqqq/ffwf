import os

import polars_fwf as pfwf


def test_read():
    specs = [
        pfwf.FieldSpec("id", 0, 5, pfwf.DType.I32),
        pfwf.FieldSpec("val", 5, 10, pfwf.DType.F64),
        pfwf.FieldSpec("tag", 15, 5, pfwf.DType.String),
    ]

    # Ensure test data is written with LF (\n)
    path = "data/test_data.fwf"
    os.makedirs("data", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"00001      1.23 ABC \n00002      4.56 DEF \n00003      7.89 GHI \n")

    df = pfwf.read_fwf(path, specs)
    print("DataFrame:")
    print(df)

    assert df.shape == (3, 3)
    assert df["id"][0] == 1
    assert df["tag"][2] == "GHI"
    print("Python test passed!")


def test_partial_specs():
    path = "data/partial_test.fwf"
    os.makedirs("data", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"1234567890\n")
        f.write(b"ABCDEFGHIJ\n")

    # Partial spec: only col 1 (0-3) and col 2 (6-9)
    # Total width = 3 + 3 = 6. Line width = 10. Gaps at 3-6 and 9-10.
    specs = [
        pfwf.FieldSpec("c1", 0, 3, "str"),
        pfwf.FieldSpec("c2", 6, 3, "str"),
    ]

    # Test Eager
    df_eager = pfwf.read_fwf(path, specs)
    assert df_eager.shape == (2, 2)
    assert df_eager["c1"].to_list() == ["123", "ABC"]
    assert df_eager["c2"].to_list() == ["789", "GHI"]

    # Test Lazy
    df_lazy = pfwf.scan_fwf(path, specs).collect()
    assert df_lazy.shape == (2, 2)
    assert df_lazy["c1"].to_list() == ["123", "ABC"]
    assert df_lazy["c2"].to_list() == ["789", "GHI"]


if __name__ == "__main__":
    test_read()
    test_partial_specs()
