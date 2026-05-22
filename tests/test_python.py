import os

import pytest

import ffwf as fw


def test_read_pl():
    specs = [
        fw.FieldSpec("id", 0, 5, fw.DType.I32),
        fw.FieldSpec("val", 5, 10, fw.DType.F64),
        fw.FieldSpec("tag", 15, 5, fw.DType.String),
    ]

    # Ensure test data is written with LF (\n)
    path = "data/test_data.fwf"
    os.makedirs("data", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"00001      1.23 ABC \n00002      4.56 DEF \n00003      7.89 GHI \n")

    df = fw.read_fwf_pl(path, specs)
    print("DataFrame:")
    print(df)

    assert df.shape == (3, 3)
    assert df["id"][0] == 1
    assert df["tag"][2] == "GHI"
    print("Python test passed!")


def test_partial_specs_pl():
    path = "data/partial_test.fwf"
    os.makedirs("data", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"1234567890\n")
        f.write(b"ABCDEFGHIJ\n")

    # Partial spec: only col 1 (0-3) and col 2 (6-9)
    # Total width = 3 + 3 = 6. Line width = 10. Gaps at 3-6 and 9-10.
    specs = [
        fw.FieldSpec("c1", 0, 3, "str"),
        fw.FieldSpec("c2", 6, 3, "str"),
    ]

    # Test Eager
    df_eager = fw.read_fwf_pl(path, specs)
    assert df_eager.shape == (2, 2)
    assert df_eager["c1"].to_list() == ["123", "ABC"]
    assert df_eager["c2"].to_list() == ["789", "GHI"]

    # Test Lazy
    df_lazy = fw.scan_fwf_pl(path, specs).collect()
    assert df_lazy.shape == (2, 2)
    assert df_lazy["c1"].to_list() == ["123", "ABC"]
    assert df_lazy["c2"].to_list() == ["789", "GHI"]


def test_field_spec_error_strategy():
    # Default strategy
    spec = fw.FieldSpec("col1", 0, 5, "int")
    assert spec.error_strategy == fw.ErrorStrategy.PushNull()
    assert str(spec.error_strategy) == "PushNull"

    # Explicit strategy
    spec2 = fw.FieldSpec(
        "col2", 5, 5, "float", error_strategy=fw.ErrorStrategy.PushNull()
    )
    assert spec2.error_strategy == fw.ErrorStrategy.PushNull()


def test_fill_strategy():
    # Basic fill
    specs = [
        fw.FieldSpec("id", 0, 5, "int"),
        fw.FieldSpec(
            "val", 5, 5, "int", error_strategy=fw.ErrorStrategy.Fill(b"99999")
        ),
    ]

    path = "data/fill_test.fwf"
    os.makedirs("data", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"0000112345\n")
        f.write(b"00002ERROR\n")

    df = fw.read_fwf_pl(path, specs)
    assert df["val"].to_list() == [12345, 99999]

    # Validation
    with pytest.raises(
        ValueError, match=r"Fill value length \(6\) exceeds field length \(5\)"
    ):
        fw.FieldSpec(
            "val", 0, 5, "int", error_strategy=fw.ErrorStrategy.Fill(b"123456")
        )

    with pytest.raises(ValueError, match="Cannot parse fill value"):
        fw.FieldSpec("val", 0, 5, "int", error_strategy=fw.ErrorStrategy.Fill(b"ABC  "))

    # Setter validation
    spec = fw.FieldSpec("val", 0, 5, "int")
    spec.error_strategy = fw.ErrorStrategy.Fill(b"123  ")
    assert bytes(spec.error_strategy._0) == b"123  "

    with pytest.raises(ValueError, match="Fill value length"):
        spec.length = 2

    spec.length = 10
    with pytest.raises(ValueError, match="Cannot parse fill value"):
        spec.dtype = fw.DType.F32
        spec.error_strategy = fw.ErrorStrategy.Fill(b"err")


if __name__ == "__main__":
    test_read_pl()
    test_partial_specs_pl()
