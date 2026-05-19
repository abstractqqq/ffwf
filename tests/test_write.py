import datetime
import os

import polars as pl
import pytest

import polars_fwf as pfwf


def test_write_fwf_basic(tmp_path):
    path = str(tmp_path / "test.fwf")
    df = pl.DataFrame(
        {
            "id": [1, 2, 100],
            "name": ["Alice", "Bob", "Charlie"],
            "val": [1.1, 2.22, 3.333],
            "active": [True, False, None],
        }
    )

    # Test inference
    specs_dict = pfwf.write_fwf(df, path)

    assert "id" in specs_dict
    assert "name" in specs_dict
    assert "val" in specs_dict
    assert "active" in specs_dict

    # Read back to verify
    specs = [
        pfwf.FieldSpec(
            "id", specs_dict["id"]["offset"], specs_dict["id"]["length"], "int"
        ),
        pfwf.FieldSpec(
            "name", specs_dict["name"]["offset"], specs_dict["name"]["length"], "str"
        ),
        pfwf.FieldSpec(
            "val", specs_dict["val"]["offset"], specs_dict["val"]["length"], "f64"
        ),
        pfwf.FieldSpec(
            "active",
            specs_dict["active"]["offset"],
            specs_dict["active"]["length"],
            "str",
        ),
    ]

    df_read = pfwf.read_fwf(path, specs)
    assert df_read.shape == (3, 4)
    assert df_read["id"].to_list() == [1, 2, 100]
    assert [s.strip() for s in df_read["name"]] == ["Alice", "Bob", "Charlie"]


def test_write_fwf_specs(tmp_path):
    path = str(tmp_path / "test_spec.fwf")
    df = pl.DataFrame({"a": [1, 10], "b": ["x", "yz"]})

    specs = [pfwf.FieldSpec("a", 0, 5, "int"), pfwf.FieldSpec("b", 5, 5, "str")]

    pfwf.write_fwf(df, path, specs=specs)

    with open(path, "r") as f:
        lines = f.readlines()
        assert lines[0] == "    1x    \n"
        assert lines[1] == "   10yz   \n"


def test_write_fwf_validation(tmp_path):
    path = str(tmp_path / "fail.fwf")
    df = pl.DataFrame({"a": [1000]})
    specs = [pfwf.FieldSpec("a", 0, 2, "int")]  # 1000 needs 4 chars

    with pytest.raises(ValueError, match="has data longer"):
        pfwf.write_fwf(df, path, specs=specs)


def test_write_fwf_contiguity():
    df = pl.DataFrame({"a": [1], "b": [2]})
    specs = [
        pfwf.FieldSpec("a", 0, 5, "int"),
        pfwf.FieldSpec("b", 10, 5, "int"),  # Gap between 5 and 10
    ]
    with pytest.raises(ValueError, match="not contiguous"):
        pfwf.write_fwf(df, "dummy", specs=specs)


def test_write_fwf_unsupported_type():
    df = pl.DataFrame({"a": [datetime.date(2023, 1, 1)]})
    with pytest.raises(TypeError, match="Unsupported column type"):
        pfwf.write_fwf(df, "dummy")


def test_write_fwf_bool_treatment(tmp_path):
    path = str(tmp_path / "bool.fwf")
    df = pl.DataFrame({"a": [True, False, None]})
    pfwf.write_fwf(df, path, bool_treatment=("YES", "NO ", "---"))

    with open(path, "r") as f:
        lines = f.readlines()
        assert lines[0] == "YES\n"
        assert lines[1] == "NO \n"
        assert lines[2] == "---\n"


def test_sink_fwf(tmp_path):
    path = str(tmp_path / "sink.fwf")
    lf = pl.DataFrame({"a": [1, 2]}).lazy()
    specs = [pfwf.FieldSpec("a", 0, 5, "int")]
    pfwf.sink_fwf(lf, path, specs=specs)
    with open(path, "r") as f:
        assert f.readline() == "    1\n"


def test_sink_fwf_batch_validation(tmp_path):
    path = str(tmp_path / "sink_fail.fwf")

    df = pl.DataFrame({"a": [1, 2, 1000, 4]})
    lf = df.lazy()

    # Spec only allows 2 chars, but 1000 needs 4.
    specs = [pfwf.FieldSpec("a", 0, 2, "int")]

    with pytest.raises(
        ValueError,
        match=r"failed validation: Column 'a' has data longer \(4\) than specified length \(2\)",
    ):
        pfwf.sink_fwf(lf, path, specs=specs)


def test_write_fwf_large_floats(tmp_path):
    path = str(tmp_path / "large_floats.fwf")
    # Huge float that would overflow (val * 10^n) if using numeric truncation
    df = pl.DataFrame({"val": [1.23456789e300, 1.23456789e-10]})

    # Test with 2 decimals
    specs_dict = pfwf.write_fwf(df, path, max_decimals=2)

    with open(path, "r") as f:
        lines = f.readlines()
        # Polars string representation for 1e300 usually has 'e300'
        # Our regex truncation should preserve the 'e300' part if it matched
        # Let's check what actually happened
        assert "1.23" in lines[0]
        assert "0.0" in lines[1]  # 1.2e-10 truncated to 2 decimals is 0.0


def test_write_fwf_negatives(tmp_path):
    path = str(tmp_path / "negatives.fwf")
    df = pl.DataFrame({"i": [-1, -100], "f": [-1.23456, -0.0001]})

    # Specs with enough width for signs
    specs = [pfwf.FieldSpec("i", 0, 5, "int"), pfwf.FieldSpec("f", 5, 8, "float")]

    pfwf.write_fwf(df, path, specs=specs, max_decimals=2)

    with open(path, "r") as f:
        lines = f.readlines()
        # "-1.23456" truncated to 2 decimals -> "-1.23" (len 5)
        # "-1" padded to width 5 -> "   -1"
        assert lines[0] == "   -1   -1.23\n"
        assert "-100" in lines[1]
        assert "-0.0" in lines[1]


if __name__ == "__main__":
    pytest.main([__file__])
