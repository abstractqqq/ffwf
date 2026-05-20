import datetime
import os

import polars as pl
import pytest

import ffwf as fw
import ffwf.polars as plfw


def test_write_fwf_basic_pl(tmp_path):
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
    specs_dict = plfw.write_fwf_pl(df, path)

    assert "id" in specs_dict
    assert "name" in specs_dict
    assert "val" in specs_dict
    assert "active" in specs_dict

    # Read back to verify
    specs = [
        fw.FieldSpec(
            "id", specs_dict["id"]["offset"], specs_dict["id"]["length"], "int"
        ),
        fw.FieldSpec(
            "name", specs_dict["name"]["offset"], specs_dict["name"]["length"], "str"
        ),
        fw.FieldSpec(
            "val", specs_dict["val"]["offset"], specs_dict["val"]["length"], "f64"
        ),
        fw.FieldSpec(
            "active",
            specs_dict["active"]["offset"],
            specs_dict["active"]["length"],
            "str",
        ),
    ]

    df_read = plfw.read_fwf_pl(path, specs)
    assert df_read.shape == (3, 4)
    assert df_read["id"].to_list() == [1, 2, 100]
    assert [s.strip() for s in df_read["name"]] == ["Alice", "Bob", "Charlie"]


def test_write_fwf_specs_pl(tmp_path):
    path = str(tmp_path / "test_spec.fwf")
    df = pl.DataFrame({"a": [1, 10], "b": ["x", "yz"]})

    specs = [fw.FieldSpec("a", 0, 5, "int"), fw.FieldSpec("b", 5, 5, "str")]

    plfw.write_fwf_pl(df, path, specs=specs)

    with open(path, "rb") as f:
        lines = f.readlines()
        assert lines[0] == b"    1x    \n"
        assert lines[1] == b"   10yz   \n"


def test_write_fwf_validation_pl(tmp_path):
    path = str(tmp_path / "fail.fwf")
    df = pl.DataFrame({"a": [1000]})
    specs = [fw.FieldSpec("a", 0, 2, "int")]  # 1000 needs 4 chars

    with pytest.raises(ValueError, match="has data longer"):
        plfw.write_fwf_pl(df, path, specs=specs)


def test_write_fwf_contiguity_pl():
    df = pl.DataFrame({"a": [1], "b": [2]})
    specs = [
        fw.FieldSpec("a", 0, 5, "int"),
        fw.FieldSpec("b", 10, 5, "int"),  # Gap between 5 and 10
    ]
    with pytest.raises(ValueError, match="not contiguous"):
        plfw.write_fwf_pl(df, "dummy", specs=specs)


def test_write_fwf_unsupported_type_pl():
    df = pl.DataFrame({"a": [datetime.date(2023, 1, 1)]})
    with pytest.raises(TypeError, match="Unsupported column type"):
        plfw.write_fwf_pl(df, "dummy")


def test_write_fwf_bool_treatment_pl(tmp_path):
    path = str(tmp_path / "bool.fwf")
    df = pl.DataFrame({"a": [True, False, None]})
    plfw.write_fwf_pl(df, path, bool_treatment=("YES", "NO ", "---"))

    with open(path, "rb") as f:
        lines = f.readlines()
        assert lines[0] == b"YES\n"
        assert lines[1] == b"NO \n"
        assert lines[2] == b"---\n"


def test_sink_fwf_pl(tmp_path):
    path = str(tmp_path / "sink.fwf")
    lf = pl.DataFrame({"a": [1, 2]}).lazy()
    specs = [fw.FieldSpec("a", 0, 5, "int")]
    plfw.sink_fwf_pl(lf, path, specs=specs)
    with open(path, "rb") as f:
        assert f.readline() == b"    1\n"


def test_sink_fwf_batch_validation_pl(tmp_path):
    path = str(tmp_path / "sink_fail.fwf")

    df = pl.DataFrame({"a": [1, 2, 1000, 4]})
    lf = df.lazy()

    # Spec only allows 2 chars, but 1000 needs 4.
    specs = [fw.FieldSpec("a", 0, 2, "int")]

    with pytest.raises(
        ValueError,
        match=r"failed validation: Column 'a' has data longer \(4\) than specified length \(2\)",
    ):
        plfw.sink_fwf_pl(lf, path, specs=specs)


def test_write_fwf_large_floats_pl(tmp_path):
    path = str(tmp_path / "large_floats.fwf")
    # Huge float
    df = pl.DataFrame({"val": [1.23456789e300, 1.23456789e-10]})

    # Test with 2 decimals
    specs_dict = plfw.write_fwf_pl(df, path, decimals=2)

    with open(path, "rb") as f:
        lines = f.readlines()
        # Polars truncate(2) should result in 1.23e300
        assert b"1.23" in lines[0]
        assert b"0.0" in lines[1]  # 1.2e-10 truncated to 2 decimals is 0.0


def test_write_fwf_float_width_validation_pl(tmp_path):
    path = str(tmp_path / "width_fail.fwf")
    # Large float that exceeds width 5 even after truncation
    df = pl.DataFrame({"val": [1.23456789e10]})
    specs = [fw.FieldSpec("val", 0, 5, "f64")]

    with pytest.raises(ValueError, match="has data longer"):
        plfw.write_fwf_pl(df, path, specs=specs, decimals=2)


def test_write_fwf_nan_inf_pl(tmp_path):
    path = str(tmp_path / "nan_inf.fwf")
    df = pl.DataFrame({"val": [float("nan"), float("inf")]})
    specs = [fw.FieldSpec("val", 0, 10, "f64")]
    plfw.write_fwf_pl(df, path, specs=specs)

    with open(path, "rb") as f:
        lines = f.readlines()
        assert b"NaN" in lines[0]
        assert b"inf" in lines[1]


def test_write_fwf_truncation_logic_pl(tmp_path):
    path = str(tmp_path / "trunc.fwf")
    # 1.999 rounded to 1 decimal should be 2.0
    df = pl.DataFrame({"val": [1.999]})
    specs = [fw.FieldSpec("val", 0, 5, "f64")]
    plfw.write_fwf_pl(df, path, specs=specs, decimals=1)

    with open(path, "rb") as f:
        line = f.read().rstrip(b"\n")
        assert line == b"  2.0"


def test_write_fwf_truncation_zero_decimals_pl(tmp_path):
    path = str(tmp_path / "trunc0.fwf")
    df = pl.DataFrame({"val": [1.99, -1.99]})

    specs = [fw.FieldSpec("val", 0, 10, "f64")]
    plfw.write_fwf_pl(df, path, specs=specs, decimals=0)

    with open(path, "rb") as f:
        lines = f.readlines()
        # Rounding 1.99 to 0 decimals should be 2.0 or 2
        assert b"2.0" in lines[0] or b"2" in lines[0]
        assert b"-2.0" in lines[1] or b"-2" in lines[1]
        assert (
            b"1" not in lines[0] or b"1.0" in lines[0]
        )  # Avoid false positives with 1.0 being in 2.0 if not careful, but here we just check logic.


def test_write_fwf_f32_bounds_pl(tmp_path):
    path = str(tmp_path / "f32_bounds.fwf")
    # Value that fits in f64 but not f32 (approx 1e38)
    df = pl.DataFrame({"val": [1e40]})

    # Spec says f32, but we only validate width now
    specs = [fw.FieldSpec("val", 0, 20, "f32")]
    plfw.write_fwf_pl(df, path, specs=specs)

    with open(path, "rb") as f:
        line = f.read().strip()
        # Polars might format as 1e+40 or 1e40 depending on version/context
        assert b"1e" in line.lower() and b"40" in line


def test_write_fwf_negatives_pl(tmp_path):
    path = str(tmp_path / "negatives.fwf")
    df = pl.DataFrame({"i": [-1, -100], "f": [-1.23456, -0.0001]})

    # Specs with enough width for signs
    specs = [fw.FieldSpec("i", 0, 5, "int"), fw.FieldSpec("f", 5, 8, "float")]

    plfw.write_fwf_pl(df, path, specs=specs, decimals=2)

    with open(path, "rb") as f:
        lines = f.readlines()
        # "-1.23456" truncated to 2 decimals -> "-1.23" (len 5)
        # "-1" padded to width 5 -> "   -1"
        assert lines[0] == b"   -1   -1.23\n"
        assert b"-100" in lines[1]
        assert b"-0.0" in lines[1]


if __name__ == "__main__":
    pytest.main([__file__])
