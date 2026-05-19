import polars as pl
import pytest

import polars_fwf as pfwf


def test_fieldspec_width_validation():
    # Negative width
    with pytest.raises(ValueError, match="width must be positive"):
        pfwf.FieldSpec("a", 0, -1, "int")

    # Zero width
    with pytest.raises(ValueError, match="width must be positive"):
        pfwf.FieldSpec("a", 0, 0, "int")


def test_fieldspec_integer_capacity():
    # I8 max 4 chars (-128)
    pfwf.FieldSpec("a", 0, 4, "i8")  # OK
    with pytest.warns(UserWarning, match="exceeds maximum capacity for I8"):
        pfwf.FieldSpec("a", 0, 5, "i8")

    # U8 max 3 chars (255)
    pfwf.FieldSpec("a", 0, 3, "u8")  # OK
    with pytest.warns(UserWarning, match="exceeds maximum capacity for U8"):
        pfwf.FieldSpec("a", 0, 4, "u8")

    # I16 max 6 chars (-32768)
    pfwf.FieldSpec("a", 0, 6, "i16")  # OK
    with pytest.warns(UserWarning, match="exceeds maximum capacity for I16"):
        pfwf.FieldSpec("a", 0, 7, "i16")

    # I32 max 11 chars
    pfwf.FieldSpec("a", 0, 11, "i32")  # OK
    pfwf.FieldSpec("a", 0, 11, "int")  # OK
    pfwf.FieldSpec("a", 0, 11, "integer")  # OK
    with pytest.warns(UserWarning, match="exceeds maximum capacity for I32"):
        pfwf.FieldSpec("a", 0, 12, "i32")
    with pytest.warns(UserWarning, match="exceeds maximum capacity for I32"):
        pfwf.FieldSpec("a", 0, 12, "int")
    with pytest.warns(UserWarning, match="exceeds maximum capacity for I32"):
        pfwf.FieldSpec("a", 0, 12, "integer")


def test_write_capacity_warning(tmp_path):
    path = str(tmp_path / "warn.fwf")
    df = pl.DataFrame({"a": [1]})

    # Spec width 12 for I32 should warn
    specs = [pfwf.FieldSpec("a", 0, 12, "i32")]

    with pytest.warns(UserWarning, match="exceeds maximum capacity"):
        pfwf.write_fwf(df, path, specs=specs)

    # Inference that results in large width should also warn
    # I64 max is 20, let's make it 21
    df_large = pl.DataFrame({"a": [10**20]})  # This will trigger width >= 21
    with pytest.warns(UserWarning, match="exceeds maximum capacity"):
        pfwf.write_fwf(df_large, path)


if __name__ == "__main__":
    pytest.main([__file__])
