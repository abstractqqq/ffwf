import polars as pl
import pytest

import polars_fwf as pfwf


def test_fieldspec_aliases():
    # String aliases
    assert pfwf.FieldSpec("c1", 0, 5, "str").dtype == pfwf.DType.String
    assert pfwf.FieldSpec("c2", 0, 5, "String").dtype == pfwf.DType.String
    assert pfwf.FieldSpec("c3", 0, 5, "int").dtype == pfwf.DType.I32
    assert pfwf.FieldSpec("c4", 0, 5, "Integer").dtype == pfwf.DType.I32
    assert pfwf.FieldSpec("c5", 0, 5, "float").dtype == pfwf.DType.F32
    assert pfwf.FieldSpec("c6", 0, 5, "double").dtype == pfwf.DType.F64
    assert pfwf.FieldSpec("c7", 0, 5, "f32").dtype == pfwf.DType.F32
    assert pfwf.FieldSpec("c8", 0, 5, "f64").dtype == pfwf.DType.F64

    # Original DType enum
    assert pfwf.FieldSpec("c9", 0, 5, pfwf.DType.I16).dtype == pfwf.DType.I16

    # Invalid alias
    with pytest.raises(ValueError, match="Unknown DType alias: invalid"):
        pfwf.FieldSpec("c10", 0, 5, "invalid")


if __name__ == "__main__":
    test_fieldspec_aliases()
