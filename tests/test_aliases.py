import pytest

import ffwf as fw


def test_fieldspec_aliases_pl():
    # String aliases
    assert fw.FieldSpec("c1", 0, 5, "str").dtype == fw.DType.String
    assert fw.FieldSpec("c2", 0, 5, "String").dtype == fw.DType.String
    assert fw.FieldSpec("c3", 0, 5, "int").dtype == fw.DType.I32
    assert fw.FieldSpec("c4", 0, 5, "Integer").dtype == fw.DType.I32
    assert fw.FieldSpec("c5", 0, 5, "float").dtype == fw.DType.F32
    assert fw.FieldSpec("c6", 0, 5, "double").dtype == fw.DType.F64
    assert fw.FieldSpec("c7", 0, 5, "f32").dtype == fw.DType.F32
    assert fw.FieldSpec("c8", 0, 5, "f64").dtype == fw.DType.F64

    # Original DType enum
    assert fw.FieldSpec("c9", 0, 5, fw.DType.I16).dtype == fw.DType.I16

    # Invalid alias
    with pytest.raises(ValueError, match="Unknown DType alias: invalid"):
        fw.FieldSpec("c10", 0, 5, "invalid")


if __name__ == "__main__":
    test_fieldspec_aliases_pl()
