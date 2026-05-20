import os

import pyarrow as pa

import ffwf as fw


def test_read_fwf_arrow():
    specs = [
        fw.FieldSpec("id", 0, 5, fw.DType.I32),
        fw.FieldSpec("val", 5, 10, fw.DType.F64),
        fw.FieldSpec("tag", 15, 5, fw.DType.String),
    ]

    path = "data/test_data_arrow.fwf"
    os.makedirs("data", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"00001      1.23 ABC \n00002      4.56 DEF \n00003      7.89 GHI \n")

    table = fw.read_fwf_arrow(path, specs)

    assert isinstance(table, pa.Table)
    assert table.num_rows == 3
    assert table.num_columns == 3
    assert table.column("id").to_pylist() == [1, 2, 3]
    assert table.column("tag").to_pylist() == ["ABC", "DEF", "GHI"]
    print("Arrow test passed!")


def test_read_fwf_arrow_empty():
    specs = [
        fw.FieldSpec("id", 0, 5, "i32"),
    ]
    path = "data/empty.fwf"
    with open(path, "wb") as f:
        pass

    table = fw.read_fwf_arrow(path, specs)
    assert table.num_rows == 0
    assert table.column("id").type == pa.int32()


if __name__ == "__main__":
    test_read_fwf_arrow()
    test_read_fwf_arrow_empty()
