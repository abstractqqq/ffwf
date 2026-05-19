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


if __name__ == "__main__":
    test_read()
