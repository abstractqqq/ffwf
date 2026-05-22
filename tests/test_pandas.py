import os

import pandas as pd

import ffwf as fw


def test_read_fwf_pd():
    specs = [
        fw.FieldSpec("id", 0, 5, fw.DType.I32),
        fw.FieldSpec("val", 5, 10, fw.DType.F64),
        fw.FieldSpec("tag", 15, 5, fw.DType.String),
    ]

    path = "data/test_data_pd.fwf"
    os.makedirs("data", exist_ok=True)
    with open(path, "wb") as f:
        f.write(b"00001      1.23 ABC \n00002      4.56 DEF \n00003      7.89 GHI \n")

    df = fw.read_fwf_pd(path, specs)

    assert isinstance(df, pd.DataFrame)
    assert df.shape == (3, 3)
    assert df["id"].to_list() == [1, 2, 3]
    assert df["tag"].to_list() == ["ABC", "DEF", "GHI"]
    print("Pandas test passed!")


if __name__ == "__main__":
    test_read_fwf_pd()
