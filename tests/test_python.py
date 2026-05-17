import os


import fwf


def test_read():
    specs = [
        fwf.FieldSpec("id", 0, 5, fwf.DType.I32),
        fwf.FieldSpec("val", 5, 10, fwf.DType.F64),
        fwf.FieldSpec("tag", 15, 5, fwf.DType.String),
    ]

    # Use existing test data
    path = "data/test_data.fwf"
    if not os.path.exists(path):
        os.makedirs("data", exist_ok=True)
        with open(path, "w") as f:
            f.write(
                "00001      1.23 ABC \n00002      4.56 DEF \n00003      7.89 GHI \n"
            )

    df = fwf.read_fwf(path, specs, 21)
    print("DataFrame:")
    print(df)

    assert df.shape == (3, 3)
    assert df["id"][0] == 1
    assert df["tag"][2] == "GHI"
    print("Python test passed!")


if __name__ == "__main__":
    test_read()
