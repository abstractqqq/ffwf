import os
import time

import polars as pl

import ffwf as fw
import ffwf.polars as plfw


def generate_data(path, num_rows):
    os.makedirs("data", exist_ok=True)
    with open(path, "wb") as f:
        for i in range(num_rows):
            state = "NY" if i % 1000 == 0 else "CA"
            f.write(f"{i:010d}{'X' * 80}{state}\n".encode("utf-8"))


def test_lazy_large_file_stability_pl():
    # 2M rows = ~186MB file
    specs = [
        fw.FieldSpec("id", 0, 10, fw.DType.I32),
        fw.FieldSpec("data", 10, 80, fw.DType.String),
        fw.FieldSpec("state", 90, 2, fw.DType.String),
    ]

    path = "data/mem_stability.fwf"
    generate_data(path, 2_000_000)

    print("\n--- Large File Stability Test ---")

    start = time.perf_counter()
    lf = plfw.scan_fwf_pl(path, specs)
    print("Schema:", lf.collect_schema())

    df = lf.filter(pl.col("state") == "NY").collect()
    end = time.perf_counter()

    print(f"Processed 2M rows in {end - start:.4f}s. Kept {len(df)} rows.")
    print("Sample:")
    print(df.head())

    assert len(df) == 2000
    print("Stability test passed!")


if __name__ == "__main__":
    test_lazy_large_file_stability_pl()
