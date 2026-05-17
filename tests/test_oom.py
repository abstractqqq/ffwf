import os
import time

import polars as pl

import polars_fwf as pfwf


def generate_data(path, num_rows):
    os.makedirs("data", exist_ok=True)
    with open(path, "w") as f:
        for i in range(num_rows):
            state = "NY" if i % 1000 == 0 else "CA"
            f.write(f"{i:010d}{'X' * 80}{state}\n")


def test_lazy_large_file_stability():
    # 2M rows = ~186MB file
    specs = [
        pfwf.FieldSpec("id", 0, 10, pfwf.DType.I32),
        pfwf.FieldSpec("data", 10, 80, pfwf.DType.String),
        pfwf.FieldSpec("state", 90, 2, pfwf.DType.String),
    ]

    path = "data/mem_stability.fwf"
    generate_data(path, 2_000_000)

    print("\n--- Large File Stability Test ---")

    start = time.perf_counter()
    lf = pfwf.scan_fwf(path, specs)
    print("Schema:", lf.schema)

    df = lf.filter(pl.col("state") == "NY").collect()
    end = time.perf_counter()

    print(f"Processed 2M rows in {end - start:.4f}s. Kept {len(df)} rows.")
    print("Sample:")
    print(df.head())

    assert (len(df) // 1000) == 2000
    print("Stability test passed!")


if __name__ == "__main__":
    test_lazy_large_file_stability()
