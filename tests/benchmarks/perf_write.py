import os
import time

import numpy as np
import polars as pl

import ffwf as fw
import ffwf.polars as plfw


def benchmark_write_pl():
    num_rows = 100_000
    num_cols = 50
    data = {f"col_{i}": np.random.rand(num_rows) for i in range(num_cols)}
    df = pl.DataFrame(data)

    path = "data/perf_test.fwf"
    os.makedirs("data", exist_ok=True)

    print(f"Benchmarking write_fwf with {num_rows} rows and {num_cols} columns...")

    # Test with inference (skip_width_check=True)
    start = time.perf_counter()
    plfw.write_fwf_pl(df, path)
    duration_infer = time.perf_counter() - start
    print(f"Write with inference: {duration_infer:.4f}s")

    # Test with provided specs (validation ON)
    specs = [fw.FieldSpec(f"col_{i}", i * 20, 20, "f64") for i in range(num_cols)]
    start = time.perf_counter()
    plfw.write_fwf_pl(df, path, specs=specs)
    duration_specs = time.perf_counter() - start
    print(f"Write with specs (validation ON): {duration_specs:.4f}s")

    # Test sink_fwf (streaming)
    start = time.perf_counter()
    plfw.sink_fwf_pl(df.lazy(), path, specs=specs)
    duration_sink = time.perf_counter() - start
    print(f"Sink FWF (streaming): {duration_sink:.4f}s")


if __name__ == "__main__":
    benchmark_write_pl()
