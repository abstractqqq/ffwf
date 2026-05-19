import os

import polars as pl
import polars.testing as pl_testing

import polars_fwf as pfwf


def test_streaming_various_chunks():
    # 1. Setup data: 10 rows
    path = "data/streaming_test.fwf"
    os.makedirs("data", exist_ok=True)

    # 10 rows, 10 chars each + newline = 11 bytes per line
    rows = [f"{i:05}ABCDE\n".encode("utf-8") for i in range(10)]
    with open(path, "wb") as f:
        f.writelines(rows)

    specs = [
        pfwf.FieldSpec("id", 0, 5, pfwf.DType.I32),
        pfwf.FieldSpec("tag", 5, 5, pfwf.DType.String),
    ]

    # 2. Reference Read (Eager)
    df_ref = pfwf.read_fwf(path, specs)
    assert len(df_ref) == 10

    # 3. Test various chunk sizes (from 1 to 15)
    # This forces different batch boundaries and remainder handling
    for chunk_size in range(1, 15):
        print(f"Testing chunk_size={chunk_size}...")
        df_streaming = pfwf.read_fwf(path, specs, chunk_size=chunk_size)

        # Verify result is identical
        pl_testing.assert_frame_equal(df_ref, df_streaming)

        # Verify lazy also works with this chunk size
        df_lazy = pfwf.scan_fwf(path, specs, chunk_size=chunk_size).collect()
        pl_testing.assert_frame_equal(df_ref, df_lazy)

    print("Streaming and chunking tests passed!")


if __name__ == "__main__":
    test_streaming_various_chunks()
