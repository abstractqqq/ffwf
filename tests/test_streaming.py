import os

import polars as pl
import polars.testing as pl_testing

import ffwf as fw
import ffwf.polars as plfw


def test_streaming_various_chunks_pl():
    # 1. Setup data: 10 rows
    path = "data/streaming_test.fwf"
    os.makedirs("data", exist_ok=True)

    # 10 rows, 10 chars each + newline = 11 bytes per line
    rows = [f"{i:05}ABCDE\n".encode("utf-8") for i in range(10)]
    with open(path, "wb") as f:
        f.writelines(rows)

    specs = [
        fw.FieldSpec("id", 0, 5, fw.DType.I32),
        fw.FieldSpec("tag", 5, 5, fw.DType.String),
    ]

    # 2. Reference Read (Eager)
    df_ref = plfw.read_fwf_pl(path, specs)
    assert len(df_ref) == 10

    # 3. Test various chunk sizes (from 1 to 15)
    # This forces different batch boundaries and remainder handling
    for chunk_size in range(1, 15):
        print(f"Testing chunk_size={chunk_size}...")
        df_streaming = plfw.read_fwf_pl(path, specs, chunk_size=chunk_size)

        # Verify result is identical
        pl_testing.assert_frame_equal(df_ref, df_streaming)

        # Verify lazy also works with this chunk size
        df_lazy = plfw.scan_fwf_pl(path, specs, chunk_size=chunk_size).collect()
        pl_testing.assert_frame_equal(df_ref, df_lazy)

    print("Streaming and chunking tests passed!")


if __name__ == "__main__":
    test_streaming_various_chunks_pl()
