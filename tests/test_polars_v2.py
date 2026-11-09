import polars as pl
import polars.testing as pl_testing
import pytest

import ffwf as fw


def test_polars_v2_capsule_returns_dataframe(tmp_path):
    """Ensure read_fwf_pl and scan_fwf_pl return DataFrames (not Struct Series in Polars v2)."""
    path = str(tmp_path / "single_and_multi.fwf")
    with open(path, "wb") as f:
        f.write(b"00001hello\n00002world\n")

    # Single-column spec (regression test for Polars v2 struct[1] Series)
    single_spec = [fw.FieldSpec("id", 0, 5, "i32")]
    df_single = fw.read_fwf_pl(path, single_spec)
    assert isinstance(df_single, pl.DataFrame)
    assert df_single.shape == (2, 1)
    assert df_single["id"].to_list() == [1, 2]

    df_single_lazy = fw.scan_fwf_pl(path, single_spec).collect()
    assert isinstance(df_single_lazy, pl.DataFrame)
    pl_testing.assert_frame_equal(df_single, df_single_lazy)


def test_scan_projection_and_predicate_pushdown(tmp_path):
    """Verify projection pushdown, reordered columns, and filter-only columns."""
    path = str(tmp_path / "pushdown.fwf")
    with open(path, "wb") as f:
        f.write(b"00001   10.50ALPHA  NY\n")
        f.write(b"00002   20.00BETA   CA\n")
        f.write(b"00003   30.25GAMMA  NY\n")
        f.write(b"00004    5.00DELTA  NY\n")

    specs = [
        fw.FieldSpec("id", 0, 5, "i32"),
        fw.FieldSpec("val", 5, 8, "f64"),
        fw.FieldSpec("name", 13, 7, "str"),
        fw.FieldSpec("state", 20, 2, "str"),
    ]

    # 1. Project subset of columns in reverse order, filter on a column NOT in select
    lf = fw.scan_fwf_pl(path, specs)
    res = (
        lf.filter((pl.col("state") == "NY") & (pl.col("val") > 10.0))
        .select(["name", "id"])
        .collect()
    )
    assert res.columns == ["name", "id"]
    assert res["name"].to_list() == ["ALPHA", "GAMMA"]
    assert res["id"].to_list() == [1, 3]

    # 2. Count / pl.len() query (minimal projection)
    count_df = lf.select(pl.len()).collect()
    assert count_df.item() == 4

    # 3. Compare streaming vs in-memory engine explicitly
    res_streaming = (
        lf.filter(pl.col("state") == "NY")
        .select(["id", "val"])
        .collect(engine="streaming")
    )
    res_in_memory = (
        lf.filter(pl.col("state") == "NY")
        .select(["id", "val"])
        .collect(engine="in-memory")
    )
    pl_testing.assert_frame_equal(res_streaming, res_in_memory)


def test_scan_n_rows_pushdown(tmp_path):
    """Verify head/limit pushdown with and without predicates."""
    path = str(tmp_path / "limit.fwf")
    with open(path, "wb") as f:
        for i in range(100):
            tag = "EVEN" if i % 2 == 0 else "ODD "
            f.write(f"{i:05d}{tag}\n".encode("utf-8"))

    specs = [
        fw.FieldSpec("id", 0, 5, "i32"),
        fw.FieldSpec("tag", 5, 4, "str"),
    ]

    # Pre-filter head(5)
    df_head = fw.scan_fwf_pl(path, specs, chunk_size=10).head(5).collect()
    assert len(df_head) == 5
    assert df_head["id"].to_list() == [0, 1, 2, 3, 4]

    # Post-filter head(5)
    df_filtered_head = (
        fw.scan_fwf_pl(path, specs, chunk_size=10)
        .filter(pl.col("tag") == "ODD")
        .head(5)
        .collect()
    )
    assert len(df_filtered_head) == 5
    assert df_filtered_head["id"].to_list() == [1, 3, 5, 7, 9]


def test_short_and_long_strings_utf8_and_errors(tmp_path):
    """Test inline (<=12B) and long (>12B) strings with ASCII, UTF-8, nulls, and fills."""
    path = str(tmp_path / "strings.fwf")
    # short_col: 10 bytes (<= 12 bytes, uses inline u128 view path)
    # long_col: 20 bytes (> 12 bytes, uses variadic buffer path)
    row1 = "café".encode("utf-8").ljust(10, b" ") + b"ABCDEFGHIJKLMNOPQRST\n"
    row2 = b"\xff\xfe        " + b"\xff\xfe                  \n"
    row3 = b"____padded" + b"____long_padded_____\n"

    with open(path, "wb") as f:
        f.write(row1 + row2 + row3)

    # 1. With PushNull
    specs_null = [
        fw.FieldSpec("short", 0, 10, "str"),
        fw.FieldSpec("long", 10, 20, "str"),
    ]
    df_null = fw.read_fwf_pl(path, specs_null)
    assert df_null["short"].to_list() == ["café", None, "____padded"]
    assert df_null["long"].to_list() == [
        "ABCDEFGHIJKLMNOPQRST",
        None,
        "____long_padded_____",
    ]

    # 2. With Fill strategy and custom padding
    specs_fill = [
        fw.FieldSpec(
            "short",
            0,
            10,
            "str",
            padding=ord("_"),
            error_strategy=fw.ErrorStrategy.Fill(b"FALLBACK"),
        ),
        fw.FieldSpec(
            "long",
            10,
            20,
            "str",
            padding=ord("_"),
            error_strategy=fw.ErrorStrategy.Fill(b"LONG_FALLBACK_VALUE"),
        ),
    ]
    df_fill = fw.read_fwf_pl(path, specs_fill)
    assert df_fill["short"].to_list() == ["café     ", "FALLBACK", "padded"]
    assert df_fill["long"].to_list() == [
        "ABCDEFGHIJKLMNOPQRST",
        "LONG_FALLBACK_VALUE",
        "long_padded",
    ]


def test_custom_padding_primitives(tmp_path):
    """Test custom padding byte for primitive integer and float columns."""
    path = str(tmp_path / "custom_pad.fwf")
    with open(path, "wb") as f:
        f.write(b"00042***3.14***\n")
        f.write(b"00BAD***BAD****\n")

    specs = [
        fw.FieldSpec(
            "num",
            0,
            5,
            "i32",
            padding=ord("0"),
            error_strategy=fw.ErrorStrategy.Fill(b"7"),
        ),
        fw.FieldSpec(
            "flt",
            5,
            10,
            "f64",
            padding=ord("*"),
            error_strategy=fw.ErrorStrategy.PushNull(),
        ),
    ]
    df = fw.read_fwf_pl(path, specs)
    assert df["num"].to_list() == [42, 7]
    assert df["flt"].to_list() == [3.14, None]


def test_empty_file_pl(tmp_path):
    """Ensure empty files return an empty DataFrame with the expected schema."""
    path = str(tmp_path / "empty.fwf")
    with open(path, "wb"):
        pass

    specs = [
        fw.FieldSpec("id", 0, 5, "i32"),
        fw.FieldSpec("name", 5, 10, "str"),
    ]

    df_eager = fw.read_fwf_pl(path, specs)
    assert df_eager.shape == (0, 2)
    assert df_eager.schema == {"id": pl.Int32, "name": pl.String}

    df_lazy = fw.scan_fwf_pl(path, specs).collect()
    assert df_lazy.shape == (0, 2)
    assert df_lazy.schema == {"id": pl.Int32, "name": pl.String}


def test_bounds_validation_pl(tmp_path):
    """Ensure both read_fwf_pl and scan_fwf_pl validate FieldSpec bounds."""
    path = str(tmp_path / "short_line.fwf")
    with open(path, "wb") as f:
        f.write(b"12345\n")

    bad_specs = [fw.FieldSpec("id", 0, 10, "i32")]

    with pytest.raises(ValueError, match="exceeds data length"):
        fw.read_fwf_pl(path, bad_specs)

    with pytest.raises(ValueError, match="exceeds data length"):
        fw.scan_fwf_pl(path, bad_specs)


def test_write_and_sink_quotes_categorical_and_nulls(tmp_path):
    """Test quote stripping, Categorical/Enum columns, and all-null columns in writers."""
    path_write = str(tmp_path / "write_extra.fwf")
    path_sink = str(tmp_path / "sink_extra.fwf")

    df = pl.DataFrame(
        {
            "text": ['a"b', "c'd", None],
            "cat": pl.Series(["x'1", 'y"2', "z"], dtype=pl.Categorical),
            "all_null": pl.Series([None, None, None], dtype=pl.Int32),
        }
    )

    specs_w = fw.write_fwf_pl(df, path_write)
    specs_s = fw.sink_fwf_pl(df.lazy(), path_sink)
    assert specs_w == specs_s

    with open(path_write, "rb") as f1, open(path_sink, "rb") as f2:
        content_w = f1.read()
        content_s = f2.read()
        assert content_w == content_s
        assert b'"' not in content_w
        assert b"'" not in content_w
