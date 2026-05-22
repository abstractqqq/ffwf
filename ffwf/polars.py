from __future__ import annotations

from typing import TYPE_CHECKING, Any, Iterator, Sequence

# Some functionalities work without requiring polars >= 1.34.
import polars as pl
import polars.selectors as cs
from polars.io.plugins import register_io_source

from . import ArrowCapsule, DType, FieldSpec, FwfParser, FwfReader, PyFieldSpec

__all__ = [
    "read_fwf_pl",
    "scan_fwf_pl",
    "write_fwf_pl",
    "sink_fwf_pl",
]

# ==============================================================================
# Polars IO Source (Lazy)
# ==============================================================================


class FwfSource:
    """
    Polars IO Source implementation for Fixed-Width Files (FWF).
    """

    def __init__(
        self,
        path: str,
        specs: Sequence[PyFieldSpec],
        line_length: int,
        chunk_size: int | None,
        parallel: bool = True,
    ):
        """
        Initialize the FWF source.

        Parameters
        ----------
        path : str
            Path to the FWF file.
        specs : Sequence[PyFieldSpec]
            List of field specifications.
        line_length : int
            The total length of each line in bytes.
        chunk_size : int | None
            The number of rows to parse per batch. If None, it's inferred.
        parallel : bool, default True
            Whether to use multi-threaded parsing in Rust.
        """
        self.path = path
        self.specs = specs
        self.line_length = line_length
        self.chunk_size = chunk_size
        self.parallel = parallel

    def __call__(
        self,
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        """
        Execute the IO source and yield DataFrames.

        Parameters
        ----------
        with_columns : list[str] | None
            List of columns to project.
        predicate : pl.Expr | None
            Optional filter expression.
        n_rows : int | None
            Optional row limit.
        batch_size : int | None
            Optional override for the chunk size.

        Yields
        ------
        pl.DataFrame
            A Polars DataFrame containing the parsed batch.
        """
        reader = FwfReader(
            self.path,
            list(self.specs),
            self.line_length,
            parallel=self.parallel,
            chunk_size=batch_size or self.chunk_size,
        )

        count = 0
        while True:
            capsule_tuples = reader.next_burst()
            if not capsule_tuples:
                break

            for capsules in capsule_tuples:
                df = pl.from_arrow(ArrowCapsule(capsules))

                if with_columns:
                    df = df.select(with_columns)

                if predicate is not None:
                    df = df.filter(predicate)

                if n_rows is not None:
                    remaining = n_rows - count
                    if remaining <= 0:
                        return
                    if len(df) > remaining:
                        df = df.head(remaining)

                if len(df) > 0:
                    count += len(df)
                    yield df

                if n_rows is not None and count >= n_rows:
                    return


# ==============================================================================
# Reading (Eager & Lazy)
# ==============================================================================


def _dtype_to_pl(dtype: DType) -> pl.DataType:
    """
    Convert an internal DType to a Polars DataType.
    """
    if dtype == DType.I8:
        return pl.Int8
    if dtype == DType.I16:
        return pl.Int16
    if dtype == DType.I32:
        return pl.Int32
    if dtype == DType.I64:
        return pl.Int64
    if dtype == DType.U8:
        return pl.UInt8
    if dtype == DType.U16:
        return pl.UInt16
    if dtype == DType.U32:
        return pl.UInt32
    if dtype == DType.U64:
        return pl.UInt64
    if dtype == DType.F32:
        return pl.Float32
    if dtype == DType.F64:
        return pl.Float64
    if dtype == DType.String:
        return pl.String
    raise ValueError(f"Unknown DType: {dtype}")


def read_fwf_pl(
    path: str,
    specs: Sequence[PyFieldSpec],
    line_length: int | None = None,
    newline: str | bytes = "\n",
    chunk_size: int | None = None,
    parallel: bool = True,
) -> pl.DataFrame:
    """
    Read a fixed-width file into a Polars DataFrame.

    Parameters
    ----------
    path : str
        Path to the FWF file.
    specs : Sequence[PyFieldSpec]
        List of field specifications defining column names, offsets, lengths, and types.
    line_length : int | None, optional
        The total length of each line in bytes (including newline). If None, it is
        automatically detected.
    newline : str | bytes, default "\\n"
        The newline character(s) used in the file.
    chunk_size : int | None, optional
        The number of rows to parse per batch. If None, it's inferred by the core parser.
    parallel : bool, default True
        Whether to use multi-threaded parsing in the Rust core.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the parsed data.

    Examples
    --------
    >>> import ffwf as fw
    >>> specs = [fw.FieldSpec("id", 0, 5, "int"), fw.FieldSpec("val", 5, 10, "f64")]
    >>> df = fw.read_fwf_pl("data.fwf", specs)
    """
    newline_bytes = newline if isinstance(newline, bytes) else newline.encode("utf-8")
    stride, data_len = FwfParser.detect_line_length(path, newline_bytes)

    actual_stride = line_length if line_length is not None else stride
    actual_data_len = actual_stride - len(newline_bytes)

    # Validate that all specs are within bounds
    for s in specs:
        if s.offset + s.length > actual_data_len:
            raise ValueError(
                f"FieldSpec '{s.name}' (offset={s.offset}, length={s.length}) "
                f"exceeds data length ({actual_data_len})."
            )

    # EAGER PATH: Use the optimized parallel parser for maximum throughput
    parser = FwfParser(
        list(specs),
        actual_stride,
        parallel=parallel,
        chunk_size=chunk_size,
    )

    # _parse_path handles the mmap and multi-threading internally in one go
    capsule_tuples = parser._parse_path(path)

    if not capsule_tuples:
        # Return empty frame with correct schema
        return pl.DataFrame(schema={s.name: _dtype_to_pl(s.dtype) for s in specs})

    dfs = [pl.from_arrow(ArrowCapsule(c)) for c in capsule_tuples]
    return pl.concat(dfs, how="vertical")


def scan_fwf_pl(
    path: str,
    specs: Sequence[PyFieldSpec],
    line_length: int | None = None,
    newline: str | bytes = "\n",
    chunk_size: int | None = None,
    parallel: bool = True,
) -> pl.LazyFrame:
    """
    Lazily scan a fixed-width file into a Polars LazyFrame.

    Parameters
    ----------
    path : str
        Path to the FWF file.
    specs : Sequence[PyFieldSpec]
        List of field specifications defining column names, offsets, lengths, and types.
    line_length : int | None, optional
        The total length of each line in bytes (including newline). If None, it is
        automatically detected.
    newline : str | bytes, default "\\n"
        The newline character(s) used in the file.
    chunk_size : int | None, optional
        The number of rows to parse per batch. If None, it's inferred by the core parser.
    parallel : bool, default True
        Whether to use multi-threaded parsing in the Rust core.

    Returns
    -------
    pl.LazyFrame
        A Polars LazyFrame representing the scan.

    Examples
    --------
    >>> import ffwf as fw
    >>> specs = [fw.FieldSpec("id", 0, 5, "int")]
    >>> lf = fw.scan_fwf_pl("data.fwf", specs)
    >>> df = lf.filter(pl.col("id") > 100).collect()
    """
    newline_bytes = newline if isinstance(newline, bytes) else newline.encode("utf-8")

    if line_length is None:
        line_length, _ = FwfParser.detect_line_length(path, newline_bytes)

    schema = pl.Schema({s.name: _dtype_to_pl(s.dtype) for s in specs})

    return register_io_source(
        io_source=FwfSource(path, specs, line_length, chunk_size, parallel),
        schema=schema,
    )


# ==============================================================================
# Writing (Eager & Streaming)
# ==============================================================================


def _check_supported_types(df: pl.DataFrame | pl.LazyFrame):
    """
    Verify that all columns in the input have types supported by the FWF writer.
    """
    supported = (
        cs.boolean()
        | cs.integer()
        | cs.float()
        | cs.string()
        | cs.categorical()
        | cs.enum()
    )
    unsupported_schema = df.select(~supported).collect_schema()
    if len(unsupported_schema) > 0:
        raise TypeError(f"Unsupported column type(s) for FWF: {unsupported_schema}")


def _check_specs_contiguity(specs: Sequence[PyFieldSpec]):
    """
    Ensure the provided field specifications are contiguous and start at offset 0.
    """
    if not specs:
        raise ValueError("Specs cannot be empty if provided")
    if specs[0].offset != 0:
        raise ValueError("First FieldSpec offset must be 0")
    for i in range(1, len(specs)):
        prev = specs[i - 1]
        curr = specs[i]
        if prev.offset + prev.length != curr.offset:
            raise ValueError(
                f"Specs are not contiguous between {prev.name} and {curr.name}"
            )


def _check_specs_capacity(specs: Sequence[PyFieldSpec]):
    """
    Check if any FieldSpec width exceeds the maximum capacity of its type and warn.
    """
    import warnings

    for s in specs:
        max_w = s.dtype.max_width()
        if max_w is not None and s.length > max_w:
            warnings.warn(
                f"Column '{s.name}' with type {s.dtype} has width {s.length} "
                f"which exceeds maximum capacity ({max_w} characters).",
                UserWarning,
                stacklevel=3,
            )


def _validate_bool_treatment(bool_treatment: Any) -> tuple[str, str, str]:
    """
    Validate and normalize the boolean mapping collection.
    """
    try:
        if len(bool_treatment) != 3:
            raise ValueError()
        res = (str(bool_treatment[0]), str(bool_treatment[1]), str(bool_treatment[2]))
        return res
    except (TypeError, ValueError, IndexError):
        raise ValueError(
            f"bool_treatment must be an indexable collection of 3 strings, got {bool_treatment}"
        )


def _infer_specs(
    df_or_lf: pl.DataFrame | pl.LazyFrame,
    bool_treatment: tuple[str, str, str],
    decimals: int,
    infer_specs_rows: int | None = None,
) -> Sequence[PyFieldSpec]:
    """
    Automatically infer column widths and types from the data.
    """
    schema = df_or_lf.collect_schema()
    agg_exprs = []

    temp_lf = df_or_lf.lazy()
    if infer_specs_rows is not None:
        temp_lf = temp_lf.head(infer_specs_rows)

    for col_name, dtype in schema.items():
        if dtype == pl.Boolean:
            pass  # Fixed width
        elif dtype.is_integer():
            agg_exprs.append(
                pl.col(col_name).cast(pl.String).str.len_bytes().max().alias(col_name)
            )
        elif dtype.is_float():
            # For inference, we use max string length + 1
            agg_exprs.append(
                (pl.col(col_name).cast(pl.String).str.len_bytes().max() + 1).alias(
                    col_name
                )
            )
        else:  # String/Categorical
            agg_exprs.append(
                (pl.col(col_name).cast(pl.String).str.len_bytes().max() + 5).alias(
                    col_name
                )
            )

    if agg_exprs:
        stats = temp_lf.select(agg_exprs).collect()
    else:
        stats = pl.DataFrame()

    final_specs = []
    offset = 0

    dtype_map = {
        pl.Int8: DType.I8,
        pl.Int16: DType.I16,
        pl.Int32: DType.I32,
        pl.Int64: DType.I64,
        pl.UInt8: DType.U8,
        pl.UInt16: DType.U16,
        pl.UInt32: DType.U32,
        pl.UInt64: DType.U64,
        pl.Float32: DType.F32,
        pl.Float64: DType.F64,
    }

    for col_name, dtype in schema.items():
        if dtype == pl.Boolean:
            length = max(len(s) for s in bool_treatment)
            out_dtype = DType.String
        elif dtype.is_integer():
            length = stats[col_name][0] or 0
            out_dtype = dtype_map.get(dtype.base_type(), DType.I32)
        elif dtype.is_float():
            length = stats[col_name][0] or 0
            out_dtype = dtype_map.get(dtype.base_type(), DType.F64)
        else:
            length = stats[col_name][0] or 0
            out_dtype = DType.String

        final_specs.append(FieldSpec(col_name, offset, length, out_dtype))
        offset += length
    return final_specs


def _validate_and_format_batch(
    df: pl.DataFrame,
    specs: Sequence[PyFieldSpec],
    decimals: int,
    bool_treatment: tuple[str, str, str],
    number_padding: str,
    str_padding: str,
    pad_str_end: bool,
    skip_width_check: bool = False,
) -> pl.DataFrame:
    """
    Transform and validate a single batch of data for FWF writing.
    """
    exprs = []
    for spec in specs:
        col = pl.col(spec.name)
        dtype = df.schema[spec.name]

        if dtype == pl.Boolean:
            col = (
                pl.when(col.is_null())
                .then(pl.lit(bool_treatment[2]))
                .when(col)
                .then(pl.lit(bool_treatment[0]))
                .otherwise(pl.lit(bool_treatment[1]))
            )
        elif dtype.is_float():
            col = col.round(decimals)
        elif dtype == pl.String or isinstance(dtype, (pl.Categorical, pl.Enum)):
            # Strip quotes
            col = col.cast(pl.String).str.replace_all(r"[\"']", "")

        col = col.cast(pl.String).fill_null("")
        exprs.append(col.alias(f"_tmp_{spec.name}"))

    temp_df = df.select(exprs)

    # Validate lengths in parallel if not skipped
    if not skip_width_check:
        max_lens = temp_df.select(pl.all().str.len_bytes().max()).row(0)

        violations = [
            f"Column '{s.name}' has data longer ({max_len or 0}) than specified length ({s.length})"
            for s, max_len in zip(specs, max_lens)
            if (max_len or 0) > s.length
        ]

        if violations:
            raise ValueError("\n".join(violations))

    # Pad and Concat
    final_exprs = []
    for spec in specs:
        col = pl.col(f"_tmp_{spec.name}")
        dtype = df.schema[spec.name]
        if dtype.is_integer() or dtype.is_float():
            col = col.str.pad_start(spec.length, number_padding)
        else:
            if pad_str_end:
                col = col.str.pad_end(spec.length, str_padding)
            else:
                col = col.str.pad_start(spec.length, str_padding)
        final_exprs.append(col)

    return temp_df.select(pl.concat_str(final_exprs).alias("raw_line"))


def _build_spec_map(
    specs: Sequence[PyFieldSpec], schema: pl.Schema, simple_dtypes: bool
) -> dict[str, dict]:
    """
    Construct the final specification dictionary returned to the user.
    """
    spec_map = {}
    for spec in specs:
        dtype = schema[spec.name]
        dt = str(spec.dtype)
        if simple_dtypes:
            if dt.startswith("I") or dt.startswith("U") or "int" in dt.lower():
                dt = "int"
            elif dt.startswith("F") or "float" in dt.lower():
                dt = "f64"
            elif dt == "String" or "str" in dt.lower():
                dt = "str"
            elif dtype == pl.Boolean:
                dt = "bool"

        spec_map[spec.name] = {
            "offset": spec.offset,
            "length": spec.length,
            "dtype": dt,
        }
    return spec_map


def write_fwf_pl(
    df: pl.DataFrame | pl.LazyFrame,
    path: str,
    specs: Sequence[PyFieldSpec] | None = None,
    number_padding: str = " ",
    str_padding: str = " ",
    pad_str_end: bool = True,
    decimals: int = 3,
    bool_treatment: tuple[str, str, str] = ("T", "F", "null"),
    simple_dtypes: bool = True,
) -> dict[str, dict]:
    """
    Write a Polars DataFrame or LazyFrame to a Fixed-Width File (FWF) eagerly.

    Parameters
    ----------
    df : pl.DataFrame | pl.LazyFrame
        The DataFrame or LazyFrame to write.
    path : str
        The path to the output file.
    specs : Sequence[PyFieldSpec] | None, optional
        A sequence of FieldSpec objects defining the output layout.
        If None, the specification is inferred from the data.
    number_padding : str, default " "
        The padding character for numeric columns (right-aligned).
    str_padding : str, default " "
        The padding character for string columns.
    pad_str_end : bool, default True
        If True, string columns are left-aligned (padded at the end).
        If False, string columns are right-aligned (padded at the start).
    decimals : int, default 3
        The precision for float columns. Floats are rounded to this value.
    bool_treatment : tuple[str, str, str], default ("T", "F", "null")
        The string representations for True, False, and Null boolean values.
    simple_dtypes : bool, default True
        If True, the returned specification dictionary uses simplified
        dtype names ('int', 'str', 'f64', 'bool').

    Returns
    -------
    dict[str, dict]
        The specification used to write the file, mapping column names
        to {offset, length, dtype}.
    """
    _check_supported_types(df)
    bool_treatment = _validate_bool_treatment(bool_treatment)

    skip_width_check = False
    if specs is not None:
        _check_specs_contiguity(specs)
        _check_specs_capacity(specs)
        target_cols = [s.name for s in specs]
        if isinstance(df, pl.LazyFrame):
            df = df.select(target_cols).collect()
        else:
            df = df.select(target_cols)
        final_specs = specs
    else:
        final_specs = _infer_specs(df, bool_treatment, decimals)
        _check_specs_capacity(final_specs)
        if isinstance(df, pl.LazyFrame):
            df = df.collect()
        skip_width_check = True

    batch_out = _validate_and_format_batch(
        df,
        final_specs,
        decimals,
        bool_treatment,
        number_padding,
        str_padding,
        pad_str_end,
        skip_width_check=skip_width_check,
    )

    batch_out.write_csv(
        path, include_header=False, quote_style="never", line_terminator="\n"
    )

    return _build_spec_map(final_specs, df.schema, simple_dtypes)


def sink_fwf_pl(
    lf: pl.LazyFrame,
    path: str,
    specs: Sequence[PyFieldSpec] | None = None,
    number_padding: str = " ",
    str_padding: str = " ",
    pad_str_end: bool = True,
    decimals: int = 3,
    bool_treatment: tuple[str, str, str] = ("T", "F", "null"),
    simple_dtypes: bool = True,
    infer_specs_rows: int | None = 1000,
) -> dict[str, dict]:
    """
    Stream a Polars LazyFrame to a Fixed-Width File (FWF).

    This function processes the data in batches to keep memory usage low,
    making it suitable for very large datasets.

    Parameters
    ----------
    lf : pl.LazyFrame
        The LazyFrame to write.
    path : str
        The path to the output file.
    specs : Sequence[PyFieldSpec] | None, optional
        A sequence of FieldSpec objects defining the output layout.
        If None, the specification is inferred from the first `infer_specs_rows`.
    number_padding : str, default " "
        The padding character for numeric columns (right-aligned).
    str_padding : str, default " "
        The padding character for string columns.
    pad_str_end : bool, default True
        If True, string columns are left-aligned (padded at the end).
        If False, string columns are right-aligned (padded at the start).
    decimals : int, default 3
        The precision for float columns. Floats are rounded to this value.
    bool_treatment : tuple[str, str, str], default ("T", "F", "null")
        The string representations for True, False, and Null boolean values.
    simple_dtypes : bool, default True
        If True, the returned specification dictionary uses simplified
        dtype names ('int', 'str', 'f64', 'bool').
    infer_specs_rows : int | None, default 1000
        Number of rows to use for schema inference if `specs` is None.

    Returns
    -------
    dict[str, dict]
        The specification used to write the file, mapping column names
        to {offset, length, dtype}.

    Examples
    --------
    >>> import polars as pl
    >>> import ffwf as fw
    >>> lf = pl.LazyFrame({"a": range(1000000)})
    >>> fw.sink_fwf_pl(lf, "large.fwf", decimals=2)
    """
    _check_supported_types(lf)
    bool_treatment = _validate_bool_treatment(bool_treatment)

    skip_width_check = False
    if specs is not None:
        _check_specs_contiguity(specs)
        _check_specs_capacity(specs)
        lf = lf.select([s.name for s in specs])
        final_specs = specs
    else:
        final_specs = _infer_specs(
            lf, bool_treatment, decimals, infer_specs_rows=infer_specs_rows
        )
        _check_specs_capacity(final_specs)
        skip_width_check = True

    current_row = 0
    with open(path, "wb") as f:
        for i, batch in enumerate(lf.collect_batches()):
            try:
                batch_out = _validate_and_format_batch(
                    batch,
                    final_specs,
                    decimals,
                    bool_treatment,
                    number_padding,
                    str_padding,
                    pad_str_end,
                    skip_width_check=skip_width_check,
                )
            except ValueError as e:
                raise ValueError(
                    f"Batch {i} (rows {current_row} - {current_row + len(batch) - 1}) failed validation: {e}"
                ) from e

            batch_out.write_csv(
                f, include_header=False, quote_style="never", line_terminator="\n"
            )
            current_row += len(batch)

    return _build_spec_map(final_specs, lf.collect_schema(), simple_dtypes)
