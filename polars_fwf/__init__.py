from __future__ import annotations

from typing import TYPE_CHECKING, Iterator

import polars as pl
from polars.io.plugins import register_io_source

from ._fwf import DType, FieldSpec, FwfParser

if TYPE_CHECKING:
    from collections.abc import Sequence

__all__ = ["FwfParser", "FieldSpec", "DType", "read_fwf", "scan_fwf"]


class ArrowCapsule:
    """
    Internal adapter to bridge Arrow C Data Interface capsules with Polars.
    """

    def __init__(self, capsules: tuple):
        """
        Initialize the adapter with a tuple of (array_capsule, schema_capsule).

        Parameters
        ----------
        capsules : tuple
            A tuple containing the Arrow C Data Interface capsules.
        """
        self.array_capsule, self.schema_capsule = capsules

    def __arrow_c_array__(self, requested_schema=None):
        """
        Implement the Arrow C Data Interface protocol.
        """
        return self.schema_capsule, self.array_capsule


class FwfSource:
    """
    Polars IO Source implementation for Fixed-Width Files (FWF).
    """

    def __init__(
        self,
        path: str,
        specs: Sequence[FieldSpec],
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
        specs : Sequence[FieldSpec]
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
        parser = FwfParser(list(self.specs), self.line_length, parallel=self.parallel)
        if batch_size or self.chunk_size:
            parser.set_chunk_size(batch_size or self.chunk_size)

        capsule_tuples = parser._parse_path(self.path)

        for capsules in capsule_tuples:
            adapter = ArrowCapsule(capsules)
            df = pl.from_arrow(adapter)

            if with_columns:
                df = df.select(with_columns)

            if n_rows is not None:
                if n_rows <= 0:
                    break
                df = df.head(n_rows)
                n_rows -= len(df)

            yield df


def read_fwf(
    path: str,
    specs: Sequence[FieldSpec],
    line_length: int | None = None,
    newline: str | bytes = "\n",
    chunk_size: int | None = None,
    parallel: bool = True,
) -> pl.DataFrame:
    """
    Read a fixed-width file into a Polars DataFrame using zero-copy Arrow transfer.

    Parameters
    ----------
    path : str
        Path to the FWF file.
    specs : Sequence[FieldSpec]
        Sequence of FieldSpec objects defining column layout and types.
    line_length : int | None, optional
        Total width of each line in bytes (including newline). If None, auto-detected.
    newline : str | bytes, default "\\n"
        Newline character(s) used in the file.
    chunk_size : int | None, optional
        Number of rows per internal batch. If None, inferred by Rust parser.
    parallel : bool, default True
        Use multi-threaded parsing in the Rust core.

    Returns
    -------
    pl.DataFrame
        A Polars DataFrame containing the parsed data.
    """
    newline_bytes = newline if isinstance(newline, bytes) else newline.encode("utf-8")
    stride, data_len = FwfParser.detect_line_length(path, newline_bytes)

    actual_stride = line_length if line_length is not None else stride
    actual_data_len = actual_stride - len(newline_bytes)

    total_spec_width = sum(s.length for s in specs)
    if total_spec_width != actual_data_len:
        raise ValueError(
            f"Partial specification detected. Total spec width ({total_spec_width}) "
            f"must match data length ({actual_data_len})."
        )

    return scan_fwf(path, specs, actual_stride, newline, chunk_size, parallel).collect()


def scan_fwf(
    path: str,
    specs: Sequence[FieldSpec],
    line_length: int | None = None,
    newline: str | bytes = "\n",
    chunk_size: int | None = None,
    parallel: bool = True,
) -> pl.LazyFrame:
    """
    Scan a fixed-width file lazily using Polars IO Plugin interface.

    Parameters
    ----------
    path : str
        Path to the FWF file.
    specs : Sequence[FieldSpec]
        Sequence of FieldSpec objects defining column layout and types.
    line_length : int | None, optional
        Total width of each line in bytes (including newline). If None, auto-detected.
    newline : str | bytes, default "\\n"
        Newline character(s) used in the file.
    chunk_size : int | None, optional
        Number of rows per internal batch. If None, inferred by Rust parser.
    parallel : bool, default True
        Use multi-threaded parsing in the Rust core.

    Returns
    -------
    pl.LazyFrame
        A Polars LazyFrame.
    """
    newline_bytes = newline if isinstance(newline, bytes) else newline.encode("utf-8")

    if line_length is None:
        line_length, _ = FwfParser.detect_line_length(path, newline_bytes)

    schema = pl.Schema({s.name: _dtype_to_pl(s.dtype) for s in specs})

    return register_io_source(
        io_source=FwfSource(path, specs, line_length, chunk_size, parallel),
        schema=schema,
    )


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


# Attach to pl namespace
pl.read_fwf = read_fwf
pl.scan_fwf = scan_fwf
