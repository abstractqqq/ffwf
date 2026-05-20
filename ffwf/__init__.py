from __future__ import annotations

import importlib.metadata
import warnings
from typing import TYPE_CHECKING, Sequence

import pyarrow as pa

from ._fwf import DType, FwfParser, FwfReader, PyFieldSpec

try:
    __version__ = importlib.metadata.version("ffwf")
except importlib.metadata.PackageNotFoundError:
    __version__ = "unknown"

if TYPE_CHECKING:
    from ._fwf import PyFieldSpec as FieldSpec

__all__ = [
    "__version__",
    "FwfParser",
    "FieldSpec",
    "DType",
    "read_fwf_arrow",
]


def FieldSpec(
    name: str,
    offset: int,
    length: int,
    dtype: DType | str,
    padding: int | None = None,
) -> PyFieldSpec:
    """
    Define a field specification for a fixed-width file.

    Parameters
    ----------
    name : str
        The name of the column.
    offset : int
        The starting byte offset of the field.
    length : int
        The length of the field in bytes.
    dtype : DType | str
        The data type of the field. Can be a DType enum or a string alias
        like 'str', 'int', 'f64', etc.
    padding : int | None, optional
        Optional padding byte.

    Returns
    -------
    PyFieldSpec
        An internal field specification object.
    """
    if isinstance(dtype, str):
        dtype_lower = dtype.lower()
        if dtype_lower in ("str", "string"):
            resolved_dtype = DType.String
        elif dtype_lower in ("int", "integer", "i32"):
            resolved_dtype = DType.I32
        elif dtype_lower == "i8":
            resolved_dtype = DType.I8
        elif dtype_lower == "i16":
            resolved_dtype = DType.I16
        elif dtype_lower == "i64":
            resolved_dtype = DType.I64
        elif dtype_lower == "u8":
            resolved_dtype = DType.U8
        elif dtype_lower == "u16":
            resolved_dtype = DType.U16
        elif dtype_lower == "u32":
            resolved_dtype = DType.U32
        elif dtype_lower == "u64":
            resolved_dtype = DType.U64
        elif dtype_lower in ("f32", "float"):
            resolved_dtype = DType.F32
        elif dtype_lower in ("f64", "double"):
            resolved_dtype = DType.F64
        else:
            raise ValueError(f"Unknown DType alias: {dtype}")
    else:
        resolved_dtype = dtype

    if length <= 0:
        raise ValueError(f"FieldSpec width must be positive, got {length}")

    # Integer width capacity validation
    max_w = resolved_dtype.max_width()
    if max_w is not None and length > max_w:
        warnings.warn(
            f"Width {length} exceeds maximum capacity for {resolved_dtype} "
            f"(max {max_w} characters). This may cause overflow or parsing errors.",
            UserWarning,
            stacklevel=2,
        )

    return PyFieldSpec(name, offset, length, resolved_dtype, padding)


class ArrowCapsule:
    """
    Internal adapter to bridge Arrow C Data Interface capsules with Arrow/Polars.
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


def read_fwf_arrow(
    path: str,
    specs: Sequence[PyFieldSpec],
    line_length: int | None = None,
    newline: str | bytes = "\n",
    chunk_size: int | None = None,
    parallel: bool = True,
) -> pa.Table:
    """
    Read a fixed-width file into a PyArrow Table using zero-copy Arrow transfer.
    """

    # DType to arrow mapping helper
    def _dt_to_pa(dt):
        if dt == DType.I8:
            return pa.int8()
        if dt == DType.I16:
            return pa.int16()
        if dt == DType.I32:
            return pa.int32()
        if dt == DType.I64:
            return pa.int64()
        if dt == DType.U8:
            return pa.uint8()
        if dt == DType.U16:
            return pa.uint16()
        if dt == DType.U32:
            return pa.uint32()
        if dt == DType.U64:
            return pa.uint64()
        if dt == DType.F32:
            return pa.float32()
        if dt == DType.F64:
            return pa.float64()
        if dt == DType.String:
            return pa.utf8()
        return pa.null()

    # Handle empty file
    import os

    if not os.path.exists(path) or os.path.getsize(path) == 0:
        schema = pa.schema([(s.name, _dt_to_pa(s.dtype)) for s in specs])
        return pa.Table.from_batches([], schema=schema)

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

    parser = FwfParser(
        list(specs),
        actual_stride,
        parallel=parallel,
        chunk_size=chunk_size,
    )

    capsule_tuples = parser._parse_path(path)

    if not capsule_tuples:
        schema = pa.schema([(s.name, _dt_to_pa(s.dtype)) for s in specs])
        return pa.Table.from_batches([], schema=schema)

    batches = [pa.record_batch(ArrowCapsule(c)) for c in capsule_tuples]
    return pa.Table.from_batches(batches)
