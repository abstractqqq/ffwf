from __future__ import annotations

from typing import TYPE_CHECKING, Sequence

try:
    import pandas as pd
except ImportError:
    pd = None

import ffwf as fw

if TYPE_CHECKING:
    from ._fwf import PyFieldSpec

__all__ = ["read_fwf_pd"]


def read_fwf_pd(
    path: str,
    specs: Sequence[PyFieldSpec],
    line_length: int | None = None,
    newline: str | bytes = "\n",
    chunk_size: int | None = None,
    parallel: bool = True,
    **kwargs,
) -> pd.DataFrame:
    """
    Read a fixed-width file into a Pandas DataFrame by going through PyArrow.

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
        Number of rows per internal batch.
    parallel : bool, default True
        Whether to use multi-threaded parsing.
    **kwargs
        Additional arguments passed to PyArrow Table's `to_pandas` method.

    Returns
    -------
    pd.DataFrame
        A Pandas DataFrame.
    """
    if pd is None:
        raise ImportError("read_fwf_pd requires pandas to be installed.")

    table = fw.read_fwf_arrow(
        path=path,
        specs=specs,
        line_length=line_length,
        newline=newline,
        chunk_size=chunk_size,
        parallel=parallel,
    )

    return table.to_pandas(**kwargs)
