# ffwf (Fast Fwf)

`ffwf` provides a high-performance Fixed-Width File (FWF) parser with a Rust core.

**🚀 Performance Focus**: By default, `ffwf` only includes `read_fwf_arrow` for zero-copy parsing into PyArrow Tables. Its true power is unlocked through the optional **Polars** integration, enabling streaming and multi-threaded lazy execution that is **~200x faster** than Pandas.

## Why Fixed-Width?

While formats like CSV are more common, Fixed-Width Files (FWF) provide a more robust **data contract** for high-integrity B2B exchanges:

- **Structural Integrity**: Unlike CSV, FWF is immune to "delimiter collision" and "quote hell." Comma, quotes, or newlines within a field cannot break the physical layout of the file.
- **Predictable Performance**: Because column positions are known at the byte level, parsers can slice data with near-zero overhead.
- **Consistency**: The fixed schema ensures that if a spec defines a column as 10 bytes, it remains 10 bytes. This prevents the "silent misalignment" often caused by poorly escaped CSVs.
- **Speed**: Parsing FWF files is faster than CSV due to the fixed schema and lack of delimiters.

`ffwf` brings the reliability of these legacy contracts into modern data ecosystems with native-speed parsing.

## Usage

The core package provides `read_fwf_arrow` for PyArrow. Integration for Polars and Pandas is available via optional modules.

### PyArrow (Default)

```python
import ffwf as fw

specs = [
    fw.FieldSpec("id", offset=0, length=5, dtype="int"),
    fw.FieldSpec("val", offset=5, length=10, dtype="float"),
    fw.FieldSpec("tag", offset=15, length=5, dtype="str"),
]

table = fw.read_fwf_arrow("data.fwf", specs)
```

### Polars (Highly Recommended)

Unlock the best performance and streaming capabilities. Polars functions are in `ffwf.polars` with a `_pl` suffix.

```python
import polars as pl
import ffwf as fw
import ffwf.polars as plfw

# 1. Define field specifications
specs = [
    fw.FieldSpec("id", offset=0, length=5, dtype="int"),
    fw.FieldSpec("val", offset=5, length=10, dtype="float"),
    fw.FieldSpec("tag", offset=15, length=5, dtype="str"),
]

# 2. Eager parsing (returns pl.DataFrame)
df = plfw.read_fwf_pl("data.fwf", specs)

# 3. Lazy parsing (returns pl.LazyFrame)
lazy_df = plfw.scan_fwf_pl("data.fwf", specs)

result = lazy_df.filter(pl.col("val") > 100.0).group_by("tag").count().collect()
```

### Pandas

For existing Pandas workflows, `ffwf.pandas` provides a simple wrapper that parses via Arrow.

```python
import ffwf as fw
import ffwf.pandas as pdfw

specs = [fw.FieldSpec("id", 0, 5, "int")]
df = pdfw.read_fwf_pd("data.fwf", specs)
```

## Writing Fixed-Width Files (Polars)

`ffwf.polars` provides eager (`write_fwf_pl`) and streaming (`sink_fwf_pl`) writers.

### Eager Writing (DataFrame)

```python
# Automatic inference of widths and types
specs = plfw.write_fwf_pl(df, "output.fwf")

# Explicit specification
specs = [
    fw.FieldSpec("id", 0, 5, "int"),
    fw.FieldSpec("val", 5, 10, "float")
]
plfw.write_fwf_pl(df, "output.fwf", specs=specs)
```

### Streaming Writing (LazyFrame)

For large datasets, use `sink_fwf_pl` to validate and write data batch-by-batch without loading the entire frame into memory.

```python
# Streaming write
plfw.sink_fwf_pl(lazy_df, "large_output.fwf", decimals=2)
```

### Key Writing Features

- **Validation**: Strict width validation before writing. `sink_fwf_pl` reports the exact batch and row range on failure.
- **Float Rounding**: Floats are rounded to `decimals` to prevent width violations.
- **Boolean Treatment**: Customizable mapping for booleans (e.g., `bool_treatment=('Y', 'N', ' ')`).
- **Quote Stripping**: Automatically strips `'` and `"` from strings.
- **Alignment**: Control string alignment with `pad_str_end`.

### Supported Data Types

Supported `fw.DType` members:
- **Integers**: `I8`, `I16`, `I32`, `I64`, `U8`, `U16`, `U32`, `U64`
- **Floats**: `F32`, `F64` (supports `NaN` and `inf`)
- **Strings**: `String`

## Benchmarks

The following benchmarks compare `ffwf` against `pandas.read_fwf` (v2.2.3) using a synthetic dataset of **200,000 rows and 200 columns (~430MB)**.

| Method | Reading | Pipeline | Aggregation |
| :--- | :--- | :--- | :--- |
| **Pandas** | 16.06s | 16.16s | 16.79s |
| **ffwf (Seq)** | 0.51s | 0.51s | 0.51s |
| **ffwf (Par)** | **0.09s** | **0.08s** | **0.08s** |

*Benchmarks conducted on a 16-core machine. ffwf is **~170x faster** than Pandas for pure reading and **~200x faster** for filtered pipelines.*

## Integration with Other Dataframe Packages

The core of `ffwf` is designed to be dataframe-agnostic by returning zero-copy PyArrow Tables. If you use a dataframe library other than Polars or Pandas (e.g., DuckDB, Daft, Modin), you can easily integrate it yourself as long as the library supports the [Arrow C Data Interface](https://arrow.apache.org/docs/format/CDataInterface.html).

For a reference implementation, see `ffwf/pandas.py`. The general pattern is:

```python
import ffwf as fw

# 1. Parse to Arrow Table
table = fw.read_fwf_arrow("data.fwf", specs)

# 2. Convert to your preferred format (if it supports Arrow)
# example_df = your_library.from_arrow(table)
```

**Note**: The package owner does not intend to add built-in support for more dataframe packages.

### Writing FWF

Please note that **writing FWF files is only available with Polars** as of now, via `plfw.write_fwf_pl` and the streaming `plfw.sink_fwf_pl` variant.

## Building Locally

```bash
# Clone the repository
git clone <repo-url>
cd ffwf

# Create a virtual environment
uv venv
source .venv/bin/activate

# Install the package in editable mode with development dependencies
uv pip install -e ".[dev]"

# Build the Rust extension
RUSTFLAGS="-C target-cpu=native" maturin develop --release
```

## Other Projects

1. My Data Science Extension to Polars [Polars DS](https://github.com/abstractqqq/polars_ds_extension)

## AI Assistance Disclosure

This project uses AI-assisted development with Gemini.
