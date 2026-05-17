import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import polars.testing as pl_testing

import polars_fwf as pfwf


def get_specs_and_widths():
    types_str = (
        ["i8"] * 15
        + ["i16"] * 15
        + ["i32"] * 15
        + ["i64"] * 15
        + ["u8"] * 15
        + ["u16"] * 15
        + ["u32"] * 15
        + ["u64"] * 15
        + ["f32"] * 19
        + ["f64"] * 20
        + ["str"] * 40
        + ["state"] * 1
    )
    widths = (
        [4] * 15
        + [6] * 15
        + [11] * 15
        + [20] * 15
        + [4] * 15
        + [6] * 15
        + [11] * 15
        + [20] * 15
        + [12] * 19
        + [20] * 20
        + [10] * 40
        + [2] * 1
    )

    specs = []
    curr = 0
    for i, (t, w) in enumerate(zip(types_str, widths)):
        name = f"col_{i}" if t != "state" else "state"
        if t == "state":
            dtype = pfwf.DType.String
        else:
            dtype = {
                "i8": pfwf.DType.I8,
                "i16": pfwf.DType.I16,
                "i32": pfwf.DType.I32,
                "i64": pfwf.DType.I64,
                "u8": pfwf.DType.U8,
                "u16": pfwf.DType.U16,
                "u32": pfwf.DType.U32,
                "u64": pfwf.DType.U64,
                "f32": pfwf.DType.F32,
                "f64": pfwf.DType.F64,
                "str": pfwf.DType.String,
            }[t]
        specs.append(pfwf.FieldSpec(name, curr, w, dtype))
        curr += w
    return specs, widths


def validate_dfs(df_pandas, df_custom):
    pl_pandas = pl.from_pandas(df_pandas.head(100))
    pl_custom = df_custom.head(100)
    pl_pandas.columns = pl_custom.columns
    try:
        pl_testing.assert_frame_equal(pl_custom, pl_pandas, check_dtypes=False)
        print("Validation successful: First 100 rows match.")
    except Exception as e:
        print(f"Validation failed: {e}")


def run_test_set(filename, specs, widths, test_fn, title, chart_name):
    print(f"\n--- {title} ---")
    results = {}

    # 1. Pandas
    start = time.perf_counter()
    res_pandas = test_fn(filename, specs, widths, mode="pandas")
    results["Pandas"] = time.perf_counter() - start
    print(f"Pandas: {results['Pandas']:.4f}s")

    # 2. polars-fwf Eager (Seq)
    start = time.perf_counter()
    test_fn(filename, specs, widths, mode="eager_seq")
    results["polars-fwf Eager (Seq)"] = time.perf_counter() - start
    print(f"polars-fwf Eager (Seq): {results['polars-fwf Eager (Seq)']:.4f}s")

    # 3. polars-fwf Eager (Par)
    start = time.perf_counter()
    res_custom = test_fn(filename, specs, widths, mode="eager_par")
    results["polars-fwf Eager (Par)"] = time.perf_counter() - start
    print(f"polars-fwf Eager (Par): {results['polars-fwf Eager (Par)']:.4f}s")

    # 4. polars-fwf Lazy (Seq)
    start = time.perf_counter()
    test_fn(filename, specs, widths, mode="lazy_seq")
    results["polars-fwf Lazy (Seq)"] = time.perf_counter() - start
    print(f"polars-fwf Lazy (Seq): {results['polars-fwf Lazy (Seq)']:.4f}s")

    # 5. polars-fwf Lazy (Par)
    start = time.perf_counter()
    test_fn(filename, specs, widths, mode="lazy_par")
    results["polars-fwf Lazy (Par)"] = time.perf_counter() - start
    print(f"polars-fwf Lazy (Par): {results['polars-fwf Lazy (Par)']:.4f}s")

    # Validation (only if eager_par and pandas returned frames)
    if title == "Pure Reading Benchmark":
        validate_dfs(res_pandas, res_custom)

    plot_results(results, title, chart_name)
    return results


def pure_reading_fn(filename, specs, widths, mode):
    if mode == "pandas":
        colspecs = []
        c = 0
        for w in widths:
            colspecs.append((c, c + w))
            c += w
        return pd.read_fwf(filename, colspecs=colspecs, header=None)
    elif mode == "eager_seq":
        return pfwf.read_fwf(filename, specs, parallel=False)
    elif mode == "eager_par":
        return pfwf.read_fwf(filename, specs, parallel=True)
    elif mode == "lazy_seq":
        return pfwf.scan_fwf(filename, specs, parallel=False).collect()
    elif mode == "lazy_par":
        return pfwf.scan_fwf(filename, specs, parallel=True).collect()


def pipeline_fn(filename, specs, widths, mode):
    def apply_pipeline(df):
        if isinstance(df, pd.DataFrame):
            df.columns = [s.name for s in specs]
            return df[(df["state"] == "NY") & (df["col_30"] > 10**15)][
                ["state", "col_30", "col_120"]
            ]
        else:
            return df.filter(
                (pl.col("state") == "NY") & (pl.col("col_30") > 10**15)
            ).select(["state", "col_30", "col_120"])

    if mode == "pandas":
        df = pure_reading_fn(filename, specs, widths, "pandas")
        return apply_pipeline(df)
    elif mode == "eager_seq":
        df = pfwf.read_fwf(filename, specs, parallel=False)
        return apply_pipeline(df)
    elif mode == "eager_par":
        df = pfwf.read_fwf(filename, specs, parallel=True)
        return apply_pipeline(df)
    elif mode == "lazy_seq":
        return (
            pfwf.scan_fwf(filename, specs, parallel=False)
            .filter((pl.col("state") == "NY") & (pl.col("col_30") > 10**15))
            .select(["state", "col_30", "col_120"])
            .collect()
        )
    elif mode == "lazy_par":
        return (
            pfwf.scan_fwf(filename, specs, parallel=True)
            .filter((pl.col("state") == "NY") & (pl.col("col_30") > 10**15))
            .select(["state", "col_30", "col_120"])
            .collect()
        )


def aggregation_fn(filename, specs, widths, mode):
    def apply_agg(df):
        if isinstance(df, pd.DataFrame):
            df.columns = [s.name for s in specs]
            return (
                df[df["col_30"] > 10**14]
                .groupby("state")["col_120"]
                .agg(["min", "max"])
            )
        else:
            return (
                df.filter(pl.col("col_30") > 10**14)
                .group_by("state")
                .agg(
                    [
                        pl.col("col_120").min().alias("min"),
                        pl.col("col_120").max().alias("max"),
                    ]
                )
            )

    if mode == "pandas":
        df = pure_reading_fn(filename, specs, widths, "pandas")
        return apply_agg(df)
    elif mode == "eager_seq":
        df = pfwf.read_fwf(filename, specs, parallel=False)
        return apply_agg(df)
    elif mode == "eager_par":
        df = pfwf.read_fwf(filename, specs, parallel=True)
        return apply_agg(df)
    elif mode == "lazy_seq":
        return (
            pfwf.scan_fwf(filename, specs, parallel=False)
            .filter(pl.col("col_30") > 10**14)
            .group_by("state")
            .agg(
                [
                    pl.col("col_120").min().alias("min"),
                    pl.col("col_120").max().alias("max"),
                ]
            )
            .collect()
        )
    elif mode == "lazy_par":
        return (
            pfwf.scan_fwf(filename, specs, parallel=True)
            .filter(pl.col("col_30") > 10**14)
            .group_by("state")
            .agg(
                [
                    pl.col("col_120").min().alias("min"),
                    pl.col("col_120").max().alias("max"),
                ]
            )
            .collect()
        )


def plot_results(results, title, filename):
    names = list(results.keys())
    values = list(results.values())
    baseline = results["Pandas"]
    normalized = [v / baseline for v in values]

    plt.figure(figsize=(14, 8))
    colors = ["gray", "lightblue", "blue", "lightgreen", "green"]
    bars = plt.bar(names, normalized, color=colors[: len(names)])
    plt.axhline(1.0, color="red", linestyle="--", label="Pandas Baseline")
    plt.ylabel("Time (Normalized to Pandas)")
    plt.title(title)
    plt.xticks(rotation=45, ha="right")
    plt.ylim(0, max(normalized) * 1.3)

    for bar, val in zip(bars, values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{val:.4f}s",
            ha="center",
            va="bottom",
            rotation=0,
        )

    plt.tight_layout()
    os.makedirs("plots", exist_ok=True)
    out_path = os.path.join("plots", filename)
    plt.savefig(out_path)
    print(f"Chart saved as {out_path}")


if __name__ == "__main__":
    path = "data/large_bench.fwf"
    if not os.path.exists(path):
        print("Data file not found. Generating...")
        from generate_large_fwf import generate_large_fwf

        generate_large_fwf(path)

    specs, widths = get_specs_and_widths()

    run_test_set(
        path,
        specs,
        widths,
        pure_reading_fn,
        "Pure Reading Benchmark",
        "read_benchmark.png",
    )
    run_test_set(
        path,
        specs,
        widths,
        pipeline_fn,
        "Typical Data Pipeline Test",
        "pipeline_benchmark.png",
    )
    run_test_set(
        path,
        specs,
        widths,
        aggregation_fn,
        "Aggregation Pipeline Test",
        "agg_benchmark.png",
    )
