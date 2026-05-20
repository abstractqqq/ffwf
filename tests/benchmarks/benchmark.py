import os
import time

import matplotlib.pyplot as plt
import pandas as pd
import polars as pl
import polars.testing as pl_testing

import ffwf as fw
import ffwf.polars as plfw


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
            dtype = fw.DType.String
        else:
            dtype = {
                "i8": fw.DType.I8,
                "i16": fw.DType.I16,
                "i32": fw.DType.I32,
                "i64": fw.DType.I64,
                "u8": fw.DType.U8,
                "u16": fw.DType.U16,
                "u32": fw.DType.U32,
                "u64": fw.DType.U64,
                "f32": fw.DType.F32,
                "f64": fw.DType.F64,
                "str": fw.DType.String,
            }[t]
        specs.append(fw.FieldSpec(name, curr, w, dtype))
        curr += w
    return specs, widths


def validate_dfs(df_pandas, df_custom, title):
    if df_pandas is None or df_custom is None:
        return

    # Standardize pandas for comparison
    if isinstance(df_pandas, pd.DataFrame):
        # Reset index and drop it to avoid 'index' column in Polars
        if not isinstance(df_pandas.index, pd.RangeIndex):
            df_pandas = df_pandas.reset_index(drop=True)
        pl_pandas = pl.from_pandas(df_pandas)
    else:
        pl_pandas = df_pandas

    pl_custom = df_custom

    # Standardize column names for comparison
    if len(pl_pandas.columns) == len(pl_custom.columns):
        pl_pandas.columns = pl_custom.columns

    try:
        # Aggregation results might need sorting for comparison
        if "Aggregation" in title:
            pl_pandas = pl_pandas.sort("state")
            pl_custom = pl_custom.sort("state")

        pl_testing.assert_frame_equal(pl_custom, pl_pandas, check_dtypes=False)
        print(f"Validation successful for '{title}'.")
    except Exception as e:
        print(f"Validation failed for '{title}': {e}")


def run_test_set(filename, specs, widths, test_fn, title, chart_name):
    print(f"\n--- {title} ---")
    results = {}

    # 1. Pandas (Baseline and Reference)
    start = time.perf_counter()
    res_pandas = test_fn(filename, specs, widths, mode="pandas")
    results["Pandas"] = time.perf_counter() - start
    print(f"Pandas: {results['Pandas']:.4f}s")

    # 2. ffwf Eager (Seq)
    start = time.perf_counter()
    res_eager_seq = test_fn(filename, specs, widths, mode="eager_seq")
    results["ffwf Eager (Seq)"] = time.perf_counter() - start
    print(f"ffwf Eager (Seq): {results['ffwf Eager (Seq)']:.4f}s")
    validate_dfs(res_pandas, res_eager_seq, f"{title} (Eager Seq)")

    # 3. ffwf Eager (Par)
    start = time.perf_counter()
    res_eager_par = test_fn(filename, specs, widths, mode="eager_par")
    results["ffwf Eager (Par)"] = time.perf_counter() - start
    print(f"ffwf Eager (Par): {results['ffwf Eager (Par)']:.4f}s")
    validate_dfs(res_pandas, res_eager_par, f"{title} (Eager Par)")

    # 4. ffwf Lazy (Seq)
    start = time.perf_counter()
    res_lazy_seq = test_fn(filename, specs, widths, mode="lazy_seq")
    results["ffwf Lazy (Seq)"] = time.perf_counter() - start
    print(f"ffwf Lazy (Seq): {results['ffwf Lazy (Seq)']:.4f}s")
    validate_dfs(res_pandas, res_lazy_seq, f"{title} (Lazy Seq)")

    # 5. ffwf Lazy (Par)
    start = time.perf_counter()
    res_lazy_par = test_fn(filename, specs, widths, mode="lazy_par")
    results["ffwf Lazy (Par)"] = time.perf_counter() - start
    print(f"ffwf Lazy (Par): {results['ffwf Lazy (Par)']:.4f}s")
    validate_dfs(res_pandas, res_lazy_par, f"{title} (Lazy Par)")

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
        return plfw.read_fwf_pl(filename, specs, parallel=False)
    elif mode == "eager_par":
        return plfw.read_fwf_pl(filename, specs, parallel=True)
    elif mode == "lazy_seq":
        return plfw.scan_fwf_pl(filename, specs, parallel=False).collect()
    elif mode == "lazy_par":
        return plfw.scan_fwf_pl(filename, specs, parallel=True).collect()


def pipeline_fn(filename, specs, widths, mode):
    def apply_pipeline(df):
        if isinstance(df, pd.DataFrame):
            df.columns = [s.name for s in specs]
            # Reset index here to avoid carrying it into comparison
            res = df[(df["state"] == "NY") & (df["col_30"] > 10**10)][
                ["state", "col_30", "col_120"]
            ]
            return res.reset_index(drop=True)
        else:
            return df.filter(
                (pl.col("state") == "NY") & (pl.col("col_30") > 10**10)
            ).select(["state", "col_30", "col_120"])

    if mode == "pandas":
        df = pure_reading_fn(filename, specs, widths, "pandas")
        return apply_pipeline(df)
    elif mode == "eager_seq":
        df = plfw.read_fwf_pl(filename, specs, parallel=False)
        return apply_pipeline(df)
    elif mode == "eager_par":
        df = plfw.read_fwf_pl(filename, specs, parallel=True)
        return apply_pipeline(df)
    elif mode == "lazy_seq":
        return (
            plfw.scan_fwf_pl(filename, specs, parallel=False)
            .filter((pl.col("state") == "NY") & (pl.col("col_30") > 10**10))
            .select(["state", "col_30", "col_120"])
            .collect()
        )
    elif mode == "lazy_par":
        return (
            plfw.scan_fwf_pl(filename, specs, parallel=True)
            .filter((pl.col("state") == "NY") & (pl.col("col_30") > 10**10))
            .select(["state", "col_30", "col_120"])
            .collect()
        )


def aggregation_fn(filename, specs, widths, mode):
    def apply_agg(df):
        if isinstance(df, pd.DataFrame):
            df.columns = [s.name for s in specs]
            res = (
                df[df["col_30"] > 10**10]
                .groupby("state")["col_120"]
                .agg(["min", "max"])
            )
            # reset_index makes 'state' a column instead of index
            return res.reset_index()
        else:
            return (
                df.filter(pl.col("col_30") > 10**10)
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
        df = plfw.read_fwf_pl(filename, specs, parallel=False)
        return apply_agg(df)
    elif mode == "eager_par":
        df = plfw.read_fwf_pl(filename, specs, parallel=True)
        return apply_agg(df)
    elif mode == "lazy_seq":
        return (
            plfw.scan_fwf_pl(filename, specs, parallel=False)
            .filter(pl.col("col_30") > 10**10)
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
            plfw.scan_fwf_pl(filename, specs, parallel=True)
            .filter(pl.col("col_30") > 10**10)
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
