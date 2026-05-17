import os

import numpy as np

STATES = [
    "AL",
    "AK",
    "AZ",
    "AR",
    "CA",
    "CO",
    "CT",
    "DE",
    "FL",
    "GA",
    "HI",
    "ID",
    "IL",
    "IN",
    "IA",
    "KS",
    "KY",
    "LA",
    "ME",
    "MD",
    "MA",
    "MI",
    "MN",
    "MS",
    "MO",
    "MT",
    "NE",
    "NV",
    "NH",
    "NJ",
    "NM",
    "NY",
    "NC",
    "ND",
    "OH",
    "OK",
    "OR",
    "PA",
    "RI",
    "SC",
    "SD",
    "TN",
    "TX",
    "UT",
    "VT",
    "VA",
    "WA",
    "WV",
    "WI",
    "WY",
]


def generate_large_fwf(filename, num_rows=200000, num_cols=200):
    print(f"Generating {filename} with {num_rows} rows and {num_cols} columns...")

    # We'll use 199 other columns + 1 'state' column
    # Total = 200

    types = (
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

    line_length = sum(widths)
    os.makedirs(os.path.dirname(filename), exist_ok=True)

    with open(filename, "w") as f:
        for i in range(num_rows):
            row_data = []
            for t, w in zip(types, widths):
                if t.startswith("i") or t.startswith("u"):
                    signed = t.startswith("i")
                    if "8" in t:
                        max_val = 127 if signed else 255
                    elif "16" in t:
                        max_val = 32767 if signed else 65535
                    elif "32" in t:
                        max_val = 2 * 10**9 if signed else 4 * 10**9
                    else:
                        max_val = 10**18
                    val = np.random.randint(0, max_val)
                    row_data.append(str(val).rjust(w))
                elif t.startswith("f"):
                    val = np.random.uniform(0, 10**6)
                    row_data.append(f"{val:{w}.4f}")
                elif t == "state":
                    row_data.append(np.random.choice(STATES))
                else:
                    val = "".join(
                        np.random.choice(list("ABCDEFGHIJKLMNOPQRSTUVWXYZ"), 5)
                    ).ljust(w)
                    row_data.append(val)
            f.write("".join(row_data) + "\n")
            if (i + 1) % 50000 == 0:
                print(f"Progress: {i + 1}/{num_rows} rows written.")

    print(f"Done. File size: {os.path.getsize(filename) / (1024 * 1024):.2f} MB")
    print(f"Line length: {line_length}")


if __name__ == "__main__":
    generate_large_fwf("data/large_bench.fwf")
