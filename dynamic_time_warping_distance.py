# NOTE: No DTW libraries are used. This is a custom DTW implementation.
# ============================================================

import time
import math
import ast
import pandas as pd

# -------------------------
# 1) INPUT PATH
# -------------------------
INPUT_XLSX = "dtw_test.xlsx"
# Output CSV path
OUTPUT_CSV = "dtw.csv"


# -------------------------
# 2) Helper: safe parsing of list-string
# -------------------------
def parse_series(s):
    """
    Converts a string like "[75.125, 75.32, 75.5]" to a list of floats.
    Uses ast.literal_eval for safety (no eval).
    """
    if isinstance(s, list):
        
        return [float(x) for x in s]
    if s is None or (isinstance(s, float) and math.isnan(s)):
        return []
    s = str(s).strip()
    if s == "":
        return []
    try:
        arr = ast.literal_eval(s)
    except Exception as e:
        raise ValueError(f"Failed to parse series: {s[:80]}...") from e

    if not isinstance(arr, (list, tuple)):
        raise ValueError(f"Parsed object is not a list/tuple: {type(arr)}")
    return [float(x) for x in arr]


# -------------------------
# 3) Euclidean local cost (1D)
# -------------------------
def euclidean_1d(a, b):
    # For scalars: sqrt((a-b)^2) == abs(a-b)
    return abs(a - b)


# -------------------------
# 4) Handmade DTW (DP) with O(min(n,m)) memory
# -------------------------
def dtw_distance(seq_a, seq_b):
    """
    Computes DTW distance between two 1D sequences using:
    - local cost = Euclidean distance between points
    - allowed steps: (i-1,j), (i,j-1), (i-1,j-1)
    Uses dynamic programming with two rows to reduce memory.
    """
    n = len(seq_a)
    m = len(seq_b)

    # Handle empty sequences (edge cases)
    if n == 0 and m == 0:
        return 0.0
    if n == 0:
        return float("inf")
    if m == 0:
        return float("inf")

    # Ensure we allocate the smaller dimension for memory efficiency:
    # We'll compute DP over the second axis (m) with two rows.
    prev = [float("inf")] * (m + 1)
    curr = [float("inf")] * (m + 1)

    prev[0] = float("inf")
    curr[0] = float("inf")

    prev[1:] = [float("inf")] * m

    dp00 = 0.0

    for i in range(1, n + 1):
        
        curr[0] = float("inf")
        for j in range(1, m + 1):
            cost = euclidean_1d(seq_a[i - 1], seq_b[j - 1])

            
            if i == 1 and j == 1:
                best_prev = dp00
            else:
                best_prev = min(prev[j],      
                                curr[j - 1],  
                                prev[j - 1])  

            curr[j] = cost + best_prev

        # swap rows
        prev, curr = curr, prev

    # dp(n,m) is in prev[m]
    return float(prev[m])


# -------------------------
# 5) Main: read, compute, write, time
# -------------------------
def main():
    # Read Excel
    df = pd.read_excel(INPUT_XLSX)

    expected_cols = set(df.columns.str.lower())
    col_id = None
    col_a = None
    col_b = None
    for c in df.columns:
        cl = c.lower()
        if cl == "id":
            col_id = c
        elif cl in ("series_a", "seq_a", "a", "s1"):
            col_a = c
        elif cl in ("series_b", "seq_b", "b", "s2"):
            col_b = c

    if col_id is None or col_a is None or col_b is None:
        raise ValueError(
            f"Could not identify required columns. Found columns: {list(df.columns)}\n"
            f"Need: id, series_a, series_b (or adjust the mapping in code)."
        )

    # Start timing ONLY the DTW computations over the whole dataset
    t0 = time.perf_counter()

    out_ids = []
    out_dists = []

    for _, row in df.iterrows():
        rid = int(row[col_id])
        seq_a = parse_series(row[col_a])
        seq_b = parse_series(row[col_b])

        dist = dtw_distance(seq_a, seq_b)

        out_ids.append(rid)
        out_dists.append(dist)

    t1 = time.perf_counter()
    total_seconds = t1 - t0

    # Create output df
    out_df = pd.DataFrame({
        "id": out_ids,
        "DTW distance": out_dists
    })

    # Write output CSV file
    out_df.to_csv(OUTPUT_CSV, index=False)

    # Print timing info
    print("============================================================")
    print(f"Processed {len(out_df)} rows")
    print(f"Output written to: {OUTPUT_CSV}")
    print(f"Total DTW computation time (seconds): {total_seconds:.6f}")
    print("============================================================")


if __name__ == "__main__":
    main()