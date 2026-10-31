from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import rstar


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", nargs="+", type=Path)
    parser.add_argument("--output", type=Path, default=Path("output/rstar.csv"))
    args = parser.parse_args()
    data = pd.concat([pd.read_csv(path) for path in args.inputs], ignore_index=True)
    keys = ["analysis", "model", "method", "rid_threshold", "ccb_threshold", "n_analysis_rows"]
    rows = []
    usable = data[data["status"].eq("success") & data["path"].notna()].copy()
    for values, group in usable.groupby(keys, dropna=False):
        value, count = rstar(group)
        row = dict(zip(keys, values))
        row.update(r_star=value, n_paths_used=count, meaningful=bool(count == 7 and row["n_analysis_rows"] >= 30))
        rows.append(row)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

