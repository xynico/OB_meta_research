from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

from common import levels, load_model, models, run_path_cell, select


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=models())
    parser.add_argument("--methods", nargs="+", choices=["mean", "mice_pmm"], default=["mean", "mice_pmm"])
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--stop", type=float, default=0.9)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output", type=Path, default=Path("output/path_b.csv"))
    args = parser.parse_args()
    rows = []
    for model in args.models:
        source = load_model(model)
        for method in args.methods:
            for rid in levels(args.start, args.stop, args.step):
                for ccb in levels(args.start, args.stop, args.step):
                    chosen = select(source, rid, ccb)
                    try:
                        fit = run_path_cell(chosen, method, "path_B", args.m, args.seed)
                        fit = fit.assign(model=model, method=method, rid_threshold=rid, ccb_threshold=ccb,
                                         n_analysis_rows=len(chosen), status="success")
                        rows.append(fit)
                    except Exception as error:
                        rows.append(pd.DataFrame([{"model": model, "method": method, "rid_threshold": rid,
                            "ccb_threshold": ccb, "n_analysis_rows": len(chosen), "analysis": "path_B",
                            "status": "failed", "error": str(error)}]))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(rows, ignore_index=True).to_csv(args.output, index=False)


if __name__ == "__main__":
    main()

