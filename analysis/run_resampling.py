from __future__ import annotations

import argparse
import hashlib
import subprocess
from concurrent.futures import ProcessPoolExecutor
from itertools import product
from pathlib import Path

import pandas as pd

import numpy as np

from common import SCALES, fit_sem, impute, levels, load_model, rstar, run_path_cell, select


def stable_seed(master: int, *parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, (master, *parts))).encode()).digest()
    return int.from_bytes(digest[:4], "big")


def alpha(frame: pd.DataFrame) -> float:
    frame = frame.dropna()
    if len(frame) < 3 or frame.shape[1] < 2:
        return np.nan
    total = frame.sum(axis=1).var(ddof=1)
    k = frame.shape[1]
    return k / (k - 1) * (1 - frame.var(ddof=1).sum() / total) if total > 0 else np.nan


def measurement(sample: pd.DataFrame, method: str, m: int, seed: int, destination: Path) -> list[dict]:
    rows = []
    for index, completed in enumerate(impute(sample, method, m, seed), 1):
        for construct, columns in SCALES.items():
            if len(columns) > 1:
                rows.append({"record_type": "alpha", "construct": construct, "metric": "cronbach_alpha",
                             "value": alpha(completed[columns]), "imputation": index})
        source = destination / f"measurement_imputation_{index:02d}.csv"
        output = destination / f"measurement_imputation_{index:02d}_cfa"
        destination.mkdir(parents=True, exist_ok=True)
        completed.to_csv(source, index=False)
        try:
            subprocess.run(["Rscript", str(Path(__file__).with_name("fit_cfa.R")), str(source), str(output)],
                           check=True, capture_output=True, text=True)
            for item in pd.read_csv(output / "fit_indices.csv").itertuples(index=False):
                rows.append({"record_type": "cfa", "construct": item.structure, "metric": item.metric,
                             "value": item.value, "imputation": index})
        except Exception as error:
            rows.append({"record_type": "cfa", "construct": "four_factor", "metric": "status",
                         "value": np.nan, "imputation": index, "error": str(error)})
    return rows


def task(spec: tuple) -> dict[str, list[dict]]:
    model, method, n, repetition, analyses, start, stop, step, m, seed, work = spec
    source = load_model(model)
    if n > len(source):
        return {"paths": [], "measurement": []}
    rng = np.random.default_rng(np.random.SeedSequence([seed, repetition]))
    sampled = source.iloc[rng.permutation(len(source))[:n]].copy()
    analysis_seed = stable_seed(seed, method, model, n, repetition)
    measurement_rows = measurement(
        sampled, method, m, analysis_seed,
        Path(work) / model / method / f"n{n}_rep{repetition:02d}" / "measurement",
    )
    for row in measurement_rows:
        row.update(model=model, method=method, n=n, repetition=repetition)
    rows = []
    for rid, ccb in product(levels(start, stop, step), repeat=2):
        selected = select(sampled, rid, ccb)
        for analysis in analyses:
            try:
                if analysis == "sem":
                    cell = Path(work) / model / method / f"n{n}_rep{repetition:02d}_rid{rid:.2f}_ccb{ccb:.2f}"
                    cell_seed = analysis_seed + int(rid * 10000) + int(ccb * 100)
                    coefficients = fit_sem(selected, cell, method, m, cell_seed)
                else:
                    cell_seed = analysis_seed + int(rid * 10000) + int(ccb * 100)
                    coefficients = run_path_cell(selected, method, analysis, m, cell_seed)
                value, count = rstar(coefficients)
                rows.append({"model": model, "method": method, "n": n, "repetition": repetition,
                             "analysis": analysis, "rid_threshold": rid, "ccb_threshold": ccb,
                             "n_analysis_rows": len(selected), "r_star": value,
                             "n_paths_used": count, "status": "success"})
            except Exception as error:
                rows.append({"model": model, "method": method, "n": n, "repetition": repetition,
                             "analysis": analysis, "rid_threshold": rid, "ccb_threshold": ccb,
                             "n_analysis_rows": len(selected), "r_star": None,
                             "n_paths_used": 0, "status": "failed", "error": str(error)})
    return {"paths": rows, "measurement": measurement_rows}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="+", required=True)
    parser.add_argument("--methods", nargs="+", choices=["mean", "mice_pmm"], default=["mean", "mice_pmm"])
    parser.add_argument("--analyses", nargs="+", choices=["sem", "path_A", "path_B"],
                        default=["sem", "path_A", "path_B"])
    parser.add_argument("--sample-sizes", nargs="+", type=int,
                        default=[50, 100, 500, 1000, 2000, 3000, 4000, 5000])
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--start", type=float, default=0.0)
    parser.add_argument("--stop", type=float, default=0.9)
    parser.add_argument("--step", type=float, default=0.05)
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261016)
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("output/resampling.csv"))
    parser.add_argument("--work", type=Path, default=Path("output/resampling_work"))
    args = parser.parse_args()
    specs = [
        (model, method, n, repetition, tuple(args.analyses), args.start, args.stop, args.step,
         args.m, args.seed, str(args.work))
        for model, method, n, repetition in product(
            args.models, args.methods, args.sample_sizes, range(1, args.repetitions + 1)
        )
    ]
    if args.jobs == 1:
        output = [task(spec) for spec in specs]
    else:
        with ProcessPoolExecutor(max_workers=args.jobs) as executor:
            output = list(executor.map(task, specs))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row for group in output for row in group["paths"]]).to_csv(args.output, index=False)
    measurement_output = args.output.with_name(f"{args.output.stem}_measurement{args.output.suffix}")
    pd.DataFrame([row for group in output for row in group["measurement"]]).to_csv(measurement_output, index=False)


if __name__ == "__main__":
    main()

