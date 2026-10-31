from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import pearsonr, spearmanr

from common import ITEMS, SCALES, impute, load_model, models, scale_scores


def alpha(frame: pd.DataFrame) -> float:
    frame = frame.dropna()
    if len(frame) < 3 or frame.shape[1] < 2:
        return np.nan
    total = frame.sum(axis=1).var(ddof=1)
    k = frame.shape[1]
    return k / (k - 1) * (1 - frame.var(ddof=1).sum() / total) if total > 0 else np.nan


def human_validity(model: str, model_data: pd.DataFrame) -> list[dict]:
    human = pd.read_csv(Path(__file__).resolve().parents[1] / "results" / "human" / "human_ratings.csv")
    human_mean = human.groupby("post_id", as_index=False).agg(
        remote_work_intensity=("remote_work_intensity", "mean"),
        autonomy=("autonomy_overall", "mean"), stress=("stress_overall", "mean"),
        satisfaction=("satisfaction_overall", "mean"), lmx=("lmx_overall", "mean"),
    )
    scored = scale_scores(model_data)
    for construct in ["autonomy", "stress", "satisfaction", "lmx"]:
        overall = f"{construct}_overall"
        if overall in model_data:
            scored[construct] = pd.to_numeric(model_data[overall], errors="coerce")
    scored = scored.merge(human_mean, on="post_id", suffixes=("_model", "_human"))
    rows = []
    for construct in SCALES:
        pair = scored[[f"{construct}_model", f"{construct}_human"]].dropna()
        if len(pair) >= 3 and pair.iloc[:, 0].nunique() > 1 and pair.iloc[:, 1].nunique() > 1:
            pr = pearsonr(pair.iloc[:, 0], pair.iloc[:, 1])
            sr = spearmanr(pair.iloc[:, 0], pair.iloc[:, 1])
            rows.append({"model": model, "construct": construct, "n": len(pair),
                         "pearson_r": pr.statistic, "pearson_p": pr.pvalue,
                         "spearman_rho": sr.statistic, "spearman_p": sr.pvalue})
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", nargs="*", default=models())
    parser.add_argument("--methods", nargs="+", choices=["mean", "mice_pmm"], default=["mean", "mice_pmm"])
    parser.add_argument("--m", type=int, default=5)
    parser.add_argument("--seed", type=int, default=20261013)
    parser.add_argument("--output-dir", type=Path, default=Path("output/validation"))
    args = parser.parse_args()
    alpha_rows, cfa_rows, validity_rows = [], [], []
    args.output_dir.mkdir(parents=True, exist_ok=True)
    for model in args.models:
        source = load_model(model)
        validity_rows.extend(human_validity(model, source))
        for method in args.methods:
            completed_sets = impute(source, method, args.m, args.seed)
            for index, completed in enumerate(completed_sets, 1):
                for construct, columns in SCALES.items():
                    if len(columns) > 1:
                        alpha_rows.append({"model": model, "method": method, "imputation": index,
                                           "construct": construct, "alpha": alpha(completed[columns])})
                source_csv = args.output_dir / f"{model}_{method}_{index:02d}_cfa_input.csv"
                destination = args.output_dir / f"{model}_{method}_{index:02d}_cfa"
                completed[["post_id", *ITEMS]].to_csv(source_csv, index=False)
                subprocess.run(["Rscript", str(Path(__file__).with_name("fit_cfa.R")),
                                str(source_csv), str(destination)], check=True)
                cfa = pd.read_csv(destination / "fit_indices.csv")
                cfa_rows.append(cfa.assign(model=model, method=method, imputation=index))
    alpha_data = pd.DataFrame(alpha_rows)
    alpha_data.to_csv(args.output_dir / "cronbach_alpha.csv", index=False)
    alpha_data.groupby(["model", "method", "construct"], as_index=False)["alpha"].mean().to_csv(
        args.output_dir / "cronbach_alpha_pooled.csv", index=False)
    pd.concat(cfa_rows, ignore_index=True).to_csv(args.output_dir / "cfa_fit_indices.csv", index=False)
    validity = pd.DataFrame(validity_rows)
    validity.to_csv(args.output_dir / "human_validity_by_construct.csv", index=False)
    macro = validity.groupby("model", as_index=False).agg(
        Vh_pearson=("pearson_r", lambda x: np.tanh(np.nanmean(np.arctanh(np.clip(x, -.999999, .999999))))),
        Vh_spearman=("spearman_rho", lambda x: np.tanh(np.nanmean(np.arctanh(np.clip(x, -.999999, .999999))))),
    )
    macro.to_csv(args.output_dir / "human_validity_by_model.csv", index=False)


if __name__ == "__main__":
    main()

