from __future__ import annotations

import subprocess
from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy.stats import t


ROOT = Path(__file__).resolve().parents[1]
ITEMS = [
    "remote_work_intensity",
    *[f"autonomy_{i}" for i in range(1, 7)],
    *[f"stress_{i}" for i in range(1, 4)],
    *[f"satisfaction_{i}" for i in range(1, 4)],
    *[f"lmx_{i}" for i in range(1, 8)],
]
SCALES = {
    "remote_work_intensity": ["remote_work_intensity"],
    "autonomy": [f"autonomy_{i}" for i in range(1, 7)],
    "stress": [f"stress_{i}" for i in range(1, 4)],
    "satisfaction": [f"satisfaction_{i}" for i in range(1, 4)],
    "lmx": [f"lmx_{i}" for i in range(1, 8)],
}
PATH_ORDER = ["a", "c1", "c2", "c3", "b1", "b2", "b3"]
IDEAL = np.array([1, -1, 1, 1, -1, 1, 1], dtype=float)
PATH_A = [
    ("autonomy", ["remote_work_intensity"], {"remote_work_intensity": "a"}),
    ("stress", ["autonomy", "remote_work_intensity"], {"autonomy": "b1", "remote_work_intensity": "c1"}),
    ("satisfaction", ["autonomy", "remote_work_intensity"], {"autonomy": "b2", "remote_work_intensity": "c2"}),
    ("lmx", ["autonomy", "remote_work_intensity"], {"autonomy": "b3", "remote_work_intensity": "c3"}),
]
PATH_B = [
    ("autonomy", ["remote_work_intensity"], {"remote_work_intensity": "a"}),
    ("stress", ["remote_work_intensity"], {"remote_work_intensity": "c1"}),
    ("satisfaction", ["remote_work_intensity"], {"remote_work_intensity": "c2"}),
    ("lmx", ["remote_work_intensity"], {"remote_work_intensity": "c3"}),
    ("stress", ["autonomy"], {"autonomy": "b1"}),
    ("satisfaction", ["autonomy"], {"autonomy": "b2"}),
    ("lmx", ["autonomy"], {"autonomy": "b3"}),
]


def models() -> list[str]:
    return sorted(path.name for path in (ROOT / "results" / "llm").iterdir() if path.is_dir())


def levels(start: float, stop: float, step: float) -> list[float]:
    return [round(value, 2) for value in np.arange(start, stop + step / 2, step)]


def load_model(model: str) -> pd.DataFrame:
    base = ROOT / "results" / "llm" / model
    evidence = pd.read_csv(base / "evidence.csv")
    ratings = pd.read_csv(base / "ratings.csv")
    if evidence["post_id"].duplicated().any() or ratings["post_id"].duplicated().any():
        raise ValueError(f"duplicate post_id in {model}")
    evidence_columns = ["post_id", "stage_index", "rid", "ccb", "breadth_adjusted_density"]
    data = evidence[evidence_columns].merge(ratings, on="post_id", validate="one_to_one")
    numeric = [*evidence_columns[1:], *ITEMS, *[column for column in ratings if column.endswith("_overall")]]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    return data


def select(data: pd.DataFrame, rid: float, ccb: float) -> pd.DataFrame:
    return data[(data["rid"] > rid) & (data["ccb"] > ccb)].copy()


def mean_impute(data: pd.DataFrame) -> list[pd.DataFrame]:
    missing = [column for column in ITEMS if data[column].notna().sum() == 0]
    if missing:
        raise ValueError("variance_insufficient: " + ", ".join(missing))
    result = data.copy()
    result[ITEMS] = result[ITEMS].fillna(result[ITEMS].mean())
    return [result]


def mice_pmm_impute(
    data: pd.DataFrame, m: int = 5, seed: int = 20261013, iterations: int = 15, donors: int = 5
) -> list[pd.DataFrame]:
    scores = data[ITEMS].apply(pd.to_numeric, errors="coerce")
    missing_columns = scores.columns[scores.notna().sum().eq(0)].tolist()
    if missing_columns:
        raise ValueError("variance_insufficient: " + ", ".join(missing_columns))
    auxiliary_columns = ["stage_index", "rid", "ccb", "breadth_adjusted_density"]
    auxiliary = data[auxiliary_columns].fillna(data[auxiliary_columns].median()).fillna(0.0)
    auxiliary = (auxiliary - auxiliary.mean()) / auxiliary.std(ddof=0).replace(0, 1)
    missing = scores.isna().to_numpy()
    initial = scores.fillna(scores.mean()).to_numpy(float)
    auxiliary = auxiliary.to_numpy(float)
    lower = np.array([0.0 if column == "remote_work_intensity" else 1.0 for column in ITEMS])
    outputs = []
    for chain in range(m):
        rng = np.random.default_rng(seed + 1009 * (chain + 1))
        completed = initial.copy()
        for j in range(len(ITEMS)):
            donors_observed = scores.iloc[:, j].dropna().to_numpy(float)
            completed[missing[:, j], j] = rng.choice(donors_observed, missing[:, j].sum())
        for _ in range(iterations):
            for j in rng.permutation(len(ITEMS)):
                miss = missing[:, j]
                if not miss.any():
                    continue
                observed = ~miss
                x = np.column_stack([np.ones(len(data)), np.delete(completed, j, axis=1), auxiliary])
                xo, xm, y = x[observed], x[miss], completed[observed, j]
                ridge = np.eye(xo.shape[1]) * 1e-6
                ridge[0, 0] = 0
                inverse = np.linalg.pinv(xo.T @ xo + ridge)
                beta = inverse @ xo.T @ y
                residual = y - xo @ beta
                df = max(len(y) - xo.shape[1], 1)
                variance = float(residual @ residual / rng.chisquare(df))
                beta = rng.multivariate_normal(beta, variance * inverse, check_valid="ignore")
                predicted_observed, predicted_missing = xo @ beta, xm @ beta
                order = np.argsort(predicted_observed)
                sorted_predictions = predicted_observed[order]
                sorted_values = y[order]
                donor_values = np.empty(len(predicted_missing))
                for i, target in enumerate(predicted_missing):
                    position = int(np.searchsorted(sorted_predictions, target))
                    low = max(0, position - donors)
                    high = min(len(sorted_predictions), position + donors + 1)
                    candidates = np.arange(low, high)
                    nearest = candidates[np.argsort(np.abs(sorted_predictions[candidates] - target))[:donors]]
                    donor_values[i] = sorted_values[rng.choice(nearest)]
                completed[miss, j] = donor_values
        result = data.copy()
        result[ITEMS] = np.clip(completed, lower, 7.0)
        outputs.append(result)
    return outputs


def impute(data: pd.DataFrame, method: str, m: int, seed: int) -> list[pd.DataFrame]:
    if method == "mean":
        return mean_impute(data)
    if method == "mice_pmm":
        return mice_pmm_impute(data, m=m, seed=seed)
    raise ValueError(method)


def scale_scores(data: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame({"post_id": data["post_id"]})
    for scale, columns in SCALES.items():
        result[scale] = data[columns].mean(axis=1)
    return result


def standardize(data: pd.DataFrame) -> pd.DataFrame:
    result = data.copy()
    for column in SCALES:
        sd = result[column].std(ddof=1)
        if not np.isfinite(sd) or sd == 0:
            raise ValueError(f"variance_insufficient: {column}")
        result[column] = (result[column] - result[column].mean()) / sd
    return result


def fit_paths(data: pd.DataFrame, equations: list[tuple], analysis: str) -> pd.DataFrame:
    rows = []
    for equation_id, (outcome, predictors, labels) in enumerate(equations, 1):
        frame = data[[outcome, *predictors]].dropna()
        if len(frame) <= len(predictors) + 1:
            raise ValueError("sample_insufficient")
        fitted = sm.OLS(frame[outcome], sm.add_constant(frame[predictors], has_constant="add")).fit()
        ci = fitted.conf_int(alpha=0.05)
        for predictor in predictors:
            rows.append({
                "analysis": analysis,
                "equation": equation_id,
                "path": labels[predictor],
                "outcome": outcome,
                "predictor": predictor,
                "estimate": fitted.params[predictor],
                "std_error": fitted.bse[predictor],
                "p_value": fitted.pvalues[predictor],
                "ci95_low": ci.loc[predictor, 0],
                "ci95_high": ci.loc[predictor, 1],
                "n": int(fitted.nobs),
                "r2": fitted.rsquared,
                "adj_r2": fitted.rsquared_adj,
                "aic": fitted.aic,
                "bic": fitted.bic,
            })
    return pd.DataFrame(rows)


def pool(frame: pd.DataFrame, keys: list[str]) -> pd.DataFrame:
    rows = []
    for values, group in frame.groupby(keys, sort=False):
        values = values if isinstance(values, tuple) else (values,)
        q = group["estimate"].to_numpy(float)
        u = np.square(group["std_error"].to_numpy(float))
        count = len(q)
        qbar, ubar = q.mean(), u.mean()
        between = q.var(ddof=1) if count > 1 else 0.0
        total = ubar + (1 + 1 / count) * between
        df = (count - 1) * (1 + ubar / ((1 + 1 / count) * between)) ** 2 if count > 1 and between > 0 else np.inf
        critical = t.ppf(0.975, df) if np.isfinite(df) else 1.959963984540054
        row = dict(zip(keys, values))
        standard_error = np.sqrt(total)
        p_value = 2 * t.sf(abs(qbar / standard_error), df) if standard_error > 0 else np.nan
        row.update(estimate=qbar, std_error=standard_error, p_value=p_value,
                   ci95_low=qbar - critical * standard_error, ci95_high=qbar + critical * standard_error,
                   df=df, m=count)
        for metric in ["n", "r2", "adj_r2", "aic", "bic"]:
            if metric in group:
                row[metric] = pd.to_numeric(group[metric], errors="coerce").mean()
        rows.append(row)
    return pd.DataFrame(rows)


def run_path_cell(data: pd.DataFrame, method: str, analysis: str, m: int, seed: int) -> pd.DataFrame:
    equations = PATH_A if analysis == "path_A" else PATH_B
    fits = []
    for index, completed in enumerate(impute(data, method, m, seed), 1):
        fit = fit_paths(standardize(scale_scores(completed)), equations, analysis)
        fit["imputation"] = index
        fits.append(fit)
    return pool(pd.concat(fits, ignore_index=True), ["analysis", "path", "outcome", "predictor"])


def fit_sem(data: pd.DataFrame, output: Path, method: str, m: int, seed: int) -> pd.DataFrame:
    output.mkdir(parents=True, exist_ok=True)
    fits = []
    for index, completed in enumerate(impute(data, method, m, seed), 1):
        source = output / f"imputation_{index:02d}.csv"
        destination = output / f"imputation_{index:02d}_sem"
        completed[["post_id", *ITEMS]].to_csv(source, index=False)
        command = ["Rscript", str(Path(__file__).with_name("fit_sem.R")), str(source), str(destination)]
        subprocess.run(command, check=True, capture_output=True, text=True)
        frame = pd.read_csv(destination / "standardized_paths.csv")
        frame["imputation"] = index
        fits.append(frame)
    return pool(pd.concat(fits, ignore_index=True), ["analysis", "path", "outcome", "predictor"])


def rstar(coefficients: pd.DataFrame) -> tuple[float, int]:
    values = coefficients.drop_duplicates("path").set_index("path")["estimate"].reindex(PATH_ORDER).to_numpy(float)
    valid = np.isfinite(values)
    if valid.sum() < 2 or np.isclose(values[valid].std(ddof=1), 0):
        return np.nan, int(valid.sum())
    return float(np.corrcoef(values[valid], IDEAL[valid])[0, 1]), int(valid.sum())

