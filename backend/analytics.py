from __future__ import annotations
import math
import re
from functools import lru_cache
from typing import Dict
import numpy as np
import pandas as pd

CRIME_COLUMNS = ["Rape", "K&A", "DD", "AoW", "AoM", "DV", "WT"]
DEFAULT_WEIGHTS = {c: 1 / len(CRIME_COLUMNS) for c in CRIME_COLUMNS}
ALIASES = {
    "state": ["state", "state/ut", "state/uts", "state_ut", "state/ut name", "states"],
    "year": ["year", "yr"],
    "Rape": ["rape", "rape cases"],
    "K&A": ["k&a", "k & a", "kidnapping and abduction", "kidnapping & abduction", "kidnapping_abduction", "ka"],
    "DD": ["dd", "dowry deaths", "dowry_deaths", "dowry death"],
    "AoW": ["aow", "assault on women", "assault_on_women", "assault against women"],
    "AoM": ["aom", "assault on minors", "assault_on_minors", "assault against minors"],
    "DV": ["dv", "domestic violence", "domestic_violence"],
    "WT": ["wt", "women trafficking", "women_trafficking", "trafficking of women"],
}

STATE_ALIASES = {
    "andaman and nicobar islands": "Andaman and Nicobar Islands",
    "a n islands": "Andaman and Nicobar Islands",
    "a n island": "Andaman and Nicobar Islands",
    "andaman nicobar islands": "Andaman and Nicobar Islands",
    "andaman and nicobar": "Andaman and Nicobar Islands",
    "andaman & nicobar islands": "Andaman and Nicobar Islands",
    "andhra pradesh": "Andhra Pradesh",
    "arunachal pradesh": "Arunachal Pradesh",
    "assam": "Assam",
    "bihar": "Bihar",
    "chandigarh": "Chandigarh",
    "chhattisgarh": "Chhattisgarh",
    "chattisgarh": "Chhattisgarh",
    # Keep DNH and Daman & Diu as separate source jurisdictions; the CSV reports them separately.
    "dadra and nagar haveli": "Dadra and Nagar Haveli",
    "dadra nagar haveli": "Dadra and Nagar Haveli",
    "d & n haveli": "Dadra and Nagar Haveli",
    "d n haveli": "Dadra and Nagar Haveli",
    "d and n haveli": "Dadra and Nagar Haveli",
    "d n haveli and daman diu": "Dadra and Nagar Haveli",
    "daman and diu": "Daman and Diu",
    "daman diu": "Daman and Diu",
    "daman & diu": "Daman and Diu",
    "delhi": "Delhi",
    "delhi ut": "Delhi",
    "nct of delhi": "Delhi",
    "goa": "Goa",
    "gujarat": "Gujarat",
    "haryana": "Haryana",
    "himachal pradesh": "Himachal Pradesh",
    "jammu kashmir": "Jammu and Kashmir",
    "jammu & kashmir": "Jammu and Kashmir",
    "jammu and kashmir": "Jammu and Kashmir",
    "jharkhand": "Jharkhand",
    "karnataka": "Karnataka",
    "kerala": "Kerala",
    "ladakh": "Ladakh",
    "lakshadweep": "Lakshadweep",
    "madhya pradesh": "Madhya Pradesh",
    "maharashtra": "Maharashtra",
    "manipur": "Manipur",
    "meghalaya": "Meghalaya",
    "mizoram": "Mizoram",
    "nagaland": "Nagaland",
    "odisha": "Odisha",
    "orissa": "Odisha",
    "puducherry": "Puducherry",
    "pondicherry": "Puducherry",
    "punjab": "Punjab",
    "rajasthan": "Rajasthan",
    "sikkim": "Sikkim",
    "tamil nadu": "Tamil Nadu",
    "telangana": "Telangana",
    "tripura": "Tripura",
    "uttar pradesh": "Uttar Pradesh",
    "uttarakhand": "Uttarakhand",
    "uttaranchal": "Uttarakhand",
    "west bengal": "West Bengal",
}


def norm_key(s):
    return re.sub(r"[^a-z0-9]+", " ", str(s).strip().lower()).strip()


def normalize_state(s):
    k = norm_key(s)
    return STATE_ALIASES.get(k, str(s).strip().title())


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    lookup = {norm_key(c): c for c in df.columns}
    rename = {}
    for target, aliases in ALIASES.items():
        for a in [target] + aliases:
            if norm_key(a) in lookup:
                rename[lookup[norm_key(a)]] = target
                break
    out = df.rename(columns=rename).copy()
    missing = [c for c in ["state", "year"] + CRIME_COLUMNS if c not in out.columns]
    if missing:
        raise ValueError("Missing required columns: " + ", ".join(missing))
    return out[["state", "year"] + CRIME_COLUMNS]


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df = canonicalize_columns(df)
    df["state"] = df["state"].astype(str).map(normalize_state)
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    for c in CRIME_COLUMNS:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["state", "year"]).copy()
    df["year"] = df["year"].astype(int)
    df[CRIME_COLUMNS] = df[CRIME_COLUMNS].fillna(0).clip(lower=0)
    # Keep one observation per State-Year after state-name normalization.
    df = df.groupby(["state", "year"], as_index=False)[CRIME_COLUMNS].sum()
    df = df[(df["year"] >= 1900) & (df["year"] <= 2100)].copy()
    df["Total Crime"] = df[CRIME_COLUMNS].sum(axis=1)
    return df.sort_values(["state", "year"]).reset_index(drop=True)


def growth_table(df: pd.DataFrame, crime: str = "Total Crime") -> pd.DataFrame:
    if crime != "Total Crime" and crime not in CRIME_COLUMNS:
        raise ValueError(f"Unknown crime category: {crime}")
    x = df.sort_values(["state", "year"]).copy()
    x["Growth Rate %"] = (
        x.groupby("state")[crime]
        .pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan)
        * 100
    )
    x["Recent Average"] = x.groupby("state")[crime].transform(
        lambda s: s.rolling(3, min_periods=1).mean()
    )
    # First-year / zero-denominator changes remain undefined (NaN) and are serialized as null.
    return x



def scoped_growth_table(df: pd.DataFrame, crime: str = "Total Crime", state: str = "All") -> pd.DataFrame:
    """Return year-over-year growth for a selected scope.

    For All India, aggregate the selected crime across all State/UT records first,
    then calculate year-over-year growth. Averaging State/UT percentage changes
    would overweight small jurisdictions and can produce misleading extreme values.
    For a selected State/UT, use that state's chronological series.
    """
    if state == "All":
        x = df.groupby("year", as_index=False)[crime].sum().sort_values("year")
        x["Growth Rate %"] = (
            x[crime].pct_change(fill_method=None)
            .replace([np.inf, -np.inf], np.nan)
            * 100
        )
        return x
    x = growth_table(df, crime)
    return x[x.state == state][["year", "Growth Rate %"]].sort_values("year")

def growth_statistics(df: pd.DataFrame, crime: str = "Total Crime", state: str | None = None) -> pd.DataFrame:
    return growth_table(df, crime)[
        lambda x: x["state"].eq(state) if state and state != "All" else pd.Series(True, index=x.index)
    ].copy()


def percentile_score(series: pd.Series) -> pd.Series:
    """Convert state crime counts into a 0-100 relative-position score.

    Using percentile ranks rather than raw min-max prevents very small jurisdictions
    from collapsing to a visually misleading CRI of 0. The lowest observed state/year
    still receives a small positive score (100/n) and the highest receives 100.
    This makes CRI a relative risk-position index, not a claim of zero crime.
    """
    s = pd.to_numeric(series, errors="coerce")
    n = int(s.notna().sum())
    if n == 0:
        return pd.Series(np.nan, index=series.index, dtype=float)
    if n == 1:
        return pd.Series(100.0, index=series.index, dtype=float)
    ranks = s.rank(method="average", ascending=True, na_option="keep")
    return ranks / n * 100.0


def cri_for_year(df: pd.DataFrame, year: int, weights: Dict[str, float] | None = None) -> pd.DataFrame:
    weights = weights or DEFAULT_WEIGHTS
    x = df[df.year == year].copy()
    if x.empty:
        return x
    # Percentile-normalize each crime category across the State/UTs in the selected year.
    # This avoids the previous min-max edge case where a small but nonzero jurisdiction
    # displayed CRI ~= 0 simply because it sat near the category minima.
    for c in CRIME_COLUMNS:
        x[c + "_norm"] = percentile_score(x[c])
    x["CRI"] = sum(x[c + "_norm"] * float(weights.get(c, 0)) for c in CRIME_COLUMNS)
    q1, q2 = x["CRI"].quantile([1 / 3, 2 / 3]).tolist()
    x["Risk"] = np.where(x.CRI <= q1, "Low", np.where(x.CRI <= q2, "Medium", "High"))
    return x.sort_values("CRI", ascending=False)


def temporal_summary(df: pd.DataFrame, state: str | None = None, crime: str = "Total Crime"):
    x = df if not state or state == "All" else df[df.state == state]
    return x.groupby("year", as_index=False)[crime].sum().sort_values("year")


def correlation(df: pd.DataFrame):
    return df[CRIME_COLUMNS + ["Total Crime"]].corr().round(4).to_dict()


def _mean_growth(values: pd.Series) -> float:
    g = values.pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan).dropna()
    if g.empty:
        return 0.0
    # Keep the literal arithmetic mean of valid historical year-over-year growth values.
    # No silent clipping is applied, so the displayed growth statistic remains descriptive.
    return float(g.mean())


DAMPING_CANDIDATES = (0.20, 0.35, 0.50, 0.65, 0.80, 1.00)

def _project_from_training(values: pd.Series, horizon: int, damping: float) -> list[float]:
    """Transparent damped-growth projection from a completed training window."""
    if len(values) < 3:
        return []
    growth = _mean_growth(values)
    baseline = max(0.0, float(values.tail(3).mean()))
    out = []
    value = baseline
    for step in range(1, horizon + 1):
        effective_growth = growth * (damping ** (step - 1))
        effective_growth = float(np.clip(effective_growth, -0.75, 0.75))
        value = max(0.0, value * (1.0 + effective_growth))
        out.append(float(value))
    return out


def _mean_growth_array(values: np.ndarray) -> float:
    if values.size < 2:
        return 0.0
    prev = values[:-1]
    curr = values[1:]
    valid = np.isfinite(prev) & np.isfinite(curr) & (prev != 0)
    if not valid.any():
        return 0.0
    growth = curr[valid] / prev[valid] - 1.0
    return float(np.mean(growth)) if growth.size else 0.0


def _project_array(values: np.ndarray, horizon: int, damping: float) -> np.ndarray:
    if values.size < 3:
        return np.empty(0, dtype=float)
    growth = _mean_growth_array(values)
    value = max(0.0, float(np.mean(values[-3:])))
    out = []
    for step in range(1, horizon + 1):
        effective_growth = float(np.clip(growth * (damping ** (step - 1)), -0.75, 0.75))
        value = max(0.0, value * (1.0 + effective_growth))
        out.append(value)
    return np.asarray(out, dtype=float)


@lru_cache(maxsize=2048)
def _select_damping_cached(values_tuple: tuple[float, ...], horizon: int) -> tuple[float, float | None]:
    values = np.asarray(values_tuple, dtype=float)
    n = len(values)
    if n < 5:
        return 0.50, None
    eval_h = max(1, min(int(horizon), 3))
    scores = []
    # Backtest only the most recent validation origins. This keeps the method responsive
    # for the dashboard while still using multiple unseen historical periods.
    start = max(3, n - eval_h - 8)
    end = n - eval_h + 1
    for damping in DAMPING_CANDIDATES:
        errors = []
        for i in range(start, end):
            pred = _project_array(values[:i], eval_h, damping)
            actual = values[i:i + eval_h]
            if pred.size == actual.size:
                errors.extend(np.abs(pred - actual).tolist())
        score = float(np.mean(errors)) if errors else math.inf
        scores.append((score, damping))
    scores.sort(key=lambda x: (x[0], -x[1]))
    best_score, best_damping = scores[0]
    return float(best_damping), (None if not math.isfinite(best_score) else float(best_score))


def _select_damping(values: pd.Series, horizon: int) -> tuple[float, float | None]:
    return _select_damping_cached(tuple(float(v) for v in values.to_numpy(dtype=float)), int(horizon))


def _series_for_scope(df: pd.DataFrame, state: str, crime: str) -> pd.DataFrame:
    """Return one chronological series for a state or the national aggregate."""
    if state == "All":
        x = (
            df.groupby("year", as_index=False)[crime]
            .sum()
            .sort_values("year")
        )
    else:
        x = df[df.state == state][["year", crime]].sort_values("year").copy()
    return x.reset_index(drop=True)


def forecast(df: pd.DataFrame, state: str, crime: str = "Total Crime", horizon: int = 3, origin_year: int | None = None):
    if crime != "Total Crime" and crime not in CRIME_COLUMNS:
        raise ValueError(f"Unknown crime category: {crime}")
    if horizon < 1 or horizon > 50:
        raise ValueError("Forecast horizon must be between 1 and 50 years.")
    if state != "All" and state not in set(df.state.unique()):
        raise ValueError(f"Unknown state: {state}")

    available = _series_for_scope(df, state, crime)
    if available.empty:
        return {"state": state, "crime": crime, "history": [], "forecast": [], "method": "no data"}

    max_year = int(available.year.max())
    min_year = int(available.year.min())
    requested_origin = max_year if origin_year is None else int(origin_year)
    origin_candidates = available[available.year <= requested_origin]
    if origin_candidates.empty:
        raise ValueError(f"No observations for {state} on or before {requested_origin}.")
    if len(origin_candidates) < 3:
        raise ValueError("Forecast origin requires at least 3 historical observations so the recent 3-year baseline is defined.")
    origin_year = int(origin_candidates.year.max())

    x = origin_candidates[["year", crime]].copy().sort_values("year")
    x["Growth Rate %"] = (
        x[crime].pct_change(fill_method=None)
        .replace([np.inf, -np.inf], np.nan) * 100
    )

    avg_growth = _mean_growth(x[crime])
    recent_average = float(x[crime].tail(3).mean())
    historical_average = float(x[crime].mean())
    baseline = max(0.0, recent_average)

    damping, calibration_mae = _select_damping(x[crime], horizon)
    # The displayed growth statistic remains descriptive. Forecasting uses the same
    # historical growth signal, but backtest-selected damping reduces runaway compounding.
    projected = _project_from_training(x[crime], horizon, damping)
    preds = []
    for i, value in enumerate(projected, start=1):
        yr = origin_year + i
        actual_row = available[available.year == yr]
        actual = float(actual_row[crime].iloc[0]) if not actual_row.empty else None
        effective_growth = float(np.clip(avg_growth * (damping ** (i - 1)), -0.75, 0.75))
        preds.append({
            "year": yr,
            "predicted": round(value, 2),
            "actual": actual,
            "effective_growth_rate": round(effective_growth * 100, 4),
        })

    # Rolling-origin one-step MAE using the selected robust damping factor.
    errors = []
    values = x[crime].to_numpy(float)
    for i in range(3, len(values)):
        prior = pd.Series(values[:i])
        one_step = _project_from_training(prior, 1, damping)
        if one_step:
            errors.append(abs(values[i] - one_step[0]))
    mae = float(np.mean(errors)) if errors else None

    forecast_end_value = float(preds[-1]["predicted"]) if preds else None
    projected_change_pct = ((forecast_end_value / baseline) - 1.0) * 100 if preds and baseline > 0 else None

    history_frame = x[["year", crime, "Growth Rate %"]].copy()
    history_frame["Growth Rate %"] = history_frame["Growth Rate %"].astype(object).where(pd.notna(history_frame["Growth Rate %"]), None)

    return {
        "state": state,
        "scope_label": "All India" if state == "All" else state,
        "crime": crime,
        "requested_origin_year": requested_origin,
        "origin_year": origin_year,
        "min_observed_year": min_year,
        "max_observed_year": max_year,
        "history": history_frame.to_dict("records"),
        "forecast": preds,
        "average": round(historical_average, 2),
        "recent_average": round(recent_average, 2),
        "forecast_baseline": round(baseline, 2),
        "average_growth_rate": round(avg_growth * 100, 2),
        "forecast_growth_rate": round(avg_growth * 100, 2),
        "damping_factor": round(damping, 2),
        "calibration_mae": round(calibration_mae, 2) if calibration_mae is not None else None,
        "forecast_end_year": int(origin_year + horizon),
        "forecast_end_value": round(forecast_end_value, 2) if forecast_end_value is not None else None,
        "projected_change_pct": round(projected_change_pct, 2) if projected_change_pct is not None else None,
        "growth_observations": int(x["Growth Rate %"].notna().sum()),
        "baseline_observations": int(x[crime].tail(3).shape[0]),
        "backtest_mae": round(mae, 2) if mae is not None else None,
        "method": (
            "Training window ends at the forecast-origin year. The recent 3-observation average is the baseline; "
            "the arithmetic mean of historical YoY growth is used with a damping factor selected by rolling-origin backtesting "
            f"(damping={damping:.2f}) so the projection is less prone to runaway compounding over longer horizons."
        ),
    }


def overview(df, year, crime="Total Crime"):
    if crime != "Total Crime" and crime not in CRIME_COLUMNS:
        raise ValueError(f"Unknown crime category: {crime}")
    c = cri_for_year(df, year)
    ydf = df[df.year == year]
    total = float(ydf[crime].sum()) if not ydf.empty else 0
    g = scoped_growth_table(df, crime, "All")
    yg = g[g.year == year]
    return {
        "year": year,
        "crime": crime,
        "selected_crime_total": round(total, 2),
        "avg_cri": round(float(c.CRI.mean()), 2) if not c.empty else 0,
        "high_risk": int((c.Risk == "High").sum()) if not c.empty else 0,
        "avg_growth": round(float(yg["Growth Rate %"].iloc[0]), 2) if not yg.empty else 0,
    }
