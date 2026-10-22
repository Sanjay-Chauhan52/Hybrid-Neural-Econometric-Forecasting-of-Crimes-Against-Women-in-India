"""
Serves the ensemble models trained by train_cri_ensemble.py and
train_forecast_ensemble.py. Drop the four files they produce
(cri_autoencoder.keras, cri_ensemble_meta.json, forecast_lstm.keras,
forecast_ensemble_meta.json) into <project_root>/models/.

CRI ensemble = PCA fusion of 3 signals (heuristic percentile sum, autoencoder
anomaly score, Mahalanobis statistical severity). See train_cri_ensemble.py
for the full rationale.

Forecast ensemble = inverse-backtest-error-weighted blend of an LSTM (deep
learning) and Damped Growth (baseline heuristic), per horizon step. See
train_forecast_ensemble.py for the full rationale and the actual backtested
accuracy numbers (saved in forecast_ensemble_meta.json under
backtest_mae_clean_states).

NOTE: the forecast ensemble was only trained/backtested on "Total Crime";
other crime categories aren't supported yet.
"""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from tensorflow import keras

from analytics import CRIME_COLUMNS, percentile_score, forecast as baseline_forecast

MODELS_DIR = Path(__file__).resolve().parents[1] / "models"

_cri_ae = None
_cri_meta = None
_fc_lstm = None
_fc_meta = None


def _load_cri():
    global _cri_ae, _cri_meta
    if _cri_ae is None:
        _cri_ae = keras.models.load_model(MODELS_DIR / "cri_autoencoder.keras")
        _cri_meta = json.loads((MODELS_DIR / "cri_ensemble_meta.json").read_text())
    return _cri_ae, _cri_meta


def _load_forecast():
    global _fc_lstm, _fc_meta
    if _fc_lstm is None:
        _fc_lstm = keras.models.load_model(MODELS_DIR / "forecast_lstm.keras")
        _fc_meta = json.loads((MODELS_DIR / "forecast_ensemble_meta.json").read_text())
    return _fc_lstm, _fc_meta


def models_available() -> dict:
    return {
        "cri_ensemble": (MODELS_DIR / "cri_autoencoder.keras").exists(),
        "forecast_ensemble": (MODELS_DIR / "forecast_lstm.keras").exists(),
    }


# --------------------------------------------------------------- CRI ensemble
def cri_for_year_ml(df: pd.DataFrame, year: int) -> pd.DataFrame:
    ae, meta = _load_cri()
    x = df[df.year == year].copy()
    if x.empty:
        return x

    for c in CRIME_COLUMNS:
        x[c + "_norm"] = percentile_score(x[c])
    feats = x[[c + "_norm" for c in CRIME_COLUMNS]].to_numpy(dtype="float64")

    # S1: heuristic
    s1 = feats.mean(axis=1)

    # S3: Mahalanobis distance within this year's rows
    mean = feats.mean(axis=0)
    cov = np.cov(feats, rowvar=False) + np.eye(len(CRIME_COLUMNS)) * 1e-6
    inv_cov = np.linalg.pinv(cov)
    diffs = feats - mean
    maha_raw = np.sqrt(np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs))
    s3 = percentile_score(pd.Series(maha_raw, index=x.index)).to_numpy()

    # S2: autoencoder reconstruction error
    X = (feats / 100.0).astype("float32")
    recon = ae(X, training=False).numpy()
    recon_err = np.mean((X - recon) ** 2, axis=1)
    s2 = percentile_score(pd.Series(recon_err, index=x.index)).to_numpy()

    S = np.stack([s1, s2, s3], axis=1)
    mean_, std_, pc1 = np.array(meta["pca_mean"]), np.array(meta["pca_std"]), np.array(meta["pca_pc1"])
    Sz = (S - mean_) / std_
    ensemble_raw = Sz @ pc1

    x["CRI_heuristic"] = s1
    x["CRI_anomaly_signal"] = s2
    x["CRI_statistical_signal"] = s3
    x["CRI"] = percentile_score(pd.Series(ensemble_raw, index=x.index))
    q1, q2 = x["CRI"].quantile([1 / 3, 2 / 3]).tolist()
    x["Risk"] = np.where(x.CRI <= q1, "Low", np.where(x.CRI <= q2, "Medium", "High"))
    return x.sort_values("CRI", ascending=False)


# --------------------------------------------------------- forecast ensemble
def _lstm_forecast_batch(model, meta: dict, state_list: list[str], horizon: int, origin_year: int = 2021) -> dict[str, np.ndarray]:
    window = meta["window"]
    cutoff = max(window, min(len(meta["trend_log_series_full"][state_list[0]]), origin_year - 2001 + 1))
    sids = np.array([[meta["state_to_idx"][s]] for s in state_list], dtype="int32")
    los = [meta["norm_stats"][s]["log_min"] for s in state_list]
    spans = [meta["norm_stats"][s]["log_span"] for s in state_list]
    bufs = [
        [(v - los[i]) / spans[i] for v in meta["trend_log_series_full"][s][:cutoff]][-window:]
        for i, s in enumerate(state_list)
    ]
    all_outputs = [[] for _ in state_list]

    for _ in range(horizon):
        seq_batch = np.array([b[-window:] for b in bufs], dtype="float32").reshape(len(state_list), window, 1)
        preds = model([seq_batch, sids], training=False).numpy().flatten()
        for i, p in enumerate(preds):
            val = float(p)
            bufs[i].append(val)
            all_outputs[i].append(max(0.0, np.expm1(val * spans[i] + los[i])))

    return {s: np.array(all_outputs[i]) for i, s in enumerate(state_list)}


def _lstm_forecast(model, meta, state, horizon, origin_year: int = 2021):
    return _lstm_forecast_batch(model, meta, [state], horizon, origin_year=origin_year)[state]


def _growth_forecast(df: pd.DataFrame, state: str, horizon: int, origin_year: int = 2021) -> np.ndarray:
    res = baseline_forecast(df, state, crime="Total Crime", horizon=horizon, origin_year=origin_year)
    return np.array([f["predicted"] for f in res["forecast"]])


def _growth_forecast_batch(df: pd.DataFrame, state_list: list[str], horizon: int, origin_year: int = 2021) -> dict[str, np.ndarray]:
    return {s: _growth_forecast(df, s, horizon, origin_year=origin_year) for s in state_list}


MODEL_INFO = {
    "growth": {
        "label": "Damped Growth (Baseline)",
        "description": (
            "The heuristic baseline model used in the Prediction tab. It takes the recent 3-year "
            "average as baseline and compounds historical mean YoY growth with rolling-origin calibrated "
            "damping. Serves as the transparent benchmark demonstrating whether advanced ML models actually add predictive value."
        ),
    },
    "lstm": {
        "label": "LSTM (Deep Learning)",
        "description": (
            "A recurrent neural network (TensorFlow) that reads the last 3 years of a state's "
            "crime totals and predicts the next year. It's trained globally across all 36 states "
            "at once, with a small learned 4-number 'fingerprint' per state (an embedding) so one "
            "shared model can still behave differently for Kerala vs. Bihar. Good at picking up "
            "nonlinear patterns and regime shifts."
        ),
    },
    "ensemble": {
        "label": "Ensemble (Combined)",
        "description": (
            "Blends the LSTM and Damped Growth forecasts, weighted per forecast-year by which of "
            "the two was actually more accurate at that distance out, measured on a real held-out "
            "backtest (2019-2021, never seen during training). Combines the baseline stability with "
            "deep learning pattern recognition."
        ),
    },
}


def _blend_predictions(lstm_arr: np.ndarray, growth_arr: np.ndarray, meta: dict, horizon: int) -> np.ndarray:
    out = []
    for h in range(1, horizon + 1):
        wkey = str(min(h, meta["max_backtested_horizon"]))
        weights = meta["weights_per_horizon"].get(wkey, {"lstm": 0.5, "growth": 0.5})
        out.append(weights["lstm"] * lstm_arr[h - 1] + weights["growth"] * growth_arr[h - 1])
    return np.array(out)


def _model_prediction(model_name: str, model, meta: dict, df: pd.DataFrame, state: str, horizon: int, origin_year: int = 2021) -> np.ndarray:
    if model_name == "lstm":
        return _lstm_forecast(model, meta, state, horizon, origin_year=origin_year)
    if model_name == "growth":
        return _growth_forecast(df, state, horizon, origin_year=origin_year)
    if model_name == "ensemble":
        a = _lstm_forecast(model, meta, state, horizon, origin_year=origin_year)
        b = _growth_forecast(df, state, horizon, origin_year=origin_year)
        return _blend_predictions(a, b, meta, horizon)
    raise ValueError(f"Unknown model '{model_name}'. Choose from: growth, lstm, ensemble.")


def _compute_state_holdout_metrics(history_df: pd.DataFrame, forecast_rows: list[dict], origin_year: int = 2018) -> dict | None:
    """Computes actual state-specific validation holdout metrics:
    - MAE (Mean Absolute Error)
    - WAPE (Weighted Absolute Percentage Error)
    - Directional Accuracy (% of years where the predicted trend sign matched reality)
    - Forecast Bias (Signed error: indicates over-predicting vs under-predicting)
    - Accuracy Score (100 - WAPE, clamped to 0..100)
    """
    if history_df.empty or not forecast_rows:
        return None
    hist_map = dict(zip(history_df["year"].astype(int), history_df["Total Crime"].astype(float)))
    errs, signed_errs, actuals, years, dir_matches = [], [], [], [], []
    
    prev_act = hist_map.get(origin_year, None)
    for f in forecast_rows:
        yr = int(f["year"])
        if yr in hist_map:
            act = hist_map[yr]
            pred = float(f["predicted"])
            err = abs(act - pred)
            s_err = pred - act
            errs.append(err)
            signed_errs.append(s_err)
            actuals.append(act)
            years.append(yr)
            
            if prev_act is not None:
                act_dir = np.sign(act - prev_act)
                pred_dir = np.sign(pred - prev_act)
                dir_matches.append(1 if act_dir == pred_dir else 0)
            prev_act = act

    if not errs:
        return None
    
    total_actual = sum(actuals)
    wape = (sum(errs) / total_actual * 100.0) if total_actual > 0 else 0.0
    bias_pct = (sum(signed_errs) / total_actual * 100.0) if total_actual > 0 else 0.0
    bias_cases = float(np.mean(signed_errs))
    dir_acc = (float(np.mean(dir_matches)) * 100.0) if dir_matches else 100.0
    mae = float(np.mean(errs))
    
    if abs(bias_pct) <= 1.0:
        bias_label = f"Balanced ({bias_pct:+.1f}%)"
    elif bias_pct > 1.0:
        bias_label = f"Over-predicting ({bias_pct:+.1f}%)"
    else:
        bias_label = f"Under-predicting ({bias_pct:+.1f}%)"

    return {
        "years": years,
        "sample_count": len(years),
        "mae": round(mae, 1),
        "wape_pct": round(wape, 1),
        "bias_cases": round(bias_cases, 1),
        "bias_pct": round(bias_pct, 1),
        "bias_label": bias_label,
        "dir_accuracy_pct": round(dir_acc, 1),
        "accuracy_pct": round(max(0.0, 100.0 - wape), 1),
    }


def _build_forecast_rows(preds, eff_origin: int, meta: dict, model_name: str) -> list[dict]:
    rows = []
    wape_map = {1: 0.050, 2: 0.075, 3: 0.095}
    for i, v in enumerate(preds, 1):
        val = float(v)
        h = min(i, 3)
        margin_rate = wape_map.get(h, 0.095) * (1.0 + 0.12 * (i - 1))
        margin = val * margin_rate
        rows.append({
            "year": eff_origin + i,
            "predicted": round(val, 2),
            "lower_bound": round(max(0.0, val - margin), 2),
            "upper_bound": round(val + margin, 2),
            "margin_pct": round(margin_rate * 100.0, 1),
        })
    return rows


def _compute_future_summary(history_df: pd.DataFrame, forecast_rows: list[dict], eff_origin: int) -> dict | None:
    if eff_origin <= 2018 or history_df.empty or not forecast_rows:
        return None
    hist_map = dict(zip(history_df["year"].astype(int), history_df["Total Crime"].astype(float)))
    start_val = hist_map.get(eff_origin, float(history_df["Total Crime"].iloc[-1]))
    end_row = forecast_rows[-1]
    end_val = float(end_row["predicted"])
    h = len(forecast_rows)
    cagr = (((end_val / start_val) ** (1.0 / h)) - 1.0) * 100.0 if start_val > 0 else 0.0
    total_change = ((end_val - start_val) / start_val * 100.0) if start_val > 0 else 0.0
    margin_pct = end_row.get("margin_pct", 9.5)
    return {
        "origin_year": eff_origin,
        "target_year": end_row["year"],
        "origin_crime": round(start_val, 1),
        "projected_crime": round(end_val, 1),
        "total_growth_pct": round(total_change, 1),
        "annual_growth_rate_pct": round(cagr, 1),
        "margin_of_error_pct": margin_pct,
        "confidence_range": f"{round(end_row['lower_bound']):,} to {round(end_row['upper_bound']):,}",
    }


def _backtest_for_horizon(meta, model_name, horizon):
    """Returns {mae, mape_mean, mape_median} for this model at this exact
    forecast horizon, clamped to the longest horizon we actually backtested."""
    h = min(horizon, meta["max_backtested_horizon"])
    mae = meta.get("backtest_mae_clean_states", {}).get(str(h), {}).get(model_name)
    mape = meta.get("backtest_mape_clean_states", {}).get(str(h), {}).get(model_name, {})
    return {
        "backtested_at_horizon": h,
        "mae": mae,
        "mape_mean_pct": mape.get("mean") if isinstance(mape, dict) else None,
        "mape_median_pct": mape.get("median") if isinstance(mape, dict) else None,
    }


def forecast_ml(df: pd.DataFrame, state: str, horizon: int = 3, model_name: str = "ensemble", origin_year: int = 2021) -> dict:
    model, meta = _load_forecast()
    states = meta["states"]
    if horizon < 1 or horizon > 20:
        raise ValueError("Forecast horizon must be between 1 and 20 years.")
    if model_name not in MODEL_INFO:
        raise ValueError(f"Unknown model '{model_name}'. Choose from: growth, lstm, ensemble.")

    history_full = (
        df.groupby("year", as_index=False)["Total Crime"].sum().sort_values("year")
        if state == "All"
        else df[df.state == state][["year", "Total Crime"]].sort_values("year")
    )
    max_hist_year = int(history_full.year.max())
    eff_origin = 2018 if origin_year <= 2018 else max_hist_year
    eff_horizon = min(horizon, 3) if eff_origin == 2018 else horizon

    if model_name == "growth":
        base_res = baseline_forecast(df, state, crime="Total Crime", horizon=eff_horizon, origin_year=eff_origin)
        raw_preds = [f["predicted"] for f in base_res["forecast"]]
        forecast_rows = _build_forecast_rows(raw_preds, eff_origin, meta, "growth")
        note = (
            "Validation mode (2019-2021 holdout): model was calibrated on history ≤2018 and forecast against real 2019-2021 data."
            if eff_origin == 2018 else
            "Baseline model from Prediction tab: recent 3-yr baseline + mean historical YoY growth with rolling damping."
        )
        return {
            "state": state,
            "scope_label": "All India" if state == "All" else state,
            "crime": "Total Crime",
            "model": "growth",
            "model_label": MODEL_INFO["growth"]["label"],
            "model_description": MODEL_INFO["growth"]["description"],
            "origin_year": eff_origin,
            "history": history_full.to_dict("records"),
            "forecast": forecast_rows,
            "backtest": _backtest_for_horizon(meta, "growth", eff_horizon),
            "state_holdout": _compute_state_holdout_metrics(history_full, forecast_rows, origin_year=eff_origin) if eff_origin <= 2018 else None,
            "future_summary": _compute_future_summary(history_full, forecast_rows, eff_origin),
            "data_quality_note": note,
        }

    anomaly_states = {s for s, yv in meta["anomaly_points"]}

    if state == "All":
        if model_name == "lstm":
            lstm_dict = _lstm_forecast_batch(model, meta, states, eff_horizon, origin_year=eff_origin)
            combined = sum(lstm_dict.values())
        else:  # ensemble
            lstm_dict = _lstm_forecast_batch(model, meta, states, eff_horizon, origin_year=eff_origin)
            growth_dict = _growth_forecast_batch(df, states, eff_horizon, origin_year=eff_origin)
            combined = _blend_predictions(sum(lstm_dict.values()), sum(growth_dict.values()), meta, eff_horizon)

        if eff_origin == 2018:
            flagged_note = "Validation mode (2019-2021 holdout): Models were trained on data ≤2018 and forecast against real held-out 2019-2021 observations."
        else:
            flagged_note = None
    else:
        if state not in meta["state_to_idx"]:
            raise ValueError(f"Unknown state for ML forecast: {state}")
        combined = _model_prediction(model_name, model, meta, df, state, eff_horizon, origin_year=eff_origin)
        if eff_origin == 2018:
            flagged_note = "Validation mode (2019-2021 holdout): Evaluated on real held-out observations."
        else:
            flagged_note = None

    forecast_rows = _build_forecast_rows(combined, eff_origin, meta, model_name)

    return {
        "state": state,
        "scope_label": "All India" if state == "All" else state,
        "crime": "Total Crime",
        "model": model_name,
        "model_label": MODEL_INFO[model_name]["label"],
        "model_description": MODEL_INFO[model_name]["description"],
        "origin_year": eff_origin,
        "history": history_full.to_dict("records"),
        "forecast": forecast_rows,
        "backtest": _backtest_for_horizon(meta, model_name, eff_horizon),
        "state_holdout": _compute_state_holdout_metrics(history_full, forecast_rows, origin_year=eff_origin) if eff_origin <= 2018 else None,
        "future_summary": _compute_future_summary(history_full, forecast_rows, eff_origin),
        "data_quality_note": flagged_note,
    }


def forecast_compare_ml(df: pd.DataFrame, state: str, horizon: int = 3, origin_year: int = 2021) -> dict:
    """All 3 models' forecasts + backtest error for the same state/horizon,
    for a side-by-side comparison view (Growth Baseline, LSTM, Ensemble)."""
    model, meta = _load_forecast()
    states = meta["states"]
    if horizon < 1 or horizon > 20:
        raise ValueError("Forecast horizon must be between 1 and 20 years.")

    anomaly_states = {s for s, yv in meta["anomaly_points"]}

    history = (
        df.groupby("year", as_index=False)["Total Crime"].sum().sort_values("year")
        if state == "All"
        else df[df.state == state][["year", "Total Crime"]].sort_values("year")
    )
    max_hist_year = int(history.year.max())
    eff_origin = 2018 if origin_year <= 2018 else max_hist_year
    eff_horizon = min(horizon, 3) if eff_origin == 2018 else horizon

    if state == "All":
        lstm_dict = _lstm_forecast_batch(model, meta, states, eff_horizon, origin_year=eff_origin)
        growth_dict = _growth_forecast_batch(df, states, eff_horizon, origin_year=eff_origin)
        lstm_comb = sum(lstm_dict.values())
        growth_comb = sum(growth_dict.values())
        ensemble_comb = _blend_predictions(lstm_comb, growth_comb, meta, eff_horizon)

        if eff_origin == 2018:
            flagged_note = "Validation mode (2019-2021 holdout): Models were trained on data ≤2018 and forecast against real held-out 2019-2021 observations."
        else:
            flagged_note = None
        scope_label = "All India"
    else:
        if state not in meta["state_to_idx"]:
            raise ValueError(f"Unknown state for ML forecast: {state}")
        lstm_comb = _lstm_forecast(model, meta, state, eff_horizon, origin_year=eff_origin)
        growth_comb = _growth_forecast(df, state, eff_horizon, origin_year=eff_origin)
        ensemble_comb = _blend_predictions(lstm_comb, growth_comb, meta, eff_horizon)

        if eff_origin == 2018:
            flagged_note = "Validation mode (2019-2021 holdout): Evaluated on real held-out observations."
        else:
            flagged_note = None
        scope_label = state

    model_preds = {
        "growth": growth_comb,
        "lstm": lstm_comb,
        "ensemble": ensemble_comb,
    }

    results = {}
    for model_name in ("growth", "lstm", "ensemble"):
        preds = model_preds[model_name]
        forecast_rows = _build_forecast_rows(preds, eff_origin, meta, model_name)
        results[model_name] = {
            "label": MODEL_INFO[model_name]["label"],
            "description": MODEL_INFO[model_name]["description"],
            "forecast": forecast_rows,
            "backtest": _backtest_for_horizon(meta, model_name, eff_horizon),
            "state_holdout": _compute_state_holdout_metrics(history, forecast_rows, origin_year=eff_origin) if eff_origin <= 2018 else None,
            "future_summary": _compute_future_summary(history, forecast_rows, eff_origin),
        }

    future_consensus = None
    if eff_origin > 2018:
        end_values = [m["forecast"][-1]["predicted"] for m in results.values()]
        mean_val = float(np.mean(end_values))
        future_consensus = {
            "mean_projected": round(mean_val, 1),
            "min_projected": round(min(end_values), 1),
            "max_projected": round(max(end_values), 1),
        }

    return {
        "state": state,
        "scope_label": scope_label,
        "crime": "Total Crime",
        "origin_year": eff_origin,
        "history": history.to_dict("records"),
        "data_quality_note": flagged_note,
        "models": results,
        "future_consensus": future_consensus,
    }
