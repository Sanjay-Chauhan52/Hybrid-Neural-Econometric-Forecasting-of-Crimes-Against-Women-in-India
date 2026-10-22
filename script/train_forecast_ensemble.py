import sys
from pathlib import Path

# Ensure root and backend in sys.path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "backend"))
sys.path.insert(0, str(ROOT))

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from ml_data_utils import flag_anomalies, repair_series_for_input
from analytics import clean_dataframe, forecast as baseline_forecast

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

WINDOW = 3
TARGET = "Total Crime"
HOLDOUT_YEARS = [2019, 2020, 2021]

MODELS_DIR = ROOT / "models"
DATASET_PATH = ROOT / "dataset" / "cleaned_crime_data.csv"


def load_state_series(df):
    series = {}
    for state, g in df.groupby("state"):
        g = g.sort_values("year")
        series[state] = dict(zip(g.year, g[TARGET]))
    return series


def make_windows(series, states, state_to_idx, anomaly_set, years_allowed):
    X_seq, X_state, y, w = [], [], [], []
    norm_stats = {}
    for state in states:
        raw = series[state]
        repaired = repair_series_for_input(raw, state, anomaly_set)
        yrs = sorted(raw.keys())
        log_vals = {yr: np.log1p(repaired[yr]) for yr in yrs}
        lo, hi = min(log_vals.values()), max(log_vals.values())
        span = hi - lo if hi > lo else 1.0
        norm_stats[state] = {"log_min": float(lo), "log_span": float(span)}

        norm = {yr: (log_vals[yr] - lo) / span for yr in yrs}
        for i in range(WINDOW, len(yrs)):
            target_year = yrs[i]
            if target_year not in years_allowed:
                continue
            window_years = yrs[i - WINDOW : i]
            X_seq.append([norm[wy] for wy in window_years])
            X_state.append(state_to_idx[state])
            y.append(norm[target_year])
            w.append(0.0 if (state, target_year) in anomaly_set else 1.0)

    X_seq = np.asarray(X_seq, dtype="float32").reshape(-1, WINDOW, 1)
    X_state = np.asarray(X_state, dtype="int32")
    y = np.asarray(y, dtype="float32").reshape(-1, 1)
    w = np.asarray(w, dtype="float32")
    return X_seq, X_state, y, w, norm_stats


def build_lstm(n_states):
    seq_in = keras.Input(shape=(WINDOW, 1), name="window")
    state_in = keras.Input(shape=(1,), name="state_id")

    x = keras.layers.LSTM(16, kernel_regularizer=keras.regularizers.l2(1e-4))(seq_in)
    x = keras.layers.Dropout(0.2)(x)

    emb = keras.layers.Embedding(n_states, 4)(state_in)
    emb = keras.layers.Flatten()(emb)

    h = keras.layers.Concatenate()([x, emb])
    h = keras.layers.Dense(8, activation="relu", kernel_regularizer=keras.regularizers.l2(1e-4))(h)
    out = keras.layers.Dense(1, activation="linear")(h)

    model = keras.Model([seq_in, state_in], out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss=keras.losses.Huber(delta=0.3))
    return model


def lstm_forecast_state(model, norm_stats, series, state, state_idx, horizon, last_year, anomaly_set):
    stats = norm_stats[state]
    lo, span = stats["log_min"], stats["log_span"]
    raw = {y: v for y, v in series[state].items() if y <= last_year}
    repaired = repair_series_for_input(raw, state, anomaly_set)
    yrs = sorted(raw.keys())
    log_vals = [np.log1p(repaired[y]) for y in yrs]
    norm_vals = [(v - lo) / span for v in log_vals]
    buf = norm_vals[-WINDOW:]

    outputs = []
    for _ in range(horizon):
        seq = np.array(buf[-WINDOW:], dtype="float32").reshape(1, WINDOW, 1)
        sid = np.array([[state_idx]], dtype="int32")
        nxt = float(model([seq, sid], training=False).numpy().flatten()[0])
        buf.append(nxt)
        outputs.append(max(0.0, np.expm1(nxt * span + lo)))
    return np.array(outputs)


def mape(errs, actuals):
    pairs = [(e, a) for e, a in zip(errs, actuals) if a > 0]
    if not pairs:
        return {"mean": None, "median": None}
    ape = [e / a * 100 for e, a in pairs]
    return {"mean": float(np.mean(ape)), "median": float(np.median(ape))}


def main():
    df = clean_dataframe(pd.read_csv(DATASET_PATH))
    series = load_state_series(df)
    states = sorted(series.keys())
    state_to_idx = {s: i for i, s in enumerate(states)}
    anomaly_set = flag_anomalies(df, TARGET)
    print(f"Flagged {len(anomaly_set)} anomalous (state, year) points.")

    repaired_series = {state: repair_series_for_input(series[state], state, anomaly_set) for state in states}
    all_years = sorted(df.year.unique().tolist())
    train_years = set(y for y in all_years if y not in HOLDOUT_YEARS)

    # 1. Train LSTM on historical years <= 2018
    X_seq, X_state, y, w, norm_stats = make_windows(series, states, state_to_idx, anomaly_set, train_years)
    n = len(y)
    idx = np.random.permutation(n)
    split = int(n * 0.85)
    tr_idx, va_idx = idx[:split], idx[split:]

    lstm = build_lstm(len(states))
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=25, restore_best_weights=True)
    lstm.fit(
        [X_seq[tr_idx], X_state[tr_idx]], y[tr_idx], sample_weight=w[tr_idx],
        validation_data=([X_seq[va_idx], X_state[va_idx]], y[va_idx], w[va_idx]),
        epochs=500, batch_size=32, verbose=0, callbacks=[early_stop],
    )
    print("LSTM trained on <=2018 history.")

    # 2. Backtest LSTM and Damped Growth on 2019-2021 holdout
    per_horizon_errs = {1: {"A": [], "B": []}, 2: {"A": [], "B": []}, 3: {"A": [], "B": []}}
    per_horizon_errs_clean = {1: {"A": [], "B": []}, 2: {"A": [], "B": []}, 3: {"A": [], "B": []}}
    per_horizon_actuals_clean = {1: [], 2: [], 3: []}

    growth_preds_by_state = {}
    for state in states:
        g_res = baseline_forecast(df, state, crime=TARGET, horizon=3, origin_year=2018)
        growth_preds_by_state[state] = [f["predicted"] for f in g_res["forecast"]]

        sidx = state_to_idx[state]
        a_preds = lstm_forecast_state(lstm, norm_stats, series, state, sidx, horizon=3, last_year=2018, anomaly_set=anomaly_set)
        b_preds = growth_preds_by_state[state]

        for h, yr in enumerate(HOLDOUT_YEARS, start=1):
            if yr not in series[state]:
                continue
            actual = series[state][yr]
            err_a = abs(a_preds[h - 1] - actual)
            err_b = abs(b_preds[h - 1] - actual)
            per_horizon_errs[h]["A"].append(err_a)
            per_horizon_errs[h]["B"].append(err_b)
            if (state, yr) not in anomaly_set:
                per_horizon_errs_clean[h]["A"].append(err_a)
                per_horizon_errs_clean[h]["B"].append(err_b)
                per_horizon_actuals_clean[h].append(actual)

    weights_per_horizon = {}
    print("\nBacktest MAE/MAPE by horizon (CLEAN states, 29/36):")
    for h in (1, 2, 3):
        mae_a = float(np.mean(per_horizon_errs_clean[h]["A"]))
        mae_b = float(np.mean(per_horizon_errs_clean[h]["B"]))
        inv_a, inv_b = 1 / max(mae_a, 1e-6), 1 / max(mae_b, 1e-6)
        w_a, w_b = inv_a / (inv_a + inv_b), inv_b / (inv_a + inv_b)
        weights_per_horizon[h] = {"lstm": w_a, "growth": w_b}
        mape_a = mape(per_horizon_errs_clean[h]["A"], per_horizon_actuals_clean[h])
        mape_b = mape(per_horizon_err_clean := per_horizon_errs_clean[h]["B"], per_horizon_actuals_clean[h])
        print(f"  h={h}: LSTM MAE={mae_a:.1f} MAPE(mean/median)={mape_a['mean']:.1f}%/{mape_a['median']:.1f}% (w={w_a:.2f})  "
              f"|  Growth MAE={mae_b:.1f} MAPE(mean/median)={mape_b['mean']:.1f}%/{mape_b['median']:.1f}% (w={w_b:.2f})")

    # 3. Ensemble Backtest
    ens_errs_by_h = {1: [], 2: [], 3: []}
    for state in states:
        if any((state, yr) in anomaly_set for yr in HOLDOUT_YEARS):
            continue
        sidx = state_to_idx[state]
        a_preds = lstm_forecast_state(lstm, norm_stats, series, state, sidx, horizon=3, last_year=2018, anomaly_set=anomaly_set)
        b_preds = growth_preds_by_state[state]
        for h, yr in enumerate(HOLDOUT_YEARS, start=1):
            if yr not in series[state]:
                continue
            actual = series[state][yr]
            w_a, w_b = weights_per_horizon[h]["lstm"], weights_per_horizon[h]["growth"]
            ens_pred = w_a * a_preds[h - 1] + w_b * b_preds[h - 1]
            ens_errs_by_h[h].append(abs(ens_pred - actual))

    mape_ens_by_h = {h: mape(ens_errs_by_h[h], per_horizon_actuals_clean[h]) for h in (1, 2, 3)}
    print("\nEnsemble (LSTM + Damped Growth) Backtest:")
    for h in (1, 2, 3):
        m = mape_ens_by_h[h]
        print(f"  h={h}: Ensemble MAE={np.mean(ens_errs_by_h[h]):.1f} MAPE(mean/median)={m['mean']:.1f}%/{m['median']:.1f}%")

    # 4. Retrain final LSTM on full history <= 2021
    print("\nRetraining final LSTM on full 2001-2021 history...")
    X_seq_f, X_state_f, y_f, w_f, norm_stats_f = make_windows(
        series, states, state_to_idx, anomaly_set, set(all_years)
    )
    nf = len(y_f)
    idxf = np.random.permutation(nf)
    splitf = int(nf * 0.85)
    trf, vaf = idxf[:splitf], idxf[splitf:]

    lstm_final = build_lstm(len(states))
    lstm_final.fit(
        [X_seq_f[trf], X_state_f[trf]], y_f[trf], sample_weight=w_f[trf],
        validation_data=([X_seq_f[vaf], X_state_f[vaf]], y_f[vaf], w_f[vaf]),
        epochs=500, batch_size=32, verbose=0, callbacks=[early_stop],
    )
    MODELS_DIR.mkdir(exist_ok=True)
    lstm_final.save(MODELS_DIR / "forecast_lstm.keras")

    full_log_series = {
        state: [np.log1p(repaired_series[state][yr]) for yr in sorted(repaired_series[state].keys())]
        for state in states
    }

    meta = {
        "window": WINDOW,
        "target": TARGET,
        "states": states,
        "state_to_idx": state_to_idx,
        "norm_stats": norm_stats_f,
        "trend_log_series_full": full_log_series,
        "weights_per_horizon": weights_per_horizon,
        "max_backtested_horizon": 3,
        "anomaly_points": [[s, int(yv)] for s, yv in anomaly_set],
        "backtest_mae_clean_states": {
            str(h): {
                "lstm": float(np.mean(per_horizon_errs_clean[h]["A"])),
                "growth": float(np.mean(per_horizon_errs_clean[h]["B"])),
                "ensemble": float(np.mean(ens_errs_by_h[h])),
            }
            for h in (1, 2, 3)
        },
        "backtest_mape_clean_states": {
            str(h): {
                "lstm": mape(per_horizon_errs_clean[h]["A"], per_horizon_actuals_clean[h]),
                "growth": mape(per_horizon_errs_clean[h]["B"], per_horizon_actuals_clean[h]),
                "ensemble": mape_ens_by_h[h],
            }
            for h in (1, 2, 3)
        },
    }
    with open(MODELS_DIR / "forecast_ensemble_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("Saved models/forecast_lstm.keras + models/forecast_ensemble_meta.json successfully.")


if __name__ == "__main__":
    main()
