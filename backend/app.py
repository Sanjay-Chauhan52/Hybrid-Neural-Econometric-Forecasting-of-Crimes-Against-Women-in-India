from pathlib import Path
import json
import math
import numpy as np
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory, make_response
from urllib.request import Request, urlopen
from analytics import (
    clean_dataframe,
    CRIME_COLUMNS,
    growth_table,
    scoped_growth_table,
    cri_for_year,
    temporal_summary,
    correlation,
    forecast,
)

try:
    from analytics_ml import cri_for_year_ml, forecast_ml, forecast_compare_ml, models_available
    ML_IMPORT_ERROR = None
except Exception as e:  # TensorFlow missing, or models not trained/placed yet
    ML_IMPORT_ERROR = str(e)

    def models_available():
        return {"cri_ensemble": False, "forecast_ensemble": False}

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "dataset"
DATA_DIR.mkdir(exist_ok=True)
FRONT = ROOT / "frontend"
app = Flask(__name__, static_folder=str(FRONT), static_url_path="")
STATE = {"df": None, "source": "No dataset loaded"}

GEO_SOURCES = [
    # 2019 state/UT boundaries align with the dataset's 36-jurisdiction structure
    # (including J&K/Ladakh and separate DNH/DD records used in this project).
    "https://raw.githubusercontent.com/india-in-data/india-states-2019/master/india_states.geojson",
    "https://cdn.jsdelivr.net/gh/AbhinavSwami28/india-official-geojson@main/india-states-simplified.geojson",
    "https://raw.githubusercontent.com/AbhinavSwami28/india-official-geojson/main/india-states-simplified.geojson",
    "https://webgis1.nic.in/nicstreet/rest/services/terraintile/MapServer/0/query?where=1%3D1&outFields=*&outSR=4326&f=geojson",
]
GEO_CACHE = None


def safe_json(value):
    if isinstance(value, dict):
        return {str(k): safe_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [safe_json(v) for v in value]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        value = float(value)
    if isinstance(value, float):
        return None if not math.isfinite(value) else value
    if pd.isna(value) if not isinstance(value, (str, bytes, bool)) else False:
        return None
    return value


def respond(**payload):
    return jsonify(safe_json(payload))


def make_demo():
    states = ["Andhra Pradesh", "Assam", "Bihar", "Gujarat", "Karnataka", "Kerala", "Madhya Pradesh", "Maharashtra", "Tamil Nadu", "Uttar Pradesh", "West Bengal"]
    rows = []
    for si, st in enumerate(states):
        base = 500 + si * 130
        for year in range(2001, 2022):
            t = year - 2001
            trend = 1 + 0.035 * t + (0.012 * si)
            rows.append({
                "state": st,
                "year": year,
                "Rape": round(base * .18 * trend),
                "K&A": round(base * .22 * trend),
                "DD": round(base * .08 * trend),
                "AoW": round(base * .20 * trend),
                "AoM": round(base * .06 * trend),
                "DV": round(base * .21 * trend),
                "WT": round(base * .05 * trend),
            })
    return clean_dataframe(pd.DataFrame(rows))


def load_default():
    candidates = list(DATA_DIR.glob("*.csv")) + list(DATA_DIR.glob("*.xlsx"))
    if not candidates:
        STATE["df"] = make_demo()
        STATE["source"] = "Synthetic demo data — upload the real dataset"
        return
    p = candidates[0]
    raw = pd.read_csv(p) if p.suffix.lower() == ".csv" else pd.read_excel(p)
    STATE["df"] = clean_dataframe(raw)
    STATE["source"] = p.name


load_default()


def df_or_error():
    if STATE["df"] is None:
        raise ValueError("No dataset loaded. Upload a CSV/XLSX file or place it in dataset/.")
    return STATE["df"]


@app.get("/")
def index():
    return send_from_directory(FRONT, "index.html")


@app.get("/favicon.ico")
def favicon():
    return make_response(("", 204))


@app.get("/<path:path>")
def serve_frontend(path):
    return send_from_directory(FRONT, path)


@app.post("/api/upload")
def upload():
    f = request.files.get("file")
    if not f:
        return respond(error="No file supplied"), 400
    try:
        raw = pd.read_csv(f) if f.filename.lower().endswith(".csv") else pd.read_excel(f)
        STATE["df"] = clean_dataframe(raw)
        STATE["source"] = f.filename
        years = sorted(STATE["df"].year.unique().tolist())
        return respond(ok=True, rows=len(STATE["df"]), source=f.filename, years=years, states=int(STATE["df"].state.nunique()))
    except Exception as e:
        return respond(error=str(e)), 400


@app.get("/api/meta")
def meta():
    try:
        df = df_or_error()
        return respond(
            source=STATE["source"],
            rows=len(df),
            states=sorted(df.state.unique().tolist()),
            years=sorted(df.year.unique().tolist()),
            crimes=CRIME_COLUMNS,
        )
    except Exception as e:
        return respond(error=str(e)), 400


@app.get("/api/overview")
def api_overview():
    from analytics import overview
    try:
        df = df_or_error()
        y = int(request.args.get("year", df.year.max()))
        crime = request.args.get("crime", "Total Crime")
        return respond(**overview(df, y, crime))
    except Exception as e:
        return respond(error=str(e)), 400


@app.get("/api/geojson")
def api_geojson():
    global GEO_CACHE
    try:
        if GEO_CACHE is not None:
            response = make_response(GEO_CACHE)
            response.headers["Content-Type"] = "application/geo+json; charset=utf-8"
            response.headers["Cache-Control"] = "public, max-age=86400"
            return response

        last_error = None
        for url in GEO_SOURCES:
            try:
                req = Request(url, headers={"User-Agent": "IndiaCrimeAnalytics/1.0"})
                with urlopen(req, timeout=12) as r:
                    payload = r.read()
                text = payload.decode("utf-8")
                obj = json.loads(text)
                if obj.get("type") == "FeatureCollection" and obj.get("features"):
                    GEO_CACHE = text
                    response = make_response(text)
                    response.headers["Content-Type"] = "application/geo+json; charset=utf-8"
                    response.headers["Cache-Control"] = "public, max-age=86400"
                    return response
            except Exception as e:
                last_error = str(e)

        raise RuntimeError(f"India boundary data could not be loaded by the server. {last_error or ''}")
    except Exception as e:
        return respond(error=str(e)), 503


@app.get("/api/map")
def api_map():
    try:
        df = df_or_error()
        y = int(request.args.get("year", df.year.max()))
        metric = request.args.get("metric", "cri")
        crime = request.args.get("crime", "Total Crime")
        if metric not in {"cri", "crime", "growth"}:
            raise ValueError("Unknown map metric. Use cri, crime, or growth.")
        if crime != "Total Crime" and crime not in CRIME_COLUMNS:
            raise ValueError(f"Unknown crime category: {crime}")

        # Build the year slice from one authoritative source first. This avoids
        # duplicate `Total Crime` columns when crime == `Total Crime`.
        base = df[df["year"] == y][["state", "Total Crime", *CRIME_COLUMNS]].copy()
        if base.empty:
            raise ValueError(f"No records found for year {y}.")

        base["Crime Value"] = base[crime] if crime in base.columns else base["Total Crime"]

        # Growth is category-specific. A first observation or a zero denominator
        # is undefined and remains null instead of becoming NaN/Infinity in JSON.
        g = growth_table(df, crime)
        g = g[g.year == y][["state", "Growth Rate %"]].copy()

        c = cri_for_year(df, y)
        c = c[["state", "CRI", "Risk"]].copy()

        x = (
            base[["state", "Total Crime", "Crime Value"]]
            .merge(g, on="state", how="left")
            .merge(c, on="state", how="left")
        )
        return respond(
            year=y,
            metric=metric,
            crime=crime,
            data=x[["state", "Total Crime", "Crime Value", "Growth Rate %", "CRI", "Risk"]].to_dict("records"),
        )
    except Exception as e:
        return respond(error=str(e)), 400


@app.get("/api/temporal")
def api_temporal():
    try:
        df = df_or_error()
        state = request.args.get("state", "All")
        crime = request.args.get("crime", "Total Crime")
        min_year = int(df.year.min())
        max_year = int(df.year.max())
        start_year = int(request.args.get("start_year", min_year))
        end_year = int(request.args.get("end_year", max_year))
        if start_year > end_year:
            raise ValueError("Start year cannot be greater than end year.")
        start_year = max(min_year, start_year)
        end_year = min(max_year, end_year)
        if start_year > end_year:
            raise ValueError("Selected temporal range is outside the available dataset years.")

        # Calculate growth over the full chronological series first so the first
        # displayed year can retain a valid comparison to the immediately
        # preceding year. Only the requested display range is returned.
        x = temporal_summary(df, state, crime)
        gg = scoped_growth_table(df, crime, state)
        out = x.merge(gg[["year", "Growth Rate %"]], on="year", how="left")
        out = out[(out.year >= start_year) & (out.year <= end_year)].copy()
        return respond(data=out.to_dict("records"), crime=crime, state=state, start_year=start_year, end_year=end_year)
    except Exception as e:
        return respond(error=str(e)), 400


@app.get("/api/spatiotemporal")
def api_spatiotemporal():
    try:
        df = df_or_error()
        crime  = request.args.get("crime", "Total Crime")
        mode   = request.args.get("mode", "normalized")   # volume|normalized|growth|rank
        sort   = request.args.get("sort", "severity")     # severity|alphabetical|growth|peak_year

        x = df.pivot_table(index="state", columns="year", values=crime,
                           aggfunc="sum", dropna=False).reset_index()
        value_cols = sorted([c for c in x.columns if c != "state"])
        years      = [int(c) for c in value_cols]
        raw        = x[value_cols].values.astype(float)
        n_states, n_years = raw.shape

        # ── missing data metadata ─────────────────────────────────────────────
        missing_points, missing_states_set = [], set()
        for i, sn in enumerate(x.state):
            for j, yr in enumerate(years):
                if np.isnan(raw[i, j]):
                    missing_points.append({"state": sn, "year": yr})
                    missing_states_set.add(sn)
        missing_count = len(missing_points)

        # ── build display matrix per mode ────────────────────────────────────
        disp = np.full_like(raw, np.nan)

        if mode == "volume":
            disp = raw.copy()
            scale_title = f"{crime} (cases)"

        elif mode == "normalized":
            for i in range(n_states):
                row = raw[i]
                valid = row[~np.isnan(row)]
                if len(valid) > 1:
                    lo, hi = valid.min(), valid.max()
                    span = hi - lo if hi > lo else 1.0
                    disp[i] = np.where(np.isnan(row), np.nan, (row - lo) / span * 100)
                elif len(valid) == 1:
                    disp[i] = np.where(np.isnan(row), np.nan, 50.0)
            scale_title = "Relative Intensity (0-100%)"

        elif mode == "growth":
            for i in range(n_states):
                for j in range(1, n_years):
                    prev, curr = raw[i, j - 1], raw[i, j]
                    if not np.isnan(prev) and not np.isnan(curr) and prev > 0:
                        disp[i, j] = (curr - prev) / prev * 100
            scale_title = "YoY Growth (%)"

        elif mode == "rank":
            for j in range(n_years):
                col = raw[:, j]
                valid_mask = ~np.isnan(col)
                n_valid = valid_mask.sum()
                if n_valid > 1:
                    ranks = np.argsort(np.argsort(col[valid_mask]))  # 0-based rank
                    pct = ranks / (n_valid - 1) * 100
                    disp[valid_mask, j] = pct
            scale_title = "Within-year Crime Rank (0=lowest, 100=highest)"

        else:
            disp = raw.copy()
            scale_title = f"{crime} (cases)"

        # ── per-state summary stats ───────────────────────────────────────────
        row_totals = np.nansum(raw, axis=1)
        row_means  = np.nanmean(raw, axis=1)

        peak_years, overall_growth = [], []
        for i in range(n_states):
            row = raw[i]
            valid_idx = np.where(~np.isnan(row))[0]
            peak_years.append(years[valid_idx[np.argmax(row[valid_idx])]] if len(valid_idx) else None)
            valid_vals = row[~np.isnan(row)]
            if len(valid_vals) >= 2 and valid_vals[0] > 0:
                n = len(valid_vals)
                overall_growth.append(round(float(((valid_vals[-1] / valid_vals[0]) ** (1 / (n - 1)) - 1) * 100), 2))
            else:
                overall_growth.append(None)

        # ── sort order ────────────────────────────────────────────────────────
        state_names = x.state.tolist()
        order = list(range(n_states))
        if sort == "alphabetical":
            order = sorted(order, key=lambda i: state_names[i])
        elif sort == "severity":
            order = sorted(order, key=lambda i: -row_totals[i])
        elif sort == "growth":
            order = sorted(order, key=lambda i: -(overall_growth[i] or -9999))
        elif sort == "peak_year":
            order = sorted(order, key=lambda i: (peak_years[i] or 0), reverse=True)

        def _clean(row):
            return [None if np.isnan(v) else round(float(v), 2) for v in row]

        return respond(
            years=years,
            states=[state_names[i] for i in order],
            matrix=[_clean(disp[i]) for i in order],
            raw_matrix=[_clean(raw[i]) for i in order],
            crime=crime,
            mode=mode,
            sort=sort,
            scale_title=scale_title,
            missing_count=missing_count,
            missing_points=missing_points,
            missing_summary=sorted(missing_states_set),
            peak_years=[peak_years[i] for i in order],
            overall_growth_pct=[overall_growth[i] for i in order],
            total_volume=[round(float(row_totals[i]), 0) for i in order],
            mean_volume=[round(float(row_means[i]), 0) for i in order],
            national_totals=[round(float(np.nansum(raw[:, j])), 0) for j in range(n_years)],
            peak_national_year=years[int(np.argmax([np.nansum(raw[:, j]) for j in range(n_years)]))] if n_years else None,
            avg_cagr_pct=round(float(np.mean([g for g in overall_growth if g is not None])), 2) if any(g is not None for g in overall_growth) else None,
        )
    except Exception as e:
        return respond(error=str(e)), 400


@app.get("/api/cri")
def api_cri():
    try:
        df = df_or_error()
        y = int(request.args.get("year", df.year.max()))
        c = cri_for_year(df, y)
        return respond(year=y, data=c[["state", "CRI", "Risk", "Total Crime"]].to_dict("records"))
    except Exception as e:
        return respond(error=str(e)), 400


@app.get("/api/correlation")
def api_corr():
    try:
        return respond(data=correlation(df_or_error()))
    except Exception as e:
        return respond(error=str(e)), 400


@app.get("/api/forecast-options")
def api_forecast_options():
    try:
        df = df_or_error()
        state = request.args.get("state", "All")
        if state == "All":
            years = sorted(df.year.unique().astype(int).tolist())
        else:
            years = sorted(df.loc[df.state == state, "year"].unique().astype(int).tolist())
        if len(years) < 3:
            raise ValueError("At least 3 observations are required for forecasting this scope.")
        forecast_years = years[2:]
        return respond(state=state, years=forecast_years, min_year=min(forecast_years), max_year=max(years), observed_years=years)
    except Exception as e:
        return respond(error=str(e)), 400


@app.get("/api/forecast")
def api_forecast():
    try:
        df = df_or_error()
        state = request.args.get("state", "All")
        crime = request.args.get("crime", "Total Crime")
        horizon = int(request.args.get("horizon", 3))
        origin_year_arg = request.args.get("origin_year")
        origin_year = int(origin_year_arg) if origin_year_arg else int(df.year.max())
        return respond(**forecast(df, state, crime=crime, horizon=horizon, origin_year=origin_year))
    except Exception as e:
        return respond(error=str(e)), 400


@app.get("/api/categories")
def api_categories():
    try:
        df = df_or_error()
        y = int(request.args.get("year", df.year.max()))
        state = request.args.get("state", "All")
        if state == "All":
            row = df[df.year == y][CRIME_COLUMNS].sum()
        else:
            x = df[(df.year == y) & (df.state == state)]
            if x.empty:
                raise ValueError("No data for selected state/year.")
            row = x[CRIME_COLUMNS].sum()
        return respond(year=y, state=state, categories={c: float(row[c]) for c in CRIME_COLUMNS})
    except Exception as e:
        return respond(error=str(e)), 400


@app.get("/api/state")
def api_state():
    try:
        df = df_or_error()
        state = request.args.get("state")
        y = int(request.args.get("year", df.year.max()))
        crime = request.args.get("crime", "Total Crime")
        if crime != "Total Crime" and crime not in CRIME_COLUMNS:
            return respond(error=f"Unknown crime category: {crime}"), 400

        # Handle the combined map geometry used for Dadra and Nagar Haveli + Daman and Diu.
        combined = state in {
            "Dadra and Nagar Haveli and Daman and Diu",
            "Dadra and Nagar Haveli & Daman and Diu",
        }
        if combined:
            x = df[df.state.isin(["Dadra and Nagar Haveli", "Daman and Diu"])].copy()
            if x.empty:
                return respond(error="State not found"), 404
            series = x.groupby("year", as_index=False)[crime].sum().sort_values("year")
            series["Growth Rate %"] = series[crime].pct_change(fill_method=None).replace([np.inf, -np.inf], np.nan) * 100
            series["Recent Average"] = series[crime].rolling(3, min_periods=1).mean()
            row_val = float(series.loc[series.year == y, crime].iloc[0]) if (series.year == y).any() else 0.0
            c_parts = cri_for_year(df, y)
            parts = c_parts[c_parts.state.isin(["Dadra and Nagar Haveli", "Daman and Diu"])].copy()
            cri_val = float(parts.CRI.mean()) if not parts.empty else None
            risk = "High" if "High" in parts.Risk.tolist() else ("Medium" if "Medium" in parts.Risk.tolist() else "Low")
            cat = {c: float(x[x.year == y][c].sum()) if not x[x.year == y].empty else 0.0 for c in CRIME_COLUMNS}
            cri_history = []
            for yr in sorted(series.year.astype(int).tolist()):
                cy = cri_for_year(df, yr)
                pr = cy[cy.state.isin(["Dadra and Nagar Haveli", "Daman and Diu"])]
                if not pr.empty:
                    cri_history.append({"year": yr, "CRI": float(pr.CRI.mean())})
            summary = {"CRI": cri_val, "Risk": risk, "Total Crime": row_val}
            return respond(
                state=state, year=y, crime=crime, summary=summary,
                selected_value=row_val, history=series.to_dict("records"),
                cri_history=cri_history, categories=cat,
            )

        c = cri_for_year(df, y)
        row = c[c.state == state]
        if row.empty:
            return respond(error="State not found for selected year"), 404
        r = growth_table(df, crime)
        rr = r[r.state == state].sort_values("year")
        cri_history = []
        for yr in sorted(rr.year.astype(int).tolist()):
            cy = cri_for_year(df, yr)
            cr = cy[cy.state == state]
            if not cr.empty:
                cri_history.append({"year": yr, "CRI": float(cr.CRI.iloc[0])})
        return respond(
            state=state,
            year=y,
            crime=crime,
            summary=row[["CRI", "Risk", "Total Crime"]].iloc[0].to_dict(),
            selected_value=float(row[crime].iloc[0]) if crime in row.columns else float(rr.loc[rr.year == y, crime].iloc[0]),
            history=rr[["year", crime, "Growth Rate %", "Recent Average"]].to_dict("records"),
            cri_history=cri_history,
            categories=row[CRIME_COLUMNS].iloc[0].to_dict(),
        )
    except Exception as e:
        return respond(error=str(e)), 400


@app.get("/api/ml-status")
def api_ml_status():
    return respond(available=models_available(), import_error=ML_IMPORT_ERROR)


@app.get("/api/cri-ml")
def api_cri_ml():
    try:
        if not models_available()["cri_ensemble"]:
            raise ValueError("CRI ensemble model not found. Run train_cri_ensemble.py and place its output in models/.")
        df = df_or_error()
        y = int(request.args.get("year", df.year.max()))
        c = cri_for_year_ml(df, y)
        return respond(
            year=y,
            data=c[["state", "CRI", "Risk", "Total Crime", "CRI_heuristic",
                     "CRI_anomaly_signal", "CRI_statistical_signal"]].to_dict("records"),
        )
    except Exception as e:
        return respond(error=str(e)), 400


@app.get("/api/forecast-ml")
def api_forecast_ml():
    try:
        if not models_available()["forecast_ensemble"]:
            raise ValueError("Forecast ensemble model not found. Run train_forecast_ensemble.py and place its output in models/.")
        df = df_or_error()
        state = request.args.get("state", "All")
        horizon = int(request.args.get("horizon", 3))
        crime = request.args.get("crime", "Total Crime")
        model = request.args.get("model", "ensemble")
        origin_year = int(request.args.get("origin_year", 2021))
        if crime != "Total Crime":
            raise ValueError("The ML forecast models currently only support 'Total Crime'.")
        return respond(**forecast_ml(df, state, horizon=horizon, model_name=model, origin_year=origin_year))
    except Exception as e:
        return respond(error=str(e)), 400


@app.get("/api/forecast-compare-ml")
def api_forecast_compare_ml():
    try:
        if not models_available()["forecast_ensemble"]:
            raise ValueError("Forecast ensemble model not found. Run train_forecast_ensemble.py and place its output in models/.")
        df = df_or_error()
        state = request.args.get("state", "All")
        horizon = int(request.args.get("horizon", 3))
        origin_year = int(request.args.get("origin_year", 2021))
        return respond(**forecast_compare_ml(df, state, horizon=horizon, origin_year=origin_year))
    except Exception as e:
        return respond(error=str(e)), 400


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=True)
