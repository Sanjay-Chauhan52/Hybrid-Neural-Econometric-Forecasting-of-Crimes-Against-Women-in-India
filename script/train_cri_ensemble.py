"""
Ensemble CRI model -- combines THREE independently-computed risk signals
instead of one formula, and fuses them with PCA:

  S1 - Heuristic:    your original equal-weight percentile sum (analytics.cri_for_year).
                      Transparent, simple, the baseline everything is compared to.

  S2 - Anomaly:      an autoencoder trained ONLY on reconstruction loss (pure
                      unsupervised -- no target, so it can't just learn to
                      copy S1). Its reconstruction error says how atypical a
                      state-year's crime *profile* (the relative mix of the
                      7 categories) is, independent of overall scale. Percentile-
                      ranked within each year -> S2.

  S3 - Statistical:  Mahalanobis distance of each state-year's category
                      vector from that year's mean, using the full
                      covariance matrix -- a classical multivariate-statistics
                      severity measure. Unlike S1 it accounts for
                      correlation between categories (e.g. Rape and DV tend
                      to move together, so being high on both isn't "twice as
                      informative" as the independent-weights sum assumes).
                      Percentile-ranked within each year -> S3.

Fusion: standardize S1/S2/S3 (zero mean, unit variance) each year, then take
the first principal component across all state-year rows. PCA finds the
single direction of maximum shared variance across the three signals --
i.e. the "consensus" score that all three broadly agree is high/low -- which
is a principled way to combine heterogeneous scores without hand-picking
weights. The component is sign-corrected (so higher = higher risk, matching
S1) and rescaled to 0-100 per year, same tertile Risk banding as before.

This also reports how well the three signals agree (eigenvalue share of
PC1, and each signal's correlation with the final ensemble) as a measure of
how much consensus -- not "accuracy" in the labeled sense, since there's no
ground-truth CRI, but a real measure of whether the three independent
methods are pointing the same direction or disagreeing.
"""
import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from analytics import CRIME_COLUMNS, percentile_score

SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)


def build_features(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for year, g in df.groupby("year"):
        g = g.copy()
        for c in CRIME_COLUMNS:
            g[c + "_norm"] = percentile_score(g[c])
        feats = g[[c + "_norm" for c in CRIME_COLUMNS]].to_numpy(dtype="float64")

        # S1: heuristic (equal-weight average of category percentiles == analytics.cri_for_year)
        g["S1_heuristic"] = feats.mean(axis=1)

        # S3: Mahalanobis distance from this year's mean crime profile
        mean = feats.mean(axis=0)
        cov = np.cov(feats, rowvar=False) + np.eye(len(CRIME_COLUMNS)) * 1e-6
        inv_cov = np.linalg.pinv(cov)
        diffs = feats - mean
        maha_raw = np.sqrt(np.einsum("ij,jk,ik->i", diffs, inv_cov, diffs))
        g["S3_mahalanobis_raw"] = maha_raw
        g["S3_statistical"] = percentile_score(pd.Series(maha_raw, index=g.index))

        rows.append(g)
    return pd.concat(rows, ignore_index=False)


def build_autoencoder(n_features: int) -> keras.Model:
    inputs = keras.Input(shape=(n_features,))
    x = keras.layers.Dense(8, activation="relu")(inputs)
    bottleneck = keras.layers.Dense(3, activation="relu", name="bottleneck")(x)
    d = keras.layers.Dense(8, activation="relu")(bottleneck)
    out = keras.layers.Dense(n_features, activation="linear")(d)
    model = keras.Model(inputs, out)
    model.compile(optimizer=keras.optimizers.Adam(1e-3), loss="mse")
    return model


def main():
    df = pd.read_csv("cleaned_crime_data.csv")
    frame = build_features(df)

    feature_cols = [c + "_norm" for c in CRIME_COLUMNS]
    X = frame[feature_cols].to_numpy(dtype="float32") / 100.0

    n = len(X)
    idx = np.random.permutation(n)
    split = int(n * 0.85)
    train_idx, val_idx = idx[:split], idx[split:]

    ae = build_autoencoder(len(feature_cols))
    early_stop = keras.callbacks.EarlyStopping(monitor="val_loss", patience=15, restore_best_weights=True)
    ae.fit(
        X[train_idx], X[train_idx],
        validation_data=(X[val_idx], X[val_idx]),
        epochs=300, batch_size=32, verbose=0, callbacks=[early_stop],
    )
    val_mse = ae.evaluate(X[val_idx], X[val_idx], verbose=0)
    print(f"Autoencoder reconstruction val MSE: {val_mse:.5f}")

    recon = ae.predict(X, verbose=0)
    recon_error = np.mean((X - recon) ** 2, axis=1)
    frame["S2_anomaly_raw"] = recon_error
    frame["S2_anomaly"] = frame.groupby("year")["S2_anomaly_raw"].transform(
        lambda s: percentile_score(s)
    )

    # ---- PCA fusion of S1, S2, S3 (standardized, pooled across all years) ----
    S = frame[["S1_heuristic", "S2_anomaly", "S3_statistical"]].to_numpy(dtype="float64")
    mean_, std_ = S.mean(axis=0), S.std(axis=0)
    std_[std_ == 0] = 1.0
    Sz = (S - mean_) / std_

    cov = np.cov(Sz, rowvar=False)
    eigvals, eigvecs = np.linalg.eigh(cov)
    order = np.argsort(eigvals)[::-1]
    eigvals, eigvecs = eigvals[order], eigvecs[:, order]
    pc1 = eigvecs[:, 0]

    # sign-correct so higher PC1 == higher S1 (keeps "higher = riskier" intuitive)
    if np.corrcoef(Sz @ pc1, Sz[:, 0])[0, 1] < 0:
        pc1 = -pc1

    frame["ensemble_raw"] = Sz @ pc1
    frame["CRI_ensemble"] = frame.groupby("year")["ensemble_raw"].transform(
        lambda s: percentile_score(s)
    )

    explained_var_ratio = float(eigvals[0] / eigvals.sum())
    corr_s1 = float(np.corrcoef(frame["ensemble_raw"], frame["S1_heuristic"])[0, 1])
    corr_s2 = float(np.corrcoef(frame["ensemble_raw"], frame["S2_anomaly"])[0, 1])
    corr_s3 = float(np.corrcoef(frame["ensemble_raw"], frame["S3_statistical"])[0, 1])
    print(f"PC1 explains {explained_var_ratio:.1%} of variance across the 3 signals")
    print(f"Ensemble correlation -> S1 heuristic: {corr_s1:.3f} | S2 anomaly: {corr_s2:.3f} | S3 statistical: {corr_s3:.3f}")

    # Where does the ensemble disagree most with the plain heuristic? (sanity check / interesting cases)
    frame["disagreement"] = (frame["CRI_ensemble"] - frame["S1_heuristic"]).abs()
    top_disagree = frame.sort_values("disagreement", ascending=False)[
        ["state", "year", "S1_heuristic", "S2_anomaly", "S3_statistical", "CRI_ensemble"]
    ].head(5)
    print("\nTop 5 state-years where ensemble diverges most from the plain heuristic:")
    print(top_disagree.to_string(index=False))

    ae.save("cri_autoencoder.keras")
    meta = {
        "feature_cols": CRIME_COLUMNS,
        "pca_mean": mean_.tolist(),
        "pca_std": std_.tolist(),
        "pca_pc1": pc1.tolist(),
        "explained_variance_ratio": explained_var_ratio,
        "correlation_with_heuristic": corr_s1,
        "correlation_with_anomaly_signal": corr_s2,
        "correlation_with_statistical_signal": corr_s3,
        "autoencoder_val_mse": float(val_mse),
    }
    with open("cri_ensemble_meta.json", "w") as f:
        json.dump(meta, f, indent=2)
    print("\nSaved cri_autoencoder.keras + cri_ensemble_meta.json")


if __name__ == "__main__":
    main()
