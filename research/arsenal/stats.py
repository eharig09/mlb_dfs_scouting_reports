"""Just enough regression to answer "does this column add anything".

Cluster-robust standard errors are not optional here. Nine hitters in a lineup share a
game, a park, a starter and a weather reading, so their residuals are correlated; treating
them as 118,000 independent observations would shrink every standard error by roughly the
square root of the lineup size and turn noise into significance. Errors are clustered on
game_pk throughout.
"""

import numpy as np
import pandas as pd


def design(frame, numeric, categorical=()):
    """Model matrix with an intercept, dropping one level per categorical."""
    parts = [np.ones((len(frame), 1))]
    names = ["const"]
    for column in numeric:
        parts.append(frame[[column]].to_numpy(dtype=float))
        names.append(column)
    for column in categorical:
        dummies = pd.get_dummies(frame[column].astype("string").fillna("?"),
                                 prefix=column, drop_first=True, dtype=float)
        parts.append(dummies.to_numpy())
        names.extend(dummies.columns)
    return np.hstack(parts), names


def ols(y, X, clusters):
    """Coefficients, cluster-robust standard errors, R-squared."""
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    resid = y - X @ beta
    n, k = X.shape
    xtx_inv = np.linalg.pinv(X.T @ X)

    # Per-cluster score vectors via a segmented sum, then meat = S'S. Looping the clusters
    # in Python costs more than the regression itself at 100k rows.
    codes = pd.factorize(clusters)[0]
    groups = int(codes.max()) + 1
    order = np.argsort(codes, kind="mergesort")
    weighted = X[order] * resid[order][:, None]
    boundaries = np.searchsorted(codes[order], np.arange(groups))
    scores = np.add.reduceat(weighted, boundaries, axis=0)
    meat = scores.T @ scores
    scale = groups / max(1, groups - 1) * (n - 1) / max(1, n - k)
    cov = xtx_inv @ (meat * scale) @ xtx_inv
    se = np.sqrt(np.clip(np.diag(cov), 0, None))

    total = ((y - y.mean()) ** 2).sum()
    r2 = 1 - (resid ** 2).sum() / total if total else np.nan
    return {"beta": beta, "se": se, "r2": r2, "n": n, "clusters": groups, "resid": resid}


def incremental(frame, y_col, controls_numeric, controls_categorical, test_col,
                cluster_col="game_pk"):
    """Fit with and without `test_col`; report what adding it bought.

    `test_col` is standardized first, so the coefficient reads as "DK points per one
    standard deviation of this feature" and is comparable across configs.
    """
    frame = frame.dropna(subset=[y_col, test_col] + list(controls_numeric)).copy()
    if len(frame) < 500:
        return None
    values = frame[test_col].to_numpy(dtype=float)
    spread = values.std()
    frame["_z"] = (values - values.mean()) / (spread if spread else 1.0)

    y = frame[y_col].to_numpy(dtype=float)
    clusters = frame[cluster_col].to_numpy()
    X0, _ = design(frame, controls_numeric, controls_categorical)
    X1, names = design(frame, list(controls_numeric) + ["_z"], controls_categorical)

    base = ols(y, X0, clusters)
    full = ols(y, X1, clusters)
    index = names.index("_z")
    coefficient = full["beta"][index]
    se = full["se"][index]
    return {
        "n": len(frame),
        "games": base["clusters"],
        "coef": coefficient,
        "se": se,
        "t": coefficient / se if se else np.nan,
        "r2_base": base["r2"],
        "r2_full": full["r2"],
        "dr2": full["r2"] - base["r2"],
        "sd": spread,
    }


def walk_forward(frame, y_col, controls_numeric, controls_categorical, test_col,
                 train_seasons, test_season):
    """Fit on earlier seasons, score the later one. The only honest read on whether the
    feature helps a forecast rather than merely fitting the past."""
    frame = frame.dropna(subset=[y_col, test_col] + list(controls_numeric)).copy()
    train = frame[frame["season"].isin(train_seasons)]
    test = frame[frame["season"] == test_season]
    if len(train) < 1000 or len(test) < 500:
        return None

    combined = pd.concat([train, test], ignore_index=True)
    X_all, names = design(combined, list(controls_numeric) + [test_col],
                          controls_categorical)
    split = len(train)
    y_train = train[y_col].to_numpy(dtype=float)
    y_test = test[y_col].to_numpy(dtype=float)

    without = [i for i, name in enumerate(names) if name != test_col]
    results = {}
    for label, columns in (("base", without), ("full", list(range(len(names))))):
        beta, *_ = np.linalg.lstsq(X_all[:split][:, columns], y_train, rcond=None)
        prediction = X_all[split:][:, columns] @ beta
        error = y_test - prediction
        results[f"rmse_{label}"] = float(np.sqrt((error ** 2).mean()))
        results[f"spearman_{label}"] = float(
            pd.Series(prediction).corr(pd.Series(y_test), method="spearman"))
    results["n_train"], results["n_test"] = split, len(test)
    results["drmse"] = results["rmse_full"] - results["rmse_base"]
    results["dspearman"] = results["spearman_full"] - results["spearman_base"]
    return results


def decile_table(frame, y_col, test_col, controls_numeric, controls_categorical,
                 bins=10, partial=True):
    """Mean outcome by bucket of the feature, optionally after removing the controls.

    Partialling matters: the top decile of any arsenal score is full of good hitters, so
    the raw table always slopes upward. What is being asked is whether it still slopes
    once the hitters are equalized.
    """
    frame = frame.dropna(subset=[y_col, test_col] + list(controls_numeric)).copy()
    if partial:
        X, _ = design(frame, controls_numeric, controls_categorical)
        clusters = frame["game_pk"].to_numpy()
        feature = frame[test_col].to_numpy(dtype=float)
        beta, *_ = np.linalg.lstsq(X, feature, rcond=None)
        frame["_x"] = feature - X @ beta
        y = frame[y_col].to_numpy(dtype=float)
        beta_y, *_ = np.linalg.lstsq(X, y, rcond=None)
        frame["_y"] = y - X @ beta_y + y.mean()
    else:
        frame["_x"] = frame[test_col]
        frame["_y"] = frame[y_col]
    frame["bucket"] = pd.qcut(frame["_x"], bins, labels=False, duplicates="drop")
    table = frame.groupby("bucket", observed=True).agg(
        n=("_y", "size"), feature=(test_col, "mean"), outcome=("_y", "mean"),
        raw_outcome=(y_col, "mean")).reset_index()
    return table
