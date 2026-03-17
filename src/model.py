"""
model.py
========
Training, evaluation, persistence, and loading of the direction-prediction
classifier used in the trading strategy.

Model summary
-------------
Input  : feature matrix X produced by ``features.build_features``
Output : binary label  y  (1 = next-day close UP, 0 = flat / DOWN)

The module tries to import XGBoost and falls back to RandomForestClassifier
if the package is not installed, so the project works in both environments.

Pipeline role
-------------
Sits between features.py (produces X, y) and strategy.py (consumes
``model.predict_proba`` to generate trading signals).
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# XGBoost is optional — fall back to RandomForest gracefully
# ---------------------------------------------------------------------------
try:
    from xgboost import XGBClassifier
    _XGBOOST_AVAILABLE = True
except ImportError:
    _XGBOOST_AVAILABLE = False
    log.info("XGBoost not found — using RandomForestClassifier as default.")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
TRAIN_RATIO: float = 0.80          # chronological split point
MODEL_DIR: Path = Path(__file__).resolve().parents[1] / "models"

# Default hyper-parameters (tuned for daily financial data at this scale)
RF_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 6,
    "min_samples_leaf": 20,        # prevents overfitting on small datasets
    "max_features": "sqrt",
    "class_weight": "balanced",    # handles mild class imbalance
    "random_state": 42,
    "n_jobs": -1,
}

XGB_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 4,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 20,
    "scale_pos_weight": 1,
    "eval_metric": "logloss",
    "random_state": 42,
    "n_jobs": -1,
    "verbosity": 0,
}

# Type alias for the two-element tuple returned by train_model
ModelBundle = tuple[Any, StandardScaler]


# ---------------------------------------------------------------------------
# Time-series split
# ---------------------------------------------------------------------------

def time_series_split(
    X: pd.DataFrame,
    y: pd.Series,
    train_ratio: float = TRAIN_RATIO,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Split X and y chronologically into train and test sets.

    Unlike ``sklearn.model_selection.train_test_split``, this function never
    shuffles.  The first ``train_ratio`` fraction of rows becomes the training
    set and the remainder becomes the test set, preserving temporal order.

    Shuffling a time series leaks future information into training because the
    model sees future feature values (e.g. a high RSI from 2023 sitting next
    to a low RSI from 2019) as if they were contemporaneous.

    Parameters
    ----------
    X:
        Feature matrix with a DatetimeIndex, as produced by
        ``features.split_features_target``.
    y:
        Corresponding binary target series.
    train_ratio:
        Fraction of rows assigned to the training set.  Default 0.80.

    Returns
    -------
    X_train, X_test, y_train, y_test : tuple of DataFrames / Series
    """
    if not 0 < train_ratio < 1:
        raise ValueError(f"train_ratio must be in (0, 1), got {train_ratio}.")

    split_idx = int(len(X) * train_ratio)
    X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
    y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

    log.info(
        "Chronological split — train: %d rows (%s → %s)  |  test: %d rows (%s → %s)",
        len(X_train),
        X_train.index[0].date(), X_train.index[-1].date(),
        len(X_test),
        X_test.index[0].date(), X_test.index[-1].date(),
    )
    return X_train, X_test, y_train, y_test


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    *,
    use_xgboost: bool | None = None,
) -> ModelBundle:
    """Fit a classifier on the training set and return it with its scaler.

    The scaler is **fit only on X_train** and must be applied to X_test via
    ``scaler.transform`` (not ``fit_transform``).  Fitting the scaler on the
    full dataset would leak future distributional information into training.

    Model selection
    ---------------
    * If ``use_xgboost=True`` (and XGBoost is installed) → XGBClassifier
    * If ``use_xgboost=False`` → RandomForestClassifier
    * If ``use_xgboost=None`` (default) → XGBoost when available, else RF

    Parameters
    ----------
    X_train:
        Feature matrix for the training period.
    y_train:
        Binary target series aligned with X_train.
    use_xgboost:
        Override automatic model selection (see above).

    Returns
    -------
    model : fitted classifier
    scaler : StandardScaler fitted on X_train
    """
    # ---- Scaler: fit on train ONLY ----------------------------------------
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_train) # kalau fit_transform di sini, scaler hanya belajar dari distribusi X_train, mencegah data leakage. Kemudian X_scaled digunakan untuk melatih model.

    # ---- Model selection ---------------------------------------------------
    want_xgb = _XGBOOST_AVAILABLE if use_xgboost is None else use_xgboost
    if want_xgb and not _XGBOOST_AVAILABLE:
        log.warning("XGBoost requested but not installed — falling back to RF.")
        want_xgb = False

    if want_xgb:
        model = XGBClassifier(**XGB_PARAMS) # ** membuat dict XGB_PARAMS di-unpack menjadi argumen keyword individual. Contohnya, n_estimators=300, max_depth=4, dll.
        model_name = "XGBClassifier"
    else:
        model = RandomForestClassifier(**RF_PARAMS)
        model_name = "RandomForestClassifier"

    log.info("Training %s on %d samples, %d features…", model_name, *X_scaled.shape)
    model.fit(X_scaled, y_train)
    log.info("Training complete.")

    return model, scaler


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_model(
    model: Any,
    scaler: StandardScaler,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, float]:
    """Evaluate a trained model on the held-out test set.

    Applies the pre-fitted scaler to X_test (transform only, never fit),
    then computes five classification metrics.

    Metrics
    -------
    accuracy
        Fraction of correct predictions overall.
    precision
        Of all "UP" predictions, how many were actually UP.
        High precision → fewer false buy signals.
    recall
        Of all actual UP days, how many did the model catch.
        High recall → fewer missed opportunities.
    f1
        Harmonic mean of precision and recall.  Balances both concerns.
    roc_auc
        Area under the ROC curve, using predicted probabilities.
        Threshold-independent measure of discriminative power.
        0.5 = random; 1.0 = perfect.

    Parameters
    ----------
    model:
        Trained classifier returned by ``train_model``.
    scaler:
        StandardScaler fitted on training data (returned by ``train_model``).
    X_test:
        Feature matrix for the test period.
    y_test:
        True binary labels for the test period.

    Returns
    -------
    dict[str, float]
        Metric name → value, rounded to 4 decimal places.
    """
    X_scaled = scaler.transform(X_test)  # transform ONLY — never fit here
    y_pred = model.predict(X_scaled)
    y_prob = model.predict_proba(X_scaled)[:, 1]  # P(class=1) 
    # predict_proba untuk klasifikasi biner mengembalikan array 2D dengan dua kolom: P(class=0) dan P(class=1). Kita ambil kolom kedua ([:, 1]) untuk mendapatkan probabilitas prediksi bahwa kelasnya adalah 1 (UP). 
    
    # contoh data (sebelum [:, 1]):
    # [[0.80, 0.20],   ← sample 1: 80% DOWN, 20% UP
    #  [0.20, 0.80],   ← sample 2: 20% DOWN, 80% UP
    #  [0.40, 0.60],   ← sample 3: 40% DOWN, 60% UP
    #  [0.90, 0.10]]   ← sample 4: 90% DOWN, 10% UP
    
    # contoh data (sesudah [:, 1]):
    # array: [0.20, 0.80, 0.60, 0.10] -> P(UP) untuk masing-masing sample
    
    metrics = {
        "accuracy":  round(float(accuracy_score(y_test, y_pred)), 4),
        "precision": round(float(precision_score(y_test, y_pred,
                                                 zero_division=0)), 4),
        "recall":    round(float(recall_score(y_test, y_pred,
                                              zero_division=0)), 4),
        "f1":        round(float(f1_score(y_test, y_pred,
                                          zero_division=0)), 4),
        "roc_auc":   round(float(roc_auc_score(y_test, y_prob)), 4), # roc_auc_score membutuhkan probabilitas prediksi (y_prob) untuk menghitung area under the ROC curve. Ini memberikan gambaran tentang seberapa baik model membedakan antara kelas 0 (DOWN) dan kelas 1 (UP) di berbagai threshold.
    }

    _log_metrics(metrics, X_test)
    return metrics


def _log_metrics(metrics: dict[str, float], X_test: pd.DataFrame) -> None:
    """Pretty-print evaluation metrics to the logger."""
    border = "─" * 38
    log.info(border)
    log.info("  Evaluation results  |  test period: %s → %s",
             X_test.index[0].date(), X_test.index[-1].date())
    log.info(border)
    for name, val in metrics.items():
        bar = "█" * int(val * 20)
        log.info("  %-12s  %.4f  %s", name, val, bar)
    log.info(border)


# ---------------------------------------------------------------------------
# Feature importance
# ---------------------------------------------------------------------------

def get_feature_importance(
    model: Any,
    feature_names: list[str],
    top_n: int = 10,
) -> pd.DataFrame:
    """Return a DataFrame of feature importances sorted descending.

    Works with both RandomForest (``feature_importances_``) and XGBoost
    (also exposes ``feature_importances_``).
    
    Feature importances dihitung berdasarkan seberapa banyak setiap fitur digunakan 
    untuk membuat split di dalam pohon-pohon keputusan (untuk RF) atau seberapa besar kontribusi 
    rata-rata setiap fitur terhadap peningkatan kinerja model (untuk XGBoost). Fitur dengan nilai importance 
    lebih tinggi dianggap lebih berpengaruh dalam prediksi model.

    Parameters
    ----------
    model:
        Trained classifier.
    feature_names:
        Column names from the training DataFrame (``X_train.columns.tolist()``).
    top_n:
        Number of top features to return.  Pass ``-1`` for all.

    Returns
    -------
    pd.DataFrame
        Columns: ``feature``, ``importance``.
    """
    if not hasattr(model, "feature_importances_"):
        raise AttributeError(
            f"{type(model).__name__} does not expose feature_importances_."
        )

    importance_df = (
        pd.DataFrame({
            "feature":    feature_names,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )

    if top_n > 0:
        importance_df = importance_df.head(top_n)

    return importance_df


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_model(
    model: Any,
    scaler: StandardScaler,
    path: str | Path,
) -> None:
    """Serialize the model and scaler together to a single ``.joblib`` file.

    Saves both objects as a dict so ``load_model`` can unpack them without
    any ambiguity about ordering.

    Parameters
    ----------
    model:
        Trained classifier.
    scaler:
        StandardScaler fitted on training data.
    path:
        Destination file path.  The parent directory is created if needed.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    bundle = {"model": model, "scaler": scaler}
    joblib.dump(bundle, path)
    log.info("Model bundle saved to %s", path)


def load_model(path: str | Path) -> ModelBundle:
    """Load a model bundle previously saved by ``save_model``.

    Parameters
    ----------
    path:
        Path to a ``.joblib`` file produced by ``save_model``.

    Returns
    -------
    model : fitted classifier
    scaler : StandardScaler

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Model file not found: {path}")

    bundle = joblib.load(path)
    log.info("Loaded model bundle from %s", path)
    return bundle["model"], bundle["scaler"]


# ---------------------------------------------------------------------------
# Convenience wrapper: full train/test pipeline in one call
# ---------------------------------------------------------------------------

def run_training_pipeline(
    df_featured: pd.DataFrame,
    train_ratio: float = TRAIN_RATIO,
    use_xgboost: bool | None = None,
    save_path: str | Path | None = None,
) -> tuple[ModelBundle, dict[str, float], pd.DataFrame, pd.Series]:
    """Run the complete model pipeline on a feature-engineered DataFrame.

    Convenience function used by ``main.py`` to avoid boilerplate.

    Steps
    -----
    1. Split X / y from the featured DataFrame.
    2. Chronological train/test split.
    3. Train model + scaler.
    4. Evaluate on test set.
    5. Optionally save the bundle.

    Parameters
    ----------
    df_featured:
        DataFrame produced by ``features.build_features``.
    train_ratio:
        Fraction of data used for training.
    use_xgboost:
        Model selection override (see ``train_model`` docs).
    save_path:
        If provided, save the trained bundle here.

    Returns
    -------
    (model, scaler) : ModelBundle
    metrics         : dict[str, float]
    X_test          : pd.DataFrame  — for use in strategy / backtest
    y_test          : pd.Series
    """
    from src.features import split_features_target  # local import avoids circularity

    X, y = split_features_target(df_featured)
    X_train, X_test, y_train, y_test = time_series_split(X, y, train_ratio)

    model, scaler = train_model(X_train, y_train, use_xgboost=use_xgboost)
    metrics = evaluate_model(model, scaler, X_test, y_test)

    feat_imp = get_feature_importance(model, X_train.columns.tolist())
    log.info("Top-5 features:\n%s", feat_imp.head(5).to_string(index=False))

    if save_path is not None:
        save_model(model, scaler, save_path)

    return (model, scaler), metrics, X_test, y_test


# ---------------------------------------------------------------------------
# Smoke test  (python src/model.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from data_loader import download_stock_data
    from features import build_features, split_features_target

    # 1. Data
    raw = download_stock_data("AAPL", "2018-01-01", "2024-01-01")

    # 2. Features
    df_feat = build_features(raw)
    X, y = split_features_target(df_feat)
    print(f"\nFull dataset  : {X.shape[0]} rows, {X.shape[1]} features")

    # 3. Chronological split
    X_train, X_test, y_train, y_test = time_series_split(X, y)
    print(f"Train         : {len(X_train)} rows")
    print(f"Test          : {len(X_test)} rows")

    # 4. Train
    model, scaler = train_model(X_train, y_train)

    # 5. Evaluate
    metrics = evaluate_model(model, scaler, X_test, y_test)
    print("\nEvaluation metrics:")
    for k, v in metrics.items():
        print(f"  {k:<12} {v:.4f}")

    # 6. Feature importance
    imp = get_feature_importance(model, X_train.columns.tolist())
    print(f"\nTop-10 features:\n{imp.to_string(index=False)}")

    # 7. Save / reload round-trip
    save_model(model, scaler, MODEL_DIR / "aapl_model.joblib")
    model2, scaler2 = load_model(MODEL_DIR / "aapl_model.joblib")
    metrics2 = evaluate_model(model2, scaler2, X_test, y_test)
    assert metrics == metrics2, "Round-trip mismatch!"
    print("\nSave/load round-trip: OK")