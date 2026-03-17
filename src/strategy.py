"""
strategy.py
===========
Converts model probability outputs into discrete trading signals.

This module sits between model.py (produces probabilities) and backtest.py
(consumes signals). It contains only decision logic — no money, no prices,
no portfolio state. Those concerns belong entirely to backtest.py.

Signal encoding
---------------
 +1  →  LONG   (buy / hold the stock)
  0  →  FLAT   (hold cash, no position)
 -1  →  SHORT  (sell short — optional, disabled by default)

Pipeline role
-------------
model.py  →  predict_proba()  →  strategy.py  →  signals  →  backtest.py
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BUY_THRESHOLD: float = 0.55   # P(UP) must exceed this to go long
DEFAULT_SELL_THRESHOLD: float = 0.45  # P(UP) below this triggers short/flat
# Antara 0.45 dan 0.55 adalah "dead zone" di mana model dianggap kurang yakin sehingga tetap flat (0) untuk menghindari overtrading. 
# Semakin lebar "dead zone" semakin sedikit trade tapi kualitas sinyal lebih tinggi, bisa disesuaikan dgn karakteristik model dan toleransi risiko.

# ---------------------------------------------------------------------------
# Core signal generation
# ---------------------------------------------------------------------------

def generate_signals(
    probabilities: pd.Series | np.ndarray,
    index: pd.Index | None = None,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    sell_threshold: float = DEFAULT_SELL_THRESHOLD,
    allow_short: bool = False,
) -> pd.Series:
    """Convert model probabilities into a discrete signal series.

    Decision rules
    --------------
    P(UP) > buy_threshold                    →  +1  (go long)
    P(UP) < sell_threshold and allow_short   →  -1  (go short)
    otherwise                                →   0  (stay flat / cash)

    The dead zone between ``sell_threshold`` and ``buy_threshold`` (default:
    0.45-0.55) forces the model to stay in cash when it is uncertain. This
    reduces overtrading and filters out low-confidence noise.

    Parameters
    ----------
    probabilities:
        1-D array of P(UP) values, one per trading day, as returned by
        ``model.predict_proba(X_scaled)[:, 1]``.
    index:
        DatetimeIndex to attach to the output Series. If ``probabilities``
        is already a pd.Series with an index, that index is used instead.
    buy_threshold:
        Minimum P(UP) required to emit a +1 signal. Default 0.55.
    sell_threshold:
        Maximum P(UP) below which a -1 signal is emitted (only when
        ``allow_short=True``). Default 0.45.
    allow_short:
        When False (default), the strategy is long-only: signals are +1
        or 0 only. When True, signals can also be -1.

    Returns
    -------
    pd.Series
        Integer signal series (dtype int8) with the same index as the input.

    Raises
    ------
    ValueError
        If thresholds are out of range or sell_threshold >= buy_threshold.
    """
    _validate_thresholds(buy_threshold, sell_threshold)

    # Normalise input to a plain numpy array
    if isinstance(probabilities, pd.Series):
        idx = probabilities.index if index is None else index
        prob_arr = probabilities.values
    else:
        prob_arr = np.asarray(probabilities, dtype=float)
        idx = index

    if prob_arr.ndim != 1:
        raise ValueError(
            f"probabilities must be 1-D, got shape {prob_arr.shape}."
        )

    # Build signal array: default flat (0)
    signals = np.zeros(len(prob_arr), dtype=np.int8)
    signals[prob_arr > buy_threshold] = 1

    if allow_short:
        signals[prob_arr < sell_threshold] = -1
    # allow_short = False, semua yang tidak > buy_threshold tetap 0 (flat), tidak pernah -1 (short), 
    # allow_short = True, yang < sell_threshold jadi -1 (short), yang di antara sell_threshold dan buy_threshold tetap 0 (flat).

    signal_series = pd.Series(signals, index=idx, name="signal", dtype="int8")

    _log_signal_summary(signal_series, buy_threshold, sell_threshold, allow_short)
    return signal_series


# ---------------------------------------------------------------------------
# Position management
# ---------------------------------------------------------------------------

def apply_position_filter(
    signals: pd.Series,
    *,
    min_holding_days: int = 1,
) -> pd.Series:
    """Suppress signals that would flip position too rapidly.

    After a BUY (+1) signal, the strategy must hold for at least
    ``min_holding_days`` before a new signal can change the position.
    This reduces transaction costs caused by rapid signal oscillation.

    A ``min_holding_days`` of 1 (default) means no filtering — every
    signal is accepted. Values of 2-5 are typical for daily strategies.
    
    Dengan `min_holding_days=1` (default), keduanya **identik** — tidak ada filtering sama sekali. 
    Perbedaan baru muncul kalau set misalnya `min_holding_days=3`:
    ```
    hari    signal    position (min_holding=3)
    Mon       1           1      ← beli
    Tue       0           1      ← signal mau flat, tapi belum 3 hari → tetap hold
    Wed       0           1      ← hari ke-2, masih hold
    Thu       0           0      ← hari ke-3, baru boleh flat
    ```

    Parameters
    ----------
    signals:
        Raw signal series from ``generate_signals``.
    min_holding_days:
        Minimum number of days to hold a position before switching.

    Returns
    -------
    pd.Series
        Filtered signal series with the same index.
    """
    if min_holding_days <= 1:
        return signals  # nothing to filter

    filtered = signals.copy()
    last_change_idx = 0

    for i in range(1, len(filtered)):
        days_since_change = i - last_change_idx
        if filtered.iloc[i] != filtered.iloc[i - 1]:
            if days_since_change < min_holding_days:
                filtered.iloc[i] = filtered.iloc[i - 1]  # hold previous
            else:
                last_change_idx = i

    n_suppressed = (signals != filtered).sum()
    if n_suppressed:
        log.info(
            "Position filter (min_holding=%d): suppressed %d signal flips.",
            min_holding_days, n_suppressed,
        )
    return filtered


# ---------------------------------------------------------------------------
# Full strategy pipeline (called by main.py / backtest.py)
# ---------------------------------------------------------------------------

def run_strategy(
    model: object,
    scaler: object,
    X_test: pd.DataFrame,
    *,
    buy_threshold: float = DEFAULT_BUY_THRESHOLD,
    sell_threshold: float = DEFAULT_SELL_THRESHOLD,
    allow_short: bool = False,
    min_holding_days: int = 1,
) -> pd.DataFrame:
    """Generate a signal DataFrame from a trained model and test features.

    This is the primary entry point called by ``main.py``. It wraps
    probability generation, signal thresholding, and position filtering
    into a single call and returns a DataFrame that backtest.py can
    consume directly.

    Parameters
    ----------
    model:
        Trained classifier from ``model.train_model``.
    scaler:
        StandardScaler fitted on training data (from ``model.train_model``).
    X_test:
        Feature matrix for the test period (DatetimeIndex required).
    buy_threshold:
        P(UP) threshold to emit a BUY signal.
    sell_threshold:
        P(UP) threshold to emit a SHORT signal (only when allow_short=True).
    allow_short:
        Enable short selling signals (-1).
    min_holding_days:
        Minimum days to hold before flipping position.

    Returns
    -------
    pd.DataFrame
        Columns: ``probability``, ``signal``, ``position``
        Index   : same DatetimeIndex as X_test.

        ``signal``   — raw signal before position filter
        ``position`` — final position after filter (this is what backtest uses)
    """
    # 1. Scale features (transform only — scaler was fit on train)
    X_scaled = scaler.transform(X_test)

    # 2. Raw probabilities P(UP)
    prob_array = model.predict_proba(X_scaled)[:, 1]
    probabilities = pd.Series(prob_array, index=X_test.index, name="probability")

    # 3. Threshold → raw signals
    raw_signals = generate_signals(
        probabilities,
        buy_threshold=buy_threshold,
        sell_threshold=sell_threshold,
        allow_short=allow_short,
    )

    # 4. Position filter
    positions = apply_position_filter(raw_signals, min_holding_days=min_holding_days)
    positions.name = "position"

    # 5. Assemble output DataFrame
    signal_df = pd.concat([probabilities, raw_signals, positions], axis=1)
    signal_df.columns = ["probability", "signal", "position"]

    log.info(
        "Strategy output: %d days  |  long=%d  flat=%d  short=%d",
        len(signal_df),
        (signal_df["position"] == 1).sum(),
        (signal_df["position"] == 0).sum(),
        (signal_df["position"] == -1).sum(),
    )
    return signal_df


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------

def summarise_signals(signal_df: pd.DataFrame) -> dict[str, float]:
    """Return a summary dict of signal statistics.

    Useful for sanity-checking that signals are not degenerate (e.g. all +1,
    which would mean the model is not filtering at all, or all 0, which would
    mean the thresholds are too aggressive).

    Parameters
    ----------
    signal_df:
        DataFrame produced by ``run_strategy`` (must contain ``position``
        and ``probability`` columns).

    Returns
    -------
    dict[str, float]
        Keys: ``long_pct``, ``flat_pct``, ``short_pct``,
              ``avg_probability``, ``total_days``.
    """
    pos = signal_df["position"]
    total = len(pos)

    summary = {
        "total_days":      float(total),
        "long_pct":        round((pos == 1).sum() / total, 4),
        "flat_pct":        round((pos == 0).sum() / total, 4),
        "short_pct":       round((pos == -1).sum() / total, 4),
        "avg_probability": round(float(signal_df["probability"].mean()), 4),
    }

    log.info(
        "Signal summary — long: %.1f%%  flat: %.1f%%  short: %.1f%%  "
        "avg P(UP): %.3f",
        summary["long_pct"] * 100,
        summary["flat_pct"] * 100,
        summary["short_pct"] * 100,
        summary["avg_probability"],
    )
    return summary


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_thresholds(buy: float, sell: float) -> None:
    if not (0 < buy <= 1):
        raise ValueError(f"buy_threshold must be in (0, 1], got {buy}.")
    if not (0 <= sell < 1):
        raise ValueError(f"sell_threshold must be in [0, 1), got {sell}.")
    if sell >= buy:
        raise ValueError(
            f"sell_threshold ({sell}) must be strictly less than "
            f"buy_threshold ({buy})."
        )


def _log_signal_summary(
    signals: pd.Series,
    buy_th: float,
    sell_th: float,
    allow_short: bool,
) -> None:
    total = len(signals)
    n_long  = (signals ==  1).sum()
    n_flat  = (signals ==  0).sum()
    n_short = (signals == -1).sum()
    log.info(
        "Signals generated  |  buy_th=%.2f  sell_th=%.2f  short=%s  "
        "→  long=%d (%.1f%%)  flat=%d (%.1f%%)  short=%d (%.1f%%)",
        buy_th, sell_th, allow_short,
        n_long,  n_long  / total * 100,
        n_flat,  n_flat  / total * 100,
        n_short, n_short / total * 100,
    )


# ---------------------------------------------------------------------------
# Smoke test  (python src/strategy.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent))

    import logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s  %(levelname)-8s  %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    from data_loader import download_stock_data
    from features import build_features, split_features_target
    from model import train_model, time_series_split

    # 1. Data + features
    raw = download_stock_data("AAPL", "2018-01-01", "2024-01-01")
    df_feat = build_features(raw)
    X, y = split_features_target(df_feat)

    # 2. Split + train
    X_train, X_test, y_train, y_test = time_series_split(X, y)
    model, scaler = train_model(X_train, y_train)

    # 3. Run strategy
    signal_df = run_strategy(model, scaler, X_test)

    print("\nSignal DataFrame (first 10 rows):")
    print(signal_df.head(10).to_string())

    # 4. Sanity check — signals must not be all the same value
    summary = summarise_signals(signal_df)
    print(f"\nSummary: {summary}")

    unique_positions = signal_df["position"].unique()
    assert len(unique_positions) > 1, (
        f"Degenerate signals — only one unique value: {unique_positions}. "
        "Adjust buy_threshold."
    )
    print("\nSanity check passed: signals are non-degenerate.")