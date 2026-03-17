"""
features.py
===========
Feature engineering pipeline: transforms a clean OHLCV DataFrame into a
feature matrix (X) and a binary target vector (y) ready for ML training.

Feature groups
--------------
- Momentum     : RSI-14
- Trend        : SMA-20, SMA-50, EMA-20, price-to-MA ratios
- MACD         : MACD line, signal line, histogram
- Volatility   : rolling 20-day close-return std, ATR-14
- Lagged returns: close returns at lags 1-5
- Target (y)   : 1 if next-day close > today's close, else 0

Pipeline role
-------------
Receives the cleaned DataFrame from data_loader.py.
Produces the (X, y) pair consumed by model.py.
"""

from __future__ import annotations

import logging

import numpy as np
import pandas as pd
import pandas_ta as ta

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
RSI_PERIOD: int = 14 # periode standar utk RSI (Relative Strength Index), mengukur kekuatan dan kecepatan perubahan harga baru-baru ini, dengan nilai 0-100.  Di atas 70 biasanya dianggap overbought (potensi sell), di bawah 30 oversold (potensi buy).
SMA_SHORT: int = 20 # periode standar utk SMA pendek, mengukur rata-rata harga dalam 20 hari terakhir
SMA_LONG: int = 50 # periode standar utk SMA panjang, mengukur rata-rata harga dalam 50 hari terakhir

EMA_SHORT: int = 20 # periode standar utk EMA pendek, mengukur rata-rata eksponensial harga dalam 20 hari terakhir
MACD_FAST: int = 12 # periode standar utk EMA cepat dalam MACD, mengukur rata-rata eksponensial harga dalam 12 hari terakhir
MACD_SLOW: int = 26 # periode standar utk EMA lambat dalam MACD, mengukur rata-rata eksponensial harga dalam 26 hari terakhir
MACD_SIGNAL: int = 9 # periode standar utk signal line dalam MACD, mengukur rata-rata eksponensial MACD line dalam 9 hari terakhir
# macd 12 26 vs 9 → MACD line = EMA12 - EMA26, signal line = EMA9(MACD line), histogram = MACD line - signal line.  Histogram menunjukkan kekuatan momentum: positif dan besar → momentum naik kuat, negatif dan besar → momentum turun kuat.

VOL_PERIOD: int = 20 # periode standar utk volatilitas, mengukur deviasi standar log returns selama 20 hari
ATR_PERIOD: int = 14 # average true range, mengukur volatilitas intraday yang memperhitungkan gap open, lebih realistis daripada rolling std untuk menangkap regime volatilitas
LAG_PERIODS: list[int] = [1, 2, 3, 4, 5] # memberikan model akses eksplisit ke return 1-5 hari yang lalu, tanpa RNN atau lainnya


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame) -> pd.DataFrame:
    """Master function: attach all features and the target to ``df``.

    Calls each sub-function in sequence, then drops rows that contain NaN
    (warm-up rows produced by long-window indicators like SMA-50).

    Parameters
    ----------
    df:
        Cleaned OHLCV DataFrame from ``data_loader.clean_data``.
        Must contain columns: Open, High, Low, Close, Volume.

    Returns
    -------
    pd.DataFrame
        Original OHLCV columns **plus** all engineered features and the
        ``target`` column. The last row is dropped because its target value
        would require a future close price.

    Raises
    ------
    ValueError
        If required OHLCV columns are absent.
    """
    _check_ohlcv(df)
    out = df.copy()

    out = add_momentum_features(out)
    out = add_trend_features(out)
    out = add_macd_features(out)
    out = add_volatility_features(out)
    out = add_lagged_returns(out)
    out = add_target(out)

    # Drop warm-up NaNs introduced by slow indicators (e.g. SMA-50)
    n_before = len(out)
    out.dropna(inplace=True)

    # The final row has a valid target only if we know the *next* day's close.
    # After dropna, drop the last row so target is always a realised value.
    out = out.iloc[:-1]

    n_dropped = n_before - len(out)
    log.info(
        "build_features: %d rows in → %d rows out (%d dropped for NaN/target).",
        n_before, len(out), n_dropped,
    )
    return out


def add_momentum_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append RSI-14 to ``df``.

    RSI (Relative Strength Index) measures the speed and magnitude of recent
    price changes on a 0-100 scale. Values above 70 conventionally signal
    overbought conditions; below 30, oversold. It is a bounded oscillator,
    which makes it a well-behaved ML feature without further normalisation.
    
    RSI < 30 → potensi buy
    
    RSI > 70 → potensi sell

    New column
    ----------
    ``rsi_14``

    Parameters
    ----------
    df:
        DataFrame with a ``Close`` column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``rsi_14`` appended in-place on a copy.
    """
    out = df.copy()
    out["rsi_14"] = ta.rsi(out["Close"], length=RSI_PERIOD) # menghitung RSI dengan periode 14 hari, menggunakan harga Close
    return out


def add_trend_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append moving-average features to ``df``.

    Raw MA values are price-scale-dependent and non-stationary.  Instead of
    using them directly, we express each MA as the *ratio* of the current
    close to the MA value:

        price_to_sma20 = Close / SMA20

    A value > 1 means price is above the MA (bullish); < 1 means below
    (bearish). These ratios are stationary and comparable across tickers.

    New columns
    -----------
    ``sma_20``, ``sma_50``, ``ema_20``,
    ``price_to_sma20``, ``price_to_sma50``, ``price_to_ema20``,
    ``sma20_sma50_ratio``  (golden-cross proxy)

    Parameters
    ----------
    df:
        DataFrame with a ``Close`` column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with trend columns appended.
    """
    out = df.copy()

    out["sma_20"] = ta.sma(out["Close"], length=SMA_SHORT)
    out["sma_50"] = ta.sma(out["Close"], length=SMA_LONG)
    out["ema_20"] = ta.ema(out["Close"], length=EMA_SHORT)

    # Ratios: stationary, scale-free trend signals
    out["price_to_sma20"] = out["Close"] / out["sma_20"]
    out["price_to_sma50"] = out["Close"] / out["sma_50"]
    out["price_to_ema20"] = out["Close"] / out["ema_20"]

    # Golden/death cross proxy: SMA20 relative to SMA50
    out["sma20_sma50_ratio"] = out["sma_20"] / out["sma_50"]

    return out


def add_macd_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append MACD line, signal line, and histogram to ``df``.

    MACD = EMA(fast) - EMA(slow).  The *histogram* (MACD - signal) is
    often the most predictive component: its sign shows momentum direction
    and its magnitude shows momentum strength.

    pandas-ta returns a DataFrame with three columns; this function renames
    them to ``macd_line``, ``macd_signal``, ``macd_hist`` for clarity.

    New columns
    -----------
    ``macd_line``, ``macd_signal``, ``macd_hist``

    Parameters
    ----------
    df:
        DataFrame with a ``Close`` column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with MACD columns appended.
    """
    out = df.copy()

    macd_df = ta.macd(
        out["Close"],
        fast=MACD_FAST,
        slow=MACD_SLOW,
        signal=MACD_SIGNAL,
        talib=False, # kalau true, akan menghasilkan NaN karena pandas-ta tidak bisa menemukan TA-Lib di lingkungan ini.  Dengan talib=False, pandas-ta menggunakan implementasi internal yang bekerja dengan baik.
    )
    # return DataFrame dengan kolom: MACD_12_26_9 (line), MACDh_12_26_9 (histogram), MACDs_12_26_9 (signal)

    # pandas-ta column names: MACD_12_26_9, MACDh_12_26_9, MACDs_12_26_9
    col_map = {
        macd_df.columns[0]: "macd_line",
        macd_df.columns[2]: "macd_signal",
        macd_df.columns[1]: "macd_hist",
    }
    macd_df = macd_df.rename(columns=col_map)

    out = pd.concat([out, macd_df[["macd_line", "macd_signal", "macd_hist"]]], axis=1)
    return out


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Append rolling volatility and ATR to ``df``.

    Two complementary volatility measures:

    * **Rolling std of returns** — captures how noisy the close-to-close
      returns have been over the past 20 days.  Computed on log returns for
      better statistical properties.
    * **ATR-14** (Average True Range) — an intraday measure that accounts
      for gap opens.  Normalised by the close price (``atr_pct``) so it is
      comparable across price levels.

    High-volatility regimes tend to exhibit different return dynamics, so
    these features allow the model to condition its predictions on the
    current noise level.

    New columns
    -----------
    ``volatility_20``, ``atr_14``, ``atr_pct``

    Parameters
    ----------
    df:
        DataFrame with ``High``, ``Low``, ``Close`` columns.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with volatility columns appended.
    """
    out = df.copy()

    log_returns = np.log(out["Close"] / out["Close"].shift(1))
    out["volatility_20"] = log_returns.rolling(window=VOL_PERIOD).std() # menghitung volatilitas sebagai rolling std dev dari log returns selama 20 hari, menangkap seberapa bergejolak harga baru-baru ini

    out["atr_14"] = ta.atr(out["High"], out["Low"], out["Close"], length=ATR_PERIOD)
    out["atr_pct"] = out["atr_14"] / out["Close"]  # scale-free

    return out


def add_lagged_returns(
    df: pd.DataFrame,
    lags: list[int] | None = None,
) -> pd.DataFrame:
    """Append lagged close-to-close returns to ``df``.

    Each lagged return ``return_lag_{k}`` is today's view of what happened
    *k* trading days ago:

        return_lag_1[t] = (Close[t] - Close[t-1]) / Close[t-1]

    These give the model explicit short-term memory without requiring a
    recurrent architecture.  Computed as simple percentage returns (not log
    returns) to keep the scale linear and easy to interpret.

    New columns
    -----------
    ``return_lag_1`` … ``return_lag_5``  (or whatever ``lags`` specifies)

    Parameters
    ----------
    df:
        DataFrame with a ``Close`` column.
    lags:
        List of integer lag periods to include.  Defaults to ``[1, 2, 3, 4, 5]``.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with lagged return columns appended.
    """
    out = df.copy()
    lags = lags if lags is not None else LAG_PERIODS

    daily_return = out["Close"].pct_change() # daily_return[t] = (Close[t] - Close[t-1]) / Close[t-1], memberikan gambaran tentang return harian dalam bentuk persentase
    for k in lags:
        out[f"return_lag_{k}"] = daily_return.shift(k) # return_lag_k[t] = daily_return[t-k]. 'return_lag_1' menjawab "Apa persentase return harian kemarin?"

    return out


def add_target(df: pd.DataFrame) -> pd.DataFrame:
    """Append the binary classification target to ``df``.

    Target definition
    -----------------
    ``target = 1``  if ``Close[t+1] > Close[t]``  (price goes up tomorrow)
    ``target = 0``  otherwise  (price stays flat or falls)

    The target is aligned so that row ``t`` contains everything known *at
    the close of day t* (features) alongside what will happen on day *t+1*
    (label).  The last row necessarily has ``target = NaN`` and is removed
    by ``build_features`` after all features are assembled.

    New column
    ----------
    ``target``  (int8: 0 or 1, with NaN on the final row)

    Parameters
    ----------
    df:
        DataFrame with a ``Close`` column.

    Returns
    -------
    pd.DataFrame
        Input DataFrame with ``target`` appended.
    """
    out = df.copy()
    future_close = out["Close"].shift(-1) # harga penutupan besok, di-shift ke atas sehingga sejajar dengan fitur hari ini dan ldibandingkan dgn harga Close hari ini
    out["target"] = (future_close > out["Close"]).astype("Int8")
    return out


def get_feature_columns(df: pd.DataFrame) -> list[str]:
    """Return the list of feature column names, excluding OHLCV and target.

    Convenience helper used by ``model.py`` to build ``X`` without having
    to hard-code column names.

    Parameters
    ----------
    df:
        DataFrame produced by ``build_features``.

    Returns
    -------
    list[str]
        Ordered list of feature column names.
    """
    exclude = {"Open", "High", "Low", "Close", "Volume", "target"}
    return [c for c in df.columns if c not in exclude]


def split_features_target(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Split a feature-engineered DataFrame into X and y.

    Parameters
    ----------
    df:
        DataFrame produced by ``build_features``.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all engineered columns, no OHLCV, no target).
    y : pd.Series
        Binary target series aligned with X.
    """
    feature_cols = get_feature_columns(df)
    X = df[feature_cols]
    y = df["target"].astype(int)
    return X, y


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _check_ohlcv(df: pd.DataFrame) -> None:
    required = {"Open", "High", "Low", "Close", "Volume"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"DataFrame is missing required columns: {missing}")


# ---------------------------------------------------------------------------
# Quick smoke test  (python src/features.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from data_loader import download_stock_data

    raw = download_stock_data("AAPL", "2018-01-01", "2024-01-01")
    featured = build_features(raw)

    X, y = split_features_target(featured)

    print(f"Feature matrix : {X.shape}")
    print(f"Target series  : {y.shape}  |  class balance: {y.mean():.1%} up")
    print(f"\nFeature columns ({len(X.columns)}):")
    for col in X.columns:
        print(f"  {col}")
    print(f"\nSample (last 3 rows):\n{featured.tail(3).to_string()}")
    print(f"\nNaNs in X: {X.isna().sum().sum()}")