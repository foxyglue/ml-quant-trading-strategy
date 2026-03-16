"""
data_loader.py
==============
Responsible for downloading, cleaning, caching, and loading
historical OHLCV stock data from Yahoo Finance.

Pipeline role
-------------
This is the entry point of the pipeline. All downstream modules
(features.py, backtest.py) consume the DataFrame produced here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd
import yfinance as yf

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DATA_DIR = Path(__file__).resolve().parents[1] / "data"
REQUIRED_COLUMNS = {"Open", "High", "Low", "Close", "Volume"}
MIN_ROWS = 60  # need at least 60 trading days for indicators like SMA-50


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def download_stock_data(
    ticker: str,
    start_date: str,
    end_date: str,
    *,
    force_download: bool = False,
) -> pd.DataFrame:
    """Download historical OHLCV data for a single ticker from Yahoo Finance.

    Results are cached to ``data/<TICKER>_<start>_<end>.csv``. Subsequent
    calls with the same arguments return the cached file unless
    ``force_download=True``.

    Parameters
    ----------
    ticker:
        Ticker symbol, e.g. ``"AAPL"`` or ``"SPY"``.
    start_date:
        Inclusive start date in ``YYYY-MM-DD`` format.
    end_date:
        Inclusive end date in ``YYYY-MM-DD`` format.
    force_download:
        When ``True``, bypass the cache and re-download from Yahoo Finance.

    Returns
    -------
    pd.DataFrame
        Cleaned OHLCV DataFrame with a ``DatetimeIndex``.

    Raises
    ------
    ValueError
        If ``ticker`` is empty, dates are malformed, or the download returns
        no data.
    RuntimeError
        If the Yahoo Finance request fails unexpectedly.
    """
    ticker = ticker.strip().upper()
    _validate_inputs(ticker, start_date, end_date)

    cache_path = _cache_path(ticker, start_date, end_date)

    if cache_path.exists() and not force_download:
        log.info("Cache hit — loading %s from %s", ticker, cache_path)
        df = load_data(cache_path)
        return df

    log.info(
        "Downloading %s  [%s → %s] from Yahoo Finance…",
        ticker, start_date, end_date,
    )
    try:
        raw = yf.download(
            ticker,
            start=start_date,
            end=end_date,
            auto_adjust=True,   # adjusts OHLC for splits/dividends, kalau false Close price ga berubah tapi Open/High/Low bisa jadi aneh setelah split
            progress=False,
        )
    except Exception as exc:
        raise RuntimeError(
            f"Yahoo Finance request failed for '{ticker}': {exc}"
        ) from exc

    if raw.empty:
        raise ValueError(
            f"No data returned for ticker '{ticker}' "
            f"between {start_date} and {end_date}. "
            "Check the ticker symbol and date range."
        )

    df = clean_data(raw)
    save_data(df, cache_path)
    log.info("Downloaded %d rows — cached to %s", len(df), cache_path)
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """Validate and clean a raw OHLCV DataFrame.

    Steps applied (in order):

    1. **Flatten MultiIndex columns** - ``yf.download`` with a single ticker
       returns single-level columns, but with multiple tickers it returns a
       MultiIndex. This step normalises both shapes.
    2. **Column presence check** - raises ``ValueError`` if any required
       OHLCV column is missing.
    3. **Index coercion** - ensures the index is a timezone-naive
       ``DatetimeIndex`` sorted ascending.
    4. **Duplicate removal** - drops duplicate index entries, keeping the
       last occurrence.
    5. **Forward-fill + drop** - forward-fills up to 3 consecutive missing
       trading days (e.g. bank holidays), then drops any remaining NaNs.
    6. **Non-positive price guard** - removes rows where ``Close <= 0``
       (data corruption sentinel).
    7. **Minimum length check** - raises ``ValueError`` if fewer than
       ``MIN_ROWS`` rows remain after cleaning.

    Parameters
    ----------
    df:
        Raw DataFrame as returned by ``yf.download``.

    Returns
    -------
    pd.DataFrame
        Cleaned DataFrame ready for feature engineering.

    Raises
    ------
    ValueError
        If required columns are missing, or the cleaned frame is too short.
    """
    df = df.copy()

    # 1. Flatten MultiIndex (yfinance >= 0.2 with multi-ticker downloads)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Normalise column names: strip whitespace, title-case
    df.columns = [str(c).strip().title() for c in df.columns]
    log.info(f"Columns after normalization: {df.columns.tolist()}")

    # 2. Required columns
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(f"Downloaded data is missing columns: {missing}")

    # Keep only the standard OHLCV columns (drop Dividends, Stock Splits, etc.)
    df = df[sorted(REQUIRED_COLUMNS)].copy()
    log.info(f"Columns after filtering required columns: {df.columns.tolist()}")

    # 3. Index coercion
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.index.name = "Date"
    df.sort_index(inplace=True) # sort berdasarkan index (tanggal) ascending

    # 4. Duplicate index entries
    n_dupes = df.index.duplicated().sum()
    if n_dupes:
        log.warning("Dropping %d duplicate index entries.", n_dupes)
        df = df[~df.index.duplicated(keep="last")]

    # 5. Forward-fill short gaps, then drop remaining NaNs
    n_before = len(df)
    df.ffill(limit=3, inplace=True) # forward-fill up to 3 consecutive missing rows (e.g. bank holidays / weekends)
    df.dropna(inplace=True)
    n_dropped = n_before - len(df)
    if n_dropped:
        log.warning("Dropped %d rows containing NaN values.", n_dropped)

    # 6. Non-positive close price
    bad_price = df["Close"] <= 0
    if bad_price.any():
        log.warning(
            "Removing %d rows with non-positive Close price.", bad_price.sum()
        )
        df = df[~bad_price]

    # 7. Minimum length
    if len(df) < MIN_ROWS:
        raise ValueError(
            f"Only {len(df)} rows remain after cleaning "
            f"(minimum required: {MIN_ROWS}). "
            "Extend the date range or choose a different ticker."
        )

    return df


def save_data(df: pd.DataFrame, path: str | Path) -> None:
    """Persist a cleaned OHLCV DataFrame to a CSV file.

    The parent directory is created automatically if it does not exist.

    Parameters
    ----------
    df:
        DataFrame to save. Must have a ``DatetimeIndex`` named ``"Date"``.
    path:
        Destination file path. Should end in ``.csv``.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=True)
    log.info("Saved %d rows to %s", len(df), path)


def load_data(path: str | Path) -> pd.DataFrame:
    """Load a previously cached OHLCV CSV back into a cleaned DataFrame.

    Parameters
    ----------
    path:
        Path to a CSV file previously written by :func:`save_data`.

    Returns
    -------
    pd.DataFrame
        DataFrame with a timezone-naive ``DatetimeIndex`` named ``"Date"``.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Cached data file not found: {path}")

    df = pd.read_csv(path, index_col="Date", parse_dates=True)
    df.index = pd.to_datetime(df.index).tz_localize(None)
    df.sort_index(inplace=True)
    return df


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _validate_inputs(ticker: str, start_date: str, end_date: str) -> None:
    """Raise ``ValueError`` early if arguments are obviously invalid."""
    if not ticker:
        raise ValueError("ticker must be a non-empty string.")

    try:
        start = pd.Timestamp(start_date)
        end = pd.Timestamp(end_date)
    except Exception as exc:
        raise ValueError(
            f"Invalid date format. Expected YYYY-MM-DD, got: "
            f"start='{start_date}', end='{end_date}'"
        ) from exc

    if start >= end:
        raise ValueError(
            f"start_date ({start_date}) must be strictly before "
            f"end_date ({end_date})."
        )

    if end > pd.Timestamp.today():
        log.warning(
            "end_date %s is in the future — Yahoo Finance will cap at today.",
            end_date,
        )


def _cache_path(ticker: str, start_date: str, end_date: str) -> Path:
    """Return the canonical cache file path for a given request."""
    fname = f"{ticker}_{start_date}_{end_date}.csv"
    return DATA_DIR / fname


# ---------------------------------------------------------------------------
# Quick smoke test  (python src/data_loader.py)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    df = download_stock_data("AAPL", "2018-01-01", "2024-01-01")
    print(df.tail())
    print(f"\nShape : {df.shape}")
    print(f"Dtypes:\n{df.dtypes}")
    print(f"NaNs  : {df.isna().sum().sum()}")