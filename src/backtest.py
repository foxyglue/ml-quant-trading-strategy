"""
backtest.py
===========
Simulates portfolio performance by replaying trading signals against
historical price data. Computes the equity curve and trade log.

This module is purely an accounting engine — it never makes decisions.
All buy/sell logic is resolved upstream in strategy.py. Here, positions
are taken as given and their financial consequences are calculated.

Key design rules
----------------
1. signal[t-1] drives return[t]  — enforces no lookahead bias.
2. Transaction costs are applied on every position change (0→1 or 1→0).
3. The buy-and-hold benchmark is always computed alongside the strategy
   so performance can be compared fairly.

Pipeline role
-------------
strategy.py  →  signal_df["position"]  →  backtest.py  →  equity curve
                                                        →  trade log
                                                        →  metrics.py
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_INITIAL_CAPITAL: float = 10_000.0   # starting portfolio value ($)
DEFAULT_TRANSACTION_COST: float = 0.001     # 0.10% per trade (one-way)
RESULTS_DIR: Path = Path(__file__).resolve().parents[1] / "results"


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------

@dataclass
class BacktestResult:
    """Container for all outputs produced by a single backtest run.

    Attributes
    ----------
    equity_curve : pd.Series
        Portfolio value in dollars for every trading day in the test period.
    benchmark_curve : pd.Series
        Buy-and-hold portfolio value over the same period.
    daily_returns : pd.Series
        Strategy daily percentage returns (used by metrics.py).
    trade_log : pd.DataFrame
        One row per completed trade with entry/exit dates, prices, and P&L.
    positions : pd.Series
        The position series that was simulated (copy from strategy_df).
    config : dict[str, Any]
        Snapshot of the parameters used for this run.
    """
    equity_curve:     pd.Series # ke metrics.py, untuk menghitung total return, max drawdown, dll
    benchmark_curve:  pd.Series # perbandingan buy & hold
    daily_returns:    pd.Series # Sharpe ratio, volatility, dll di metrics.py
    trade_log:        pd.DataFrame
    positions:        pd.Series
    config:           dict[str, Any] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# Core backtest engine
# ---------------------------------------------------------------------------

def run_backtest(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
    *,
    initial_capital: float = DEFAULT_INITIAL_CAPITAL,
    transaction_cost: float = DEFAULT_TRANSACTION_COST,
) -> BacktestResult:
    """Simulate a trading strategy and return performance data.

    The engine iterates over the test period day by day, applying the
    previous day's position to the current day's return.  This one-day
    lag is the critical detail that prevents lookahead bias: you act on
    yesterday's signal, not today's.

    Concretely, for each day t:

        daily_return[t] = position[t-1] x price_return[t] - cost[t]

    where cost[t] is applied only when the position changes (a trade occurs).
    
    TANPA SHIFT(1):
    Masalahnya: kamu baru **tahu** `position=1` di Senin setelah pasar **tutup** Senin. Tapi `price_return` Senin juga dihitung dari harga **tutup** Senin. Artinya kamu menggunakan informasi yang baru tersedia di akhir hari untuk menghasilkan return di hari yang sama — mustahil dilakukan di dunia nyata.

    Dengan `shift(1)`:
    ```
    tanggal      position    lagged_position    price_return    return_strategi
    ───────────────────────────────────────────────────────────────────────────
    Senin           1              0  (NaN→0)       +2%              0%   ✓
    Selasa          0              1               -1%              -1%   ✓
    Rabu            1              0               +3%               0%   ✓
    Kamis           0              1               +1%              +1%   ✓
    ```

    Parameters
    ----------
    signal_df:
        DataFrame produced by ``strategy.run_strategy``.
        Must contain a ``position`` column with a DatetimeIndex.
    price_df:
        Cleaned OHLCV DataFrame from ``data_loader``.
        Must contain a ``Close`` column covering at least the signal period.
    initial_capital:
        Starting portfolio value in dollars.
    transaction_cost:
        Fraction of portfolio value charged per trade (one-way).
        0.001 = 0.10% per trade, matching a typical low-cost broker.

    Returns
    -------
    BacktestResult
        Full simulation output — see :class:`BacktestResult`.

    Raises
    ------
    ValueError
        If the signal and price DataFrames have no overlapping dates.
    """
    # ---- Align price data to signal period --------------------------------
    log.info(f"Signal df columns: {signal_df.columns.tolist()}")
    log.info(f"Price df columns: {price_df.columns.tolist()}")
    log.info(f"Signal df date range: {signal_df.index[0].date()} → {signal_df.index[-1].date()}")
    prices = _align_prices(signal_df, price_df)

    # ---- Daily price returns (simple, not log) ----------------------------
    # Simple returns used here because they are additive for portfolio math:
    # portfolio_value[t] = portfolio_value[t-1] * (1 + return[t])
    price_returns = prices["Close"].pct_change().fillna(0.0) # perubahan persentase harga dr hari sebelumnya, untuk menghitung return harian, diisi 0 untuk hari pertama karena tidak ada perubahan harga sebelumnya

    # ---- Position series — shift by 1 to enforce the lag -----------------
    # position[t-1] determines exposure during day t.
    # Without this shift, you would be trading on today's close price to generate today's return — a classic lookahead bias.
    positions = signal_df["position"].reindex(prices.index).fillna(0) # reindex memastikan positions punya tanggal yang sama dgn prices, kalau ada tanggal di prices yang gak ada di signal_df, diisi 0 (flat)
    lagged_positions = positions.shift(1).fillna(0)

    # ---- Transaction cost mask -------------------------------------------
    # A trade occurs on any day the position changes value.
    position_changes = positions.diff().abs().fillna(0) > 0
    cost_series = position_changes.astype(float) * transaction_cost

    # ---- Strategy daily returns -------------------------------------------
    # return portofolio pada hari itu, brp persen berubah dari hari sebelumnya, dihitung sebagai posisi kemarin (lagged_positions) 
    # dikali return harga hari ini (price_returns), dikurangi biaya transaksi jika ada perubahan posisi
    strategy_returns = lagged_positions * price_returns - cost_series

    # ---- Equity curves ----------------------------------------------------
    equity_curve = _compute_equity_curve(strategy_returns, initial_capital)
    benchmark_curve = _compute_equity_curve(price_returns, initial_capital)
    
    # Equity curve : NILAI PORTOFOLIO SETIAP HARINYA, dihitung dengan mengalikan return harga harian dengan posisi kemarin (lagged_positions) untuk mendapatkan return strategi harian, lalu dikonversi ke nilai portofolio dengan mengalikan return harian kumulatif dengan initial capital.
    # "portofolio kamu tumbuh seberapa banyak setiap harinya, kalau kamu ikuti strategi ini?"
    # Benchmark curve : NILAI PORTOFOLIO JIKA BUY & HOLD SEJAK AWAL PERIODE, dihitung dengan mengalikan return harga harian dengan posisi 1 (selalu long); 
    # "kalau kamu beli di hari pertama dan pegang terus, berapa nilai portofolio setiap harinya?"

    # ---- Trade log ---------------------------------------------------------
    trade_log = _build_trade_log(positions, prices["Close"], initial_capital)

    result = BacktestResult(
        equity_curve=equity_curve,
        benchmark_curve=benchmark_curve,
        daily_returns=strategy_returns,
        trade_log=trade_log,
        positions=positions,
        config={
            "initial_capital":   initial_capital,
            "transaction_cost":  transaction_cost,
            "start_date":        str(prices.index[0].date()),
            "end_date":          str(prices.index[-1].date()),
            "n_trading_days":    len(prices),
        },
    )

    _log_summary(result)
    return result


# ---------------------------------------------------------------------------
# Equity curve
# ---------------------------------------------------------------------------

def _compute_equity_curve(
    daily_returns: pd.Series,
    initial_capital: float,
) -> pd.Series:
    """Convert a daily returns series into a dollar equity curve."""
    growth_factors = (1 + daily_returns).cumprod()
    return (initial_capital * growth_factors).rename("equity")


# ---------------------------------------------------------------------------
# Trade log builder
# ---------------------------------------------------------------------------

def _build_trade_log(
    positions: pd.Series,
    close_prices: pd.Series,
    initial_capital: float,
) -> pd.DataFrame:
    """Build a trade log with one row per completed round-trip trade.

    A round-trip trade is defined as:
        entry: position transitions from 0 → 1
        exit:  position transitions from 1 → 0

    Each row records the entry/exit dates and prices, holding period,
    return, and approximate P&L in dollars.
    
    Triggered setiap kali posisi berubah dari flat (0) ke long (1) → entry, dan dari long (1) ke flat (0) → exit.
    Return dihitung sebagai (exit_price - entry_price) / entry_price, lalu dikonversi ke persen dengan mengalikan 100. 
    P&L dalam dolar dihitung dengan mengalikan return dengan initial capital, sebagai pendekatan kasar untuk berapa banyak uang yang dihasilkan atau hilang dari trade itu.

    Contoh output::

        entry_date   exit_date    entry_price   exit_price   holding_days   return_pct
        2022-10-31   2022-11-07   150.67        136.73       7              -9.25

    Artinya::

        2022-10-31  position: 0->1   ENTRY, beli di 150.67
        2022-11-01  position: 1      hold
        2022-11-02  position: 1      hold
        ...
        2022-11-07  position: 1->0   EXIT, jual di 136.73
                                     return = (136.73 - 150.67) / 150.67 = -9.25%

    Parameters
    ----------
    positions:
        Position series (+1 / 0) from strategy.py.
    close_prices:
        Close price series aligned to the same DatetimeIndex.
    initial_capital:
        Used to approximate dollar P&L per trade.

    Returns
    -------
    pd.DataFrame
        Columns: entry_date, exit_date, entry_price, exit_price,
                 holding_days, return_pct, pnl_dollars.
    """
    trades: list[dict] = []
    in_trade = False
    entry_date = None
    entry_price = None

    for date, pos in positions.items():
        prev_pos = positions.shift(1).get(date, 0)

        # Entry: flat → long
        if pos == 1 and prev_pos != 1 and not in_trade:
            in_trade = True
            entry_date = date
            entry_price = close_prices.get(date, np.nan)

        # Exit: long → flat
        elif pos != 1 and prev_pos == 1 and in_trade:
            exit_price = close_prices.get(date, np.nan)
            ret = (exit_price - entry_price) / entry_price if entry_price else 0
            holding = (date - entry_date).days  # type: ignore[operator]

            trades.append({
                "entry_date":   entry_date,
                "exit_date":    date,
                "entry_price":  round(entry_price, 4),
                "exit_price":   round(exit_price, 4),
                "holding_days": holding,
                "return_pct":   round(ret * 100, 4),
                "pnl_dollars":  round(ret * initial_capital, 2),
            })
            in_trade = False
            entry_date = None
            entry_price = None

    # Handle open trade at end of period
    if in_trade and entry_date is not None:
        last_date = positions.index[-1]
        exit_price = close_prices.get(last_date, np.nan)
        ret = (exit_price - entry_price) / entry_price if entry_price else 0
        holding = (last_date - entry_date).days

        trades.append({
            "entry_date":   entry_date,
            "exit_date":    last_date,
            "entry_price":  round(entry_price, 4),
            "exit_price":   round(exit_price, 4),
            "holding_days": holding,
            "return_pct":   round(ret * 100, 4),
            "pnl_dollars":  round(ret * initial_capital, 2),
            "note":         "open at end of period",
        })

    if not trades:
        log.warning("No completed trades found in the backtest period.")
        return pd.DataFrame()

    return pd.DataFrame(trades)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(
    result: BacktestResult,
    output_dir: str | Path = RESULTS_DIR,
) -> None:
    """Save equity curves and trade log to the results/ directory.

    Saved files
    -----------
    equity_curve.csv    — daily portfolio and benchmark values
    trade_log.csv       — one row per completed round-trip trade

    Parameters
    ----------
    result:
        BacktestResult produced by ``run_backtest``.
    output_dir:
        Directory to write files into. Created if it does not exist.
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    # Equity curves
    curves = pd.DataFrame({
        "strategy":  result.equity_curve,
        "benchmark": result.benchmark_curve,
    })
    curves.to_csv(out / "equity_curve.csv")
    log.info("Saved equity_curve.csv  (%d rows)", len(curves))

    # Trade log
    if not result.trade_log.empty:
        result.trade_log.to_csv(out / "trade_log.csv", index=False)
        log.info("Saved trade_log.csv  (%d trades)", len(result.trade_log))


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _align_prices(
    signal_df: pd.DataFrame,
    price_df: pd.DataFrame,
) -> pd.DataFrame:
    """Restrict price_df to the date range covered by signal_df."""
    start = signal_df.index[0] # data signal mulai dari tanggal berapa
    end   = signal_df.index[-1] # data signal sampai tanggal berapa (-1 berarti index terakhir)
    aligned = price_df.loc[start:end]

    if aligned.empty:
        raise ValueError(
            f"No overlapping dates between signals ({start.date()} → "
            f"{end.date()}) and price data ({price_df.index[0].date()} → "
            f"{price_df.index[-1].date()})."
        )
    return aligned


def _log_summary(result: BacktestResult) -> None:
    """Log a compact summary of backtest results."""
    final_val  = result.equity_curve.iloc[-1]
    bench_val  = result.benchmark_curve.iloc[-1]
    init_cap   = result.config["initial_capital"]
    n_trades   = len(result.trade_log)

    log.info("─" * 44)
    log.info("  Backtest summary")
    log.info("─" * 44)
    log.info("  Period         %s → %s",
             result.config["start_date"], result.config["end_date"])
    log.info("  Trading days   %d", result.config["n_trading_days"])
    log.info("  Initial cap    $%.2f", init_cap)
    log.info("  Final value    $%.2f  (%+.2f%%)",
             final_val, (final_val / init_cap - 1) * 100)
    log.info("  Buy & hold     $%.2f  (%+.2f%%)",
             bench_val, (bench_val / init_cap - 1) * 100)
    log.info("  Trades         %d", n_trades)
    log.info("─" * 44)


# ---------------------------------------------------------------------------
# Smoke test  (python src/backtest.py)
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
    from model import train_model, time_series_split
    from strategy import run_strategy

    # 1. Data + features
    raw = download_stock_data("AAPL", "2018-01-01", "2024-01-01")
    df_feat = build_features(raw)
    X, y = split_features_target(df_feat)

    # 2. Split + train
    X_train, X_test, y_train, y_test = time_series_split(X, y)
    model, scaler = train_model(X_train, y_train)

    # 3. Strategy signals
    signal_df = run_strategy(model, scaler, X_test)

    # 4. Backtest
    result = run_backtest(signal_df, raw)

    # 5. Inspect outputs
    print("\nEquity curve (last 5 rows):")
    print(pd.DataFrame({
        "strategy":  result.equity_curve,
        "benchmark": result.benchmark_curve,
    }).tail().to_string())

    print(f"\nTrade log ({len(result.trade_log)} trades):")
    print(result.trade_log.to_string(index=False))

    # 6. Basic sanity checks
    assert not result.equity_curve.isna().any(), "NaN in equity curve!"
    assert result.equity_curve.iloc[0] > 0,      "Equity starts at zero!"
    assert len(result.trade_log) > 0,             "No trades recorded!"
    print("\nAll sanity checks passed.")

    # 7. Save
    save_results(result)
    print(f"Results saved to {RESULTS_DIR}")