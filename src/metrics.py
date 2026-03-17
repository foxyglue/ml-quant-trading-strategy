"""
metrics.py
==========
Computes and reports trading performance metrics from a BacktestResult.

All metrics are derived from two inputs:
    - daily_returns  : strategy return series from backtest.py
    - equity_curve   : cumulative portfolio value from backtest.py

Metrics computed
----------------
cumulative_return   : total % gain/loss over the full period
annualized_return   : cumulative return dikonversi ke setara 1 tahun. Berguna untuk membandingkan strategi yang dijalankan di periode berbeda
sharpe_ratio        : risk-adjusted return (annualised, rf=0); return per unit risiko. Negatif = strategi lebih buruk dari cash; 0-1 = suboptimal; 1-2 = acceptable; >2 = excellent (jarang terjadi)
max_drawdown        : largest peak-to-trough loss (%); kerugian terbesar dari puncak ke lembah
win_rate            : Dari semua hari aktif di pasar, berapa persen yang profitable. Hanya menghitung hari position=1
profit_factor       : total keuntungan dibagi total kerugian. Di atas 1.0 = lebih banyak untung dari rugi.
avg_trade_return    : rata-rata return per trade dari trade_log. Beda dgn win_rate yg per hari, ini per trade (round-trip), dihitung dari harga entry ke harga exit, bukan harian. Berguna untuk memahami seberapa besar rata-rata keuntungan atau kerugian setiap kali kamu melakukan trading, terlepas dari berapa banyak hari yang kamu pegang posisi itu.
benchmark_return    : buy-and-hold cumulative return for comparison; hasil buy-and-hold di periode yang sama -- jika strategi yg digunakan lebih buruk dr benchmark, berarti kamu bisa dapat hasil yang sama atau lebih baik dengan cuma beli di hari pertama dan pegang terus

Pipeline role
-------------
backtest.py  →  BacktestResult  →  metrics.py  →  performance_metrics.json
                                               →  printed report
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger(__name__)

TRADING_DAYS_PER_YEAR: int = 252
RESULTS_DIR: Path = Path(__file__).resolve().parents[1] / "results"


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

def cumulative_return(equity_curve: pd.Series) -> float:
    """Total percentage return from start to end of the equity curve.
    
    Total keuntungan/kerugian dari awal sampai akhir.

    Formula
    -------
        (final_value / initial_value) - 1

    Parameters
    ----------
    equity_curve:
        Portfolio value series produced by ``backtest.run_backtest``.

    Returns
    -------
    float
        Cumulative return as a decimal, e.g. 0.25 = +25%.
    """
    return float(equity_curve.iloc[-1] / equity_curve.iloc[0] - 1)


def annualized_return(equity_curve: pd.Series) -> float:
    """Cumulative return scaled to a one-year equivalent.

    Useful for comparing strategies run over different time horizons.
    
    Cumulative return dikonversi ke setara 1 tahun. Berguna untuk membandingkan strategi yang dijalankan di periode berbeda.

    Formula
    -------
        (1 + cumulative_return) ^ (252 / n_days) - 1
        
    252 = trading days per year; n_days = jumlah hari dalam equity curve
    
    - n_days < 252, artinya strategi dijalankan kurang dari 1 tahun, maka cumulative return akan di-boost untuk mencerminkan potensi return tahunan. 
    - n_days > 252, artinya strategi dijalankan lebih dari 1 tahun, maka cumulative return akan di-damp untuk mencerminkan return tahunan yang lebih konservatif.

    Parameters
    ----------
    equity_curve:
        Portfolio value series.

    Returns
    -------
    float
        Annualised return as a decimal.
    """
    n_days = len(equity_curve)
    cum_ret = cumulative_return(equity_curve)
    return float((1 + cum_ret) ** (TRADING_DAYS_PER_YEAR / n_days) - 1)


def sharpe_ratio(daily_returns: pd.Series, risk_free_rate: float = 0.0) -> float:
    """Annualised Sharpe ratio using daily returns.

    Return per unit risiko — seberapa banyak return yang didapatkan
    untuk setiap unit risiko yang diambil.

    The Sharpe ratio measures return per unit of risk. A higher value is
    better. Common benchmarks:
    
        < 1.0   : suboptimal
        1.0-2.0 : acceptable
        > 2.0   : excellent (rare in practice)

    Formula (from Investopedia)
    ---------------------------
    The canonical formula uses annualised figures:

        Sharpe = (Rp - Rf) / σp

    where Rp = annualised portfolio return,
          Rf = annualised risk-free rate,
          σp = annualised portfolio volatility.

    Because this module works with ***daily*** returns, we convert to annual:

        mean_annual = mean_daily x 252
        std_annual  = std_daily  x √252

    Substituting:

        Sharpe = (mean_daily x 252) / (std_daily x √252)
               = (mean_daily / std_daily) x (252 / √252)
               = (mean_daily / std_daily) x √252

    The √252 factor is therefore not an arbitrary scaling constant — it is
    the algebraic result of annualising both the numerator and denominator
    simultaneously from daily data.

    risk_free_rate defaults to 0 because most retail strategies compare
    against cash. Pass ``risk_free_rate=0.05/252`` for a 5% annual
    risk-free benchmark.

    Parameters
    ----------
    daily_returns:
        Strategy daily return series from ``backtest.run_backtest``.
    risk_free_rate:
        Daily risk-free rate. Default 0.

    Returns
    -------
    float
        Annualised Sharpe ratio. Returns 0.0 if std is zero.
    """
    excess = daily_returns - risk_free_rate
    std = excess.std()
    if std == 0:
        log.warning("Sharpe ratio undefined: daily return std is zero.")
        return 0.0
    return float(excess.mean() / std * np.sqrt(TRADING_DAYS_PER_YEAR))


def max_drawdown(equity_curve: pd.Series) -> float:
    """Maximum peak-to-trough decline over the full equity curve.
    
    Kerugian terbesar dari puncak ke lembah selama periode backtest.

    This is the worst-case loss an investor would have experienced if they
    bought at the worst possible moment (a peak) and sold at the subsequent
    trough. It is the primary measure of downside risk.

    Formula
    -------
        max(1 - equity_curve / running_peak)

    Parameters
    ----------
    equity_curve:
        Portfolio value series.

    Returns
    -------
    float
        Maximum drawdown as a negative decimal, e.g. -0.15 = -15%.
    """
    running_peak = equity_curve.cummax() # cummax merupakan kumulatif maksimum dari equity curve, menghasilkan seri yang menunjukkan nilai tertinggi yang pernah dicapai portofolio. (menyimpan puncak tertinggi yang pernah dicapai)
    drawdown = equity_curve / running_peak - 1
    return float(drawdown.min())


def win_rate(daily_returns: pd.Series, positions: pd.Series) -> float:
    """Fraction of active trading days (position=1) that were profitable.
    
    Dari semua hari aktif di pasar, berapa persen yang profitable. Hanya menghitung hari position=1 (strategy sedang long).

    Only days where the strategy was actually in the market are counted.
    Days in cash are excluded because a 0% return on a cash day is not
    a meaningful "win".

    Parameters
    ----------
    daily_returns:
        Strategy daily return series.
    positions:
        Position series (+1 / 0) from strategy.py.

    Returns
    -------
    float
        Win rate as a decimal, e.g. 0.54 = 54% of active days were up.
    """
    active_returns = daily_returns[positions == 1]
    if len(active_returns) == 0:
        log.warning("Win rate undefined: no active trading days found.")
        return 0.0
    return float((active_returns > 0).sum() / len(active_returns))


def profit_factor(daily_returns: pd.Series) -> float:
    """Ratio of gross profit to gross loss across all trading days.
    
    Total keuntungan dibagi total kerugian. Di atas 1.0 = lebih banyak untung dari rugi.

    A profit factor > 1.0 means the strategy made more money than it lost.
    Values above 1.5 are generally considered strong.

    Formula
    -------
        sum(positive_returns) / abs(sum(negative_returns))

    Parameters
    ----------
    daily_returns:
        Strategy daily return series.

    Returns
    -------
    float
        Profit factor. Returns inf if there are no losing days.
        Returns 0.0 if there are no winning days.
    """
    gains  = daily_returns[daily_returns > 0].sum()
    losses = daily_returns[daily_returns < 0].sum()

    if losses == 0:
        return float("inf")
    if gains == 0:
        return 0.0
    return float(gains / abs(losses))


def avg_trade_return(trade_log: pd.DataFrame) -> float:
    """Mean return per completed round-trip trade.
    
    Rata-rata return per trade dari trade_log. Beda dgn win_rate yg per hari, ini per trade (round-trip), dihitung dari harga entry ke harga exit, bukan harian. 
    Berguna untuk memahami seberapa besar rata-rata keuntungan atau kerugian setiap kali melakukan trading, terlepas dari berapa banyak hari pegang posisi itu.

    Parameters
    ----------
    trade_log:
        DataFrame produced by ``backtest._build_trade_log``.
        Must contain a ``return_pct`` column.

    Returns
    -------
    float
        Average trade return in percent, e.g. 0.85 = +0.85% per trade.
        Returns 0.0 if the trade log is empty.
    """
    if trade_log.empty or "return_pct" not in trade_log.columns:
        return 0.0
    # return_pct disimpan dalam persen (e.g. -9.25), 
    # dibagi 100 agar konsisten dengan metrik lain yang disimpan sebagai desimal
    return float(trade_log["return_pct"].mean() / 100)


# ---------------------------------------------------------------------------
# Master function
# ---------------------------------------------------------------------------

def compute_all_metrics(
    result: object,  # BacktestResult — avoid circular import
) -> dict[str, float]:
    """Compute all trading metrics from a BacktestResult object.

    This is the primary function called by ``main.py``. It pulls
    ``equity_curve``, ``daily_returns``, ``positions``, ``trade_log``,
    and ``benchmark_curve`` directly from the BacktestResult dataclass.

    Parameters
    ----------
    result:
        ``BacktestResult`` instance produced by ``backtest.run_backtest``.

    Returns
    -------
    dict[str, float]
        All metric names mapped to their values, rounded to 4 decimal places.
    """
    eq   = result.equity_curve
    dr   = result.daily_returns
    pos  = result.positions
    tlog = result.trade_log
    bm   = result.benchmark_curve
    
    print("DEBUG avg_trade_return raw:", avg_trade_return(tlog))

    metrics = {
        # Strategy metrics
        "cumulative_return":   round(cumulative_return(eq),        4),
        "annualized_return":   round(annualized_return(eq),        4),
        "sharpe_ratio":        round(sharpe_ratio(dr),             4),
        "max_drawdown":        round(max_drawdown(eq),             4),
        "win_rate":            round(win_rate(dr, pos),            4),
        "profit_factor":       round(profit_factor(dr),            4),
        "avg_trade_return_pct":round(avg_trade_return(tlog),       4),

        # Benchmark comparison
        "benchmark_return":    round(cumulative_return(bm),        4),

        # Context
        "n_trades":            float(len(tlog)),
        "n_trading_days":      float(len(eq)),
    }

    _log_report(metrics)
    return metrics


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_metrics(
    metrics: dict[str, float],
    path: str | Path | None = None,
) -> None:
    """Save metrics dictionary to a JSON file.

    Parameters
    ----------
    metrics:
        Dictionary produced by ``compute_all_metrics``.
    path:
        Destination path. Defaults to ``results/performance_metrics.json``.
    """
    if path is None:
        path = RESULTS_DIR / "performance_metrics.json"

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    with open(path, "w") as f:
        json.dump(metrics, f, indent=4)

    log.info("Metrics saved to %s", path)


def load_metrics(path: str | Path) -> dict[str, float]:
    """Load a metrics JSON file previously saved by ``save_metrics``.

    Parameters
    ----------
    path:
        Path to a JSON file produced by ``save_metrics``.

    Returns
    -------
    dict[str, float]

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics file not found: {path}")

    with open(path) as f:
        return json.load(f)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _log_report(metrics: dict[str, float]) -> None:
    """Pretty-print the full metrics report to the logger."""
    border = "═" * 44
    log.info(border)
    log.info("  PERFORMANCE REPORT")
    log.info(border)

    sections = {
        "Returns": ["cumulative_return", "annualized_return", "benchmark_return"],
        "Risk":    ["sharpe_ratio", "max_drawdown"],
        "Trading": ["win_rate", "profit_factor", "avg_trade_return_pct"],
        "Context": ["n_trades", "n_trading_days"],
    }

    for section, keys in sections.items():
        log.info("  %s", section)
        for k in keys:
            v = metrics[k]
            # Format percentages vs plain numbers
            if any(x in k for x in ["return", "drawdown", "win_rate"]):
                log.info("    %-26s  %+.2f%%", k, v * 100)
            elif k in ("n_trades", "n_trading_days"):
                log.info("    %-26s  %d", k, int(v))
            else:
                log.info("    %-26s  %.4f", k, v)

    # Alpha: strategy vs benchmark
    alpha = metrics["cumulative_return"] - metrics["benchmark_return"]
    log.info("  ─" * 22)
    log.info("    %-26s  %+.2f%%", "alpha vs benchmark", alpha * 100)
    log.info(border)


# ---------------------------------------------------------------------------
# Smoke test  (python src/metrics.py)
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
    from backtest import run_backtest

    # Full pipeline
    raw      = download_stock_data("AAPL", "2018-01-01", "2024-01-01")
    df_feat  = build_features(raw)
    X, y     = split_features_target(df_feat)
    X_train, X_test, y_train, y_test = time_series_split(X, y)
    model, scaler = train_model(X_train, y_train)
    signal_df     = run_strategy(model, scaler, X_test)
    result        = run_backtest(signal_df, raw)

    # Metrics
    metrics = compute_all_metrics(result)

    # Save
    save_metrics(metrics)

    # print(f"Average Trade Return: {metrics['avg_trade_return_pct']:.4f}")
    # print(f"Sample Trade Returns:\n{result.trade_log['return_pct'].tolist()}")

    # Sanity checks
    assert -1.0 <= metrics["max_drawdown"] <= 0.0, "Max drawdown out of range!"
    assert  0.0 <= metrics["win_rate"]     <= 1.0, "Win rate out of range!"
    assert metrics["n_trades"] > 0,                "No trades recorded!"
    print("\nAll sanity checks passed.")
    print(f"\nFull metrics:\n{json.dumps(metrics, indent=4)}")