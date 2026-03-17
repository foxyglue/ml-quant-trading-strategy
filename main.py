"""
main.py
=======
Entry point for the ML Quant Trading Strategy pipeline.

Orchestrates all modules end-to-end:
    data_loader → features → model → strategy → backtest → metrics

Usage
-----
    python main.py
    python main.py --ticker MSFT --start 2019-01-01 --end 2024-01-01
    python main.py --ticker AAPL --buy-threshold 0.58 --no-xgboost

All outputs are saved to results/:
    results/equity_curve.csv
    results/trade_log.csv
    results/performance_metrics.json
    results/equity_curve.png
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd

# ---------------------------------------------------------------------------
# Path setup — allows running from project root: python main.py
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
SRC  = ROOT / "src"
sys.path.insert(0, str(SRC))

from data_loader import download_stock_data
from features    import build_features, split_features_target
from model       import train_model, time_series_split, evaluate_model
from strategy    import run_strategy, summarise_signals
from backtest    import run_backtest, save_results
from metrics     import compute_all_metrics, save_metrics

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

RESULTS_DIR = ROOT / "results"


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="ML Quant Trading Strategy — end-to-end pipeline"
    )
    parser.add_argument("--ticker",        default="AAPL",       help="Yahoo Finance ticker symbol (default: AAPL)")
    parser.add_argument("--start",         default="2018-01-01", help="Start date YYYY-MM-DD (default: 2018-01-01)")
    parser.add_argument("--end",           default="2024-01-01", help="End date YYYY-MM-DD (default: 2024-01-01)")
    parser.add_argument("--train-ratio",   default=0.80, type=float, help="Train/test split ratio (default: 0.80)")
    parser.add_argument("--buy-threshold", default=0.55, type=float, help="P(UP) threshold for BUY signal (default: 0.55)")
    parser.add_argument("--initial-capital", default=10_000.0, type=float, help="Starting capital in USD (default: 10000)")
    parser.add_argument("--no-xgboost",   action="store_true",   help="Force RandomForest even if XGBoost is installed")
    parser.add_argument("--force-download", action="store_true", help="Re-download data even if cache exists")
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Visualisation
# ---------------------------------------------------------------------------

def plot_equity_curve(
    result: object,
    ticker: str,
    save_path: Path,
) -> None:
    """Plot strategy equity curve vs buy-and-hold benchmark."""
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={"height_ratios": [3, 1]})
    fig.suptitle(f"ML Trading Strategy — {ticker}", fontsize=14, fontweight="bold")

    # --- Top panel: equity curves -----------------------------------------
    ax1 = axes[0]
    ax1.plot(result.equity_curve.index,   result.equity_curve.values,   label="ML Strategy",    color="#2196F3", linewidth=2)
    ax1.plot(result.benchmark_curve.index, result.benchmark_curve.values, label="Buy & Hold",   color="#FF9800", linewidth=1.5, linestyle="--")
    ax1.set_ylabel("Portfolio Value ($)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax1.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax1.xaxis.get_majorticklabels(), rotation=45, ha="right")

    # Add initial capital reference line
    init_cap = result.config["initial_capital"]
    ax1.axhline(y=init_cap, color="gray", linestyle=":", linewidth=1, alpha=0.7, label=f"Initial (${init_cap:,.0f})")

    # --- Bottom panel: drawdown -------------------------------------------
    ax2 = axes[1]
    running_peak = result.equity_curve.cummax()
    drawdown = (result.equity_curve / running_peak - 1) * 100
    ax2.fill_between(drawdown.index, drawdown.values, 0, color="#F44336", alpha=0.4, label="Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Date")
    ax2.legend(loc="lower left")
    ax2.grid(True, alpha=0.3)
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y-%m"))
    ax2.xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    plt.setp(ax2.xaxis.get_majorticklabels(), rotation=45, ha="right")

    plt.tight_layout()
    save_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    log.info("Equity curve plot saved to %s", save_path)


def print_summary(metrics: dict, ticker: str, model_metrics: dict) -> None:
    """Print a final human-readable summary to stdout."""
    sep = "=" * 52
    print(f"\n{sep}")
    print(f"  FINAL RESULTS — {ticker}")
    print(sep)
    print(f"  {'Model ROC-AUC':<28}  {model_metrics['roc_auc']:.4f}")
    print(f"  {'Model Accuracy':<28}  {model_metrics['accuracy']:.4f}")
    print(sep)
    print(f"  {'Cumulative Return':<28}  {metrics['cumulative_return']*100:+.2f}%")
    print(f"  {'Annualized Return':<28}  {metrics['annualized_return']*100:+.2f}%")
    print(f"  {'Benchmark (Buy & Hold)':<28}  {metrics['benchmark_return']*100:+.2f}%")
    print(f"  {'Alpha vs Benchmark':<28}  {(metrics['cumulative_return']-metrics['benchmark_return'])*100:+.2f}%")
    print(sep)
    print(f"  {'Sharpe Ratio':<28}  {metrics['sharpe_ratio']:.4f}")
    print(f"  {'Max Drawdown':<28}  {metrics['max_drawdown']*100:+.2f}%")
    print(f"  {'Win Rate':<28}  {metrics['win_rate']*100:.2f}%")
    print(f"  {'Profit Factor':<28}  {metrics['profit_factor']:.4f}")
    print(f"  {'Avg Trade Return':<28}  {metrics['avg_trade_return_pct']*100:+.4f}%")
    print(sep)
    print(f"  {'Total Trades':<28}  {int(metrics['n_trades'])}")
    print(f"  {'Trading Days':<28}  {int(metrics['n_trading_days'])}")
    print(sep)
    print(f"\n  Results saved to: results/")
    print(f"  - performance_metrics.json")
    print(f"  - equity_curve.csv")
    print(f"  - trade_log.csv")
    print(f"  - equity_curve.png\n")


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    log.info("=" * 52)
    log.info("  ML Quant Trading Strategy")
    log.info("  Ticker: %s  |  %s → %s", args.ticker, args.start, args.end)
    log.info("=" * 52)

    # ------------------------------------------------------------------
    # Step 1: Download data
    # ------------------------------------------------------------------
    log.info("[1/6] Downloading data...")
    raw = download_stock_data(
        ticker=args.ticker,
        start_date=args.start,
        end_date=args.end,
        force_download=args.force_download,
    )
    log.info("Data shape: %s", raw.shape)

    # ------------------------------------------------------------------
    # Step 2: Feature engineering
    # ------------------------------------------------------------------
    log.info("[2/6] Building features...")
    df_featured = build_features(raw)
    X, y = split_features_target(df_featured)
    log.info("Feature matrix: %d rows × %d features", *X.shape)

    # ------------------------------------------------------------------
    # Step 3: Train / test split + model training
    # ------------------------------------------------------------------
    log.info("[3/6] Training model (train_ratio=%.0f%%)...", args.train_ratio * 100)
    X_train, X_test, y_train, y_test = time_series_split(X, y, args.train_ratio)

    use_xgb = None if not args.no_xgboost else False
    model, scaler = train_model(X_train, y_train, use_xgboost=use_xgb)

    # Evaluate model classification performance
    model_metrics = evaluate_model(model, scaler, X_test, y_test)

    # ------------------------------------------------------------------
    # Step 4: Generate trading signals
    # ------------------------------------------------------------------
    log.info("[4/6] Generating signals (buy_threshold=%.2f)...", args.buy_threshold)
    signal_df = run_strategy(
        model, scaler, X_test,
        buy_threshold=args.buy_threshold,
    )
    summarise_signals(signal_df)

    # ------------------------------------------------------------------
    # Step 5: Backtest
    # ------------------------------------------------------------------
    log.info("[5/6] Running backtest (initial_capital=$%.0f)...", args.initial_capital)
    result = run_backtest(
        signal_df,
        raw,
        initial_capital=args.initial_capital,
    )
    save_results(result, RESULTS_DIR)

    # ------------------------------------------------------------------
    # Step 6: Metrics + visualisation
    # ------------------------------------------------------------------
    log.info("[6/6] Computing metrics and generating plots...")
    metrics = compute_all_metrics(result)
    save_metrics(metrics, RESULTS_DIR / "performance_metrics.json")

    plot_equity_curve(result, args.ticker, RESULTS_DIR / "equity_curve.png")

    # ------------------------------------------------------------------
    # Final summary
    # ------------------------------------------------------------------
    print_summary(metrics, args.ticker, model_metrics)


if __name__ == "__main__":
    main()