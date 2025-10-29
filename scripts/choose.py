#!/usr/bin/env python3
"""
choose.py - Minimal on-demand Value strategy selector.

Usage:
    python scripts/choose.py

Behavior:
- Checks US market hours (ET). If closed, warns but still can run.
- Downloads 90 days of S&P500 data via get_clean_sp500_data(days=90).
- Calculates Value signals, gets current signals, returns Top 10 longs and shorts.
- Prints simple recommendations for a $1,000,000 portfolio.
"""
import sys
from pathlib import Path
from datetime import datetime, timedelta

# allow imports from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytz
from data.data import get_clean_sp500_data
from signals.signal_value import ValueSignal


def is_market_open(now: datetime = None):
    tz = pytz.timezone('US/Eastern')
    now = now or datetime.now(tz)
    # Market hours Mon-Fri 9:30-16:00 ET
    if now.weekday() >= 5:
        return False, "Market closed (weekend)"
    open_time = now.replace(hour=9, minute=30, second=0, microsecond=0)
    close_time = now.replace(hour=16, minute=0, second=0, microsecond=0)
    if now < open_time or now >= close_time:
        return False, f"Market closed (hours: 9:30-16:00 ET, now: {now.strftime('%H:%M')})"
    return True, "Market open"


def main():
    print("=" * 60)
    print("CHOOSE.PY — Value strategy quick selector")
    print("=" * 60)
    now = datetime.now(pytz.timezone('US/Eastern'))
    print(f"Timestamp (ET): {now.strftime('%Y-%m-%d %H:%M')}")

    market_open, msg = is_market_open(now)
    print(msg)
    print()

    print("Loading data (90 days)...")
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    data, tickers = get_clean_sp500_data(start_date=start_date, end_date=end_date)
    print(f"Data loaded: {len(tickers)} tickers, date range: {data.index.get_level_values('Date').min().date()} -> {data.index.get_level_values('Date').max().date()}\n")

    print("Calculating Value signals...")
    vsig = ValueSignal()
    signals = vsig.calculate(data)

    current = vsig.get_current_signals(signals)
    if current is None or current.empty:
        print("No signals generated (data too recent?). Exiting.")
        return

    # get top 10
    top = vsig.get_top_signals(current, n=10)
    portfolio_value = 1_000_000

    print("TOP 10 LONGS (Undervalued)")
    print("Ticker  Signal   Weight($) ")
    for ticker, row in top['buy'].iterrows():
        # For simplicity assume equal split across 20 positions (10 longs + 10 shorts)
        dollars = portfolio_value * 0.5 / 10
        print(f"{ticker:<6} {row['signal_zscore']:>+6.2f}   ${dollars:,.0f}")

    print("\nTOP 10 SHORTS (Overvalued)")
    print("Ticker  Signal   Weight($) ")
    for ticker, row in top['sell'].iterrows():
        dollars = portfolio_value * 0.5 / 10
        print(f"{ticker:<6} {row['signal_zscore']:>+6.2f}   ${dollars:,.0f}")

    print("\nDone.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nInterrupted')
    except Exception as e:
        print(f"Error: {e}")
        raise
