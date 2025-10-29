#!/usr/bin/env python3
"""
Main script to run the mean reversion stock picking strategy.

Usage:
    python run_strategy.py [--backtest] [--days DAYS] [--portfolio-value VALUE]
"""
import argparse
from datetime import datetime, timedelta
import sys
import pandas as pd

from data import get_clean_sp500_data
from signal_meanrev import MeanReversionSignal, calculate_signal_quality
from portfolio import Portfolio
from backtest import Backtest


def run_live_strategy(portfolio_value: float = 1_000_000, days: int = 730,
                     top_n_long: int = None, top_n_short: int = None):
    """
    Run strategy for current date and output recommendations.
    
    Args:
        portfolio_value: Portfolio value in dollars
        days: Number of days of historical data to fetch
        top_n_long: Only buy top N long positions (None = all)
        top_n_short: Only short top N short positions (None = all)
    """
    print("=" * 70)
    print("MEAN REVERSION STOCK PICKING STRATEGY")
    print("=" * 70)
    print(f"Date: {datetime.now().strftime('%Y-%m-%d')}")
    print(f"Portfolio Value: ${portfolio_value:,.0f}")
    if top_n_long or top_n_short:
        print(f"Mode: Top {top_n_long or 0} Longs / Top {top_n_short or 0} Shorts")
    print()
    
    # Get data
    start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
    print(f"Fetching data from {start_date}...\n")
    
    data, tickers = get_clean_sp500_data(start_date=start_date)
    
    # Calculate signals
    print("\nCalculating mean reversion signals...")
    signal_calc = MeanReversionSignal()
    signals = signal_calc.calculate(data)
    
    # Get current signals
    current_signals = signal_calc.get_current_signals(signals)
    print(f"Generated signals for {len(current_signals)} stocks")
    
    # Calculate signal quality
    print("\nEvaluating signal quality...")
    quality = calculate_signal_quality(signals)
    print(f"Information Coefficient: {quality['information_coefficient']:.4f}")
    print(f"Historical Win Rate: {quality['win_rate']:.1%}")
    
    # Build portfolio
    print("\nConstructing portfolio...")
    portfolio_mgr = Portfolio(top_n_long=top_n_long, top_n_short=top_n_short)
    positions = portfolio_mgr.calculate_positions(current_signals)
    
    # Display recommendations
    print("\n")
    recommendations = portfolio_mgr.format_recommendations(positions, portfolio_value)
    print(recommendations)
    
    # Top picks
    print("\n" + "=" * 70)
    print("TOP 10 PICKS")
    print("=" * 70)
    
    top_signals = signal_calc.get_top_signals(current_signals, n=10)
    
    print("\nStrongest BUY Signals (Most Oversold):")
    print("-" * 70)
    for ticker, row in top_signals['buy'].iterrows():
        weight = positions.loc[ticker, 'weight'] if ticker in positions.index else 0
        dollars = weight * portfolio_value
        print(f"{ticker:<8} Z-Score: {row['signal_zscore']:>6.2f}  "
              f"Weight: {weight*100:>5.2f}%  Value: ${dollars:>10,.0f}")
    
    print("\nStrongest SELL Signals (Most Overbought):")
    print("-" * 70)
    for ticker, row in top_signals['sell'].iterrows():
        weight = positions.loc[ticker, 'weight'] if ticker in positions.index else 0
        dollars = weight * portfolio_value
        print(f"{ticker:<8} Z-Score: {row['signal_zscore']:>6.2f}  "
              f"Weight: {weight*100:>5.2f}%  Value: ${dollars:>10,.0f}")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("1. Review the recommendations above")
    print("2. Execute trades in your brokerage account")
    print("3. Run this script again next Monday for updated signals")
    print("=" * 70)


def run_backtest_mode(days: int = 3650, capital: float = 1_000_000,
                      start_date: str = None, end_date: str = None,
                      top_n_long: int = None, top_n_short: int = None):
    """
    Run historical backtest.
    
    Args:
        days: Number of days to backtest (default: 10 years)
        capital: Starting capital (default: $1M)
        start_date: Specific start date for BACKTEST (overrides days)
        end_date: Specific end date (default: today)
        top_n_long: Only buy top N long positions (None = all)
        top_n_short: Only short top N short positions (None = all)
    """
    print("=" * 70)
    print("BACKTEST MODE")
    print("=" * 70)
    
    # For backtesting, we need data BEFORE the backtest period for signal calculation
    # We need at least 252 trading days (1 year) of history before backtest start
    
    if start_date is None:
        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')
        data_start = start_date  # Use same start for data
    else:
        # Start loading data 400 days before backtest start to have enough history
        backtest_start = datetime.strptime(start_date, '%Y-%m-%d')
        data_start = (backtest_start - timedelta(days=400)).strftime('%Y-%m-%d')
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Backtest: {start_date} to {end_date}")
    if top_n_long or top_n_short:
        print(f"Strategy: Top {top_n_long or 0} Longs / Top {top_n_short or 0} Shorts")
    print(f"Capital: ${capital:,.0f}\n")
    
    # Load data (no downloading!)
    print("Loading data...")
    data, tickers = get_clean_sp500_data(start_date=data_start, end_date=end_date)
    
    # Calculate signals
    signal_calc = MeanReversionSignal()
    signals = signal_calc.calculate(data)
    
    # Filter data and signals to backtest period only
    backtest_start_dt = pd.Timestamp(start_date)
    data = data[data.index.get_level_values('Date') >= backtest_start_dt]
    signals = signals[signals.index.get_level_values('Date') >= backtest_start_dt]
    
    print(f"Trading days: {len(data.index.get_level_values('Date').unique())}")
    
    # Evaluate signal quality
    quality = calculate_signal_quality(signals)
    print(f"Signal IC: {quality['information_coefficient']:.3f}, Win Rate: {quality['win_rate']:.1%}\n")
    
    # Run backtest
    portfolio_mgr = Portfolio(top_n_long=top_n_long, top_n_short=top_n_short)
    backtest = Backtest(initial_capital=capital)
    results = backtest.run(data, signals, portfolio_mgr, verbose=True)
    
    # Calculate metrics
    metrics = backtest.calculate_metrics(results)
    
    # Print summary
    backtest.print_summary(metrics)
    
    # Plot results
    print("\nGenerating performance plots...")
    try:
        backtest.plot_performance(results, save_path='backtest_results.png')
    except Exception as e:
        print(f"Could not generate plot: {e}")
        print("(This is normal if running in a headless environment)")
    
    # Save results
    results.to_csv('backtest_results.csv')
    print(f"\nBacktest results saved to backtest_results.csv")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Mean Reversion Stock Picking Strategy'
    )
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run historical backtest instead of live recommendations'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=730,
        help='Number of days of historical data to fetch (default: 730)'
    )
    parser.add_argument(
        '--portfolio-value',
        type=float,
        default=1_000_000,
        help='Portfolio value in dollars (default: 1,000,000)'
    )
    parser.add_argument(
        '--start-date',
        type=str,
        default=None,
        help='Start date for backtest in YYYY-MM-DD format (overrides --days)'
    )
    parser.add_argument(
        '--end-date',
        type=str,
        default=None,
        help='End date for backtest in YYYY-MM-DD format (default: today)'
    )
    parser.add_argument(
        '--top-n-long',
        type=int,
        default=None,
        help='Only buy top N long positions (default: all)'
    )
    parser.add_argument(
        '--top-n-short',
        type=int,
        default=None,
        help='Only short top N short positions (default: all)'
    )
    parser.add_argument(
        '--random-month',
        action='store_true',
        help='Pick a random month from available data for backtesting'
    )
    
    args = parser.parse_args()
    
    # Handle random month selection
    if args.random_month and args.backtest:
        import random
        from data import get_clean_sp500_data
        
        # Load all available data to find date range
        print("Loading data to find available date range...")
        all_data, _ = get_clean_sp500_data(start_date='2015-01-01')
        dates = all_data.index.get_level_values('Date').unique().sort_values()
        
        # Find all possible month start dates (need 400 days before for signals + 30 days for backtest)
        earliest_backtest_start = dates[0] + pd.Timedelta(days=400)
        latest_backtest_start = dates[-1] - pd.Timedelta(days=30)
        
        # Generate all month starts in the valid range
        possible_months = pd.date_range(
            start=earliest_backtest_start, 
            end=latest_backtest_start, 
            freq='MS'  # Month Start
        )
        
        # Pick a random month
        random_start = random.choice(possible_months)
        random_end = random_start + pd.DateOffset(months=1)
        
        args.start_date = random_start.strftime('%Y-%m-%d')
        args.end_date = random_end.strftime('%Y-%m-%d')
        
        print(f"Randomly selected month: {random_start.strftime('%B %Y')}\n")
    
    try:
        if args.backtest:
            run_backtest_mode(
                days=args.days,
                capital=args.portfolio_value,
                start_date=args.start_date,
                end_date=args.end_date,
                top_n_long=args.top_n_long,
                top_n_short=args.top_n_short
            )
        else:
            run_live_strategy(
                portfolio_value=args.portfolio_value,
                days=args.days,
                top_n_long=args.top_n_long,
                top_n_short=args.top_n_short
            )
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(0)
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
