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

from data.data import get_clean_sp500_data
from signals.signal_meanrev import MeanReversionSignal, calculate_signal_quality
from signals.signal_momentum import MomentumSignal
from signals.signal_sector_relative import SectorRelativeSignal
from signals.signal_value import ValueSignal
from signals.signal_combined import CombinedSignal
from strategies.portfolio import Portfolio
from strategies.backtest import Backtest


def run_live_strategy(portfolio_value: float = 1_000_000, days: int = 730,
                     top_n_long: int = None, top_n_short: int = None,
                     strategy: str = 'combined', meanrev_weight: float = 0.5,
                     momentum_weight: float = 0.5, sector_weight: float = 0.0,
                     value_weight: float = 0.0):
    """
    Run strategy for current date and output recommendations.
    
    Args:
        portfolio_value: Portfolio value in dollars
        days: Number of days of historical data to fetch
        top_n_long: Only buy top N long positions (None = all)
        top_n_short: Only short top N short positions (None = all)
        strategy: Signal strategy to use ('meanrev', 'momentum', or 'combined')
        meanrev_weight: Weight for mean reversion in combined strategy (default: 0.5)
        momentum_weight: Weight for momentum in combined strategy (default: 0.5)
        sector_weight: Weight for sector relative in combined strategy (default: 0.0)
        value_weight: Weight for value in combined strategy (default: 0.0)
    """
    print("=" * 70)
    if strategy == 'combined':
        print("COMBINED SIGNAL STOCK PICKING STRATEGY")
        print(f"Mean Reversion: {meanrev_weight:.0%}, Momentum: {momentum_weight:.0%}")
    elif strategy == 'momentum':
        print("MOMENTUM STOCK PICKING STRATEGY")
    elif strategy == 'sector':
        print("SECTOR RELATIVE STOCK PICKING STRATEGY")
    elif strategy == 'value':
        print("VALUE STOCK PICKING STRATEGY")
    else:
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
    
    # Calculate signals based on strategy
    if strategy == 'combined':
        signal_calc = CombinedSignal(
            meanrev_weight=meanrev_weight,
            momentum_weight=momentum_weight,
            sector_weight=sector_weight,
            value_weight=value_weight
        )
        signals, meanrev_signals, momentum_signals, sector_signals, value_signals = signal_calc.calculate(data)
        
        # Evaluate signal quality (skip if insufficient data)
        try:
            quality_results = signal_calc.evaluate_signals(signals, meanrev_signals, momentum_signals, 
                                                           sector_signals, value_signals)
            quality = quality_results['combined']
        except ValueError:
            print("\nSkipping quality evaluation (insufficient data for forward returns)")
            quality = {'information_coefficient': 0.0, 'win_rate': 0.5}
        
    elif strategy == 'momentum':
        print("\nCalculating momentum signals...")
        signal_calc = MomentumSignal()
        signals = signal_calc.calculate(data)
        
        # Calculate signal quality
        from signals.signal_momentum import calculate_signal_quality as calc_quality_mom
        print("\nEvaluating signal quality...")
        quality = calc_quality_mom(signals)
        
    elif strategy == 'sector':
        print("\nCalculating sector relative signals...")
        signal_calc = SectorRelativeSignal()
        signals = signal_calc.calculate(data)
        
        # Calculate signal quality
        from signals.signal_sector_relative import calculate_signal_quality as calc_quality_sector
        print("\nEvaluating signal quality...")
        quality = calc_quality_sector(signals)
        
    elif strategy == 'value':
        print("\nCalculating value signals...")
        signal_calc = ValueSignal()
        signals = signal_calc.calculate(data)
        
        # Calculate signal quality
        from signals.signal_value import calculate_signal_quality as calc_quality_value
        print("\nEvaluating signal quality...")
        quality = calc_quality_value(signals)
        
    else:  # meanrev
        print("\nCalculating mean reversion signals...")
        signal_calc = MeanReversionSignal()
        signals = signal_calc.calculate(data)
        
        # Calculate signal quality
        print("\nEvaluating signal quality...")
        quality = calculate_signal_quality(signals)
    
    # Get current signals
    current_signals = signal_calc.get_current_signals(signals)
    print(f"Generated signals for {len(current_signals)} stocks")
    
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
    
    print("\nStrongest BUY Signals:")
    print("-" * 70)
    for ticker, row in top_signals['buy'].iterrows():
        weight = positions.loc[ticker, 'weight'] if ticker in positions.index else 0
        dollars = weight * portfolio_value
        zscore_val = row['signal_zscore']
        print(f"{ticker:<8} Z-Score: {zscore_val:>6.2f}  "
              f"Weight: {weight*100:>5.2f}%  Value: ${dollars:>10,.0f}")
    
    print("\nStrongest SELL Signals:")
    print("-" * 70)
    for ticker, row in top_signals['sell'].iterrows():
        weight = positions.loc[ticker, 'weight'] if ticker in positions.index else 0
        dollars = weight * portfolio_value
        zscore_val = row['signal_zscore']
        print(f"{ticker:<8} Z-Score: {zscore_val:>6.2f}  "
              f"Weight: {weight*100:>5.2f}%  Value: ${dollars:>10,.0f}")
    
    print("\n" + "=" * 70)
    print("Next Steps:")
    print("1. Review the recommendations above")
    print("2. Execute trades in your brokerage account")
    print("3. Run this script again next Monday for updated signals")
    print("=" * 70)


def run_backtest_mode(days: int = 30, capital: float = 1_000_000,
                      start_date: str = None, end_date: str = None,
                      top_n_long: int = None, top_n_short: int = None,
                      strategy: str = 'combined', meanrev_weight: float = 0.5,
                      momentum_weight: float = 0.5, sector_weight: float = 0.0,
                      value_weight: float = 0.0):
    """
    Run historical backtest.
    
    Args:
        days: Number of days to backtest (default: 1 month)
        capital: Starting capital (default: $1M)
        start_date: Specific start date for BACKTEST (overrides days)
        end_date: Specific end date (default: today)
        top_n_long: Only buy top N long positions (None = all)
        top_n_short: Only short top N short positions (None = all)
        strategy: Signal strategy to use ('meanrev', 'momentum', or 'combined')
        meanrev_weight: Weight for mean reversion in combined strategy (default: 0.5)
        momentum_weight: Weight for momentum in combined strategy (default: 0.5)
    """
    print("=" * 70)
    print("BACKTEST MODE")
    if strategy == 'combined':
        print(f"Strategy: Combined (Mean Rev: {meanrev_weight:.0%}, Momentum: {momentum_weight:.0%})")
    elif strategy == 'sector':
        print(f"Strategy: Sector Relative")
    elif strategy == 'value':
        print(f"Strategy: Value")
    else:
        print(f"Strategy: {strategy.title()}")
    print("=" * 70)
    
    # For backtesting, we need data BEFORE the backtest period for signal calculation
    # We need at least 252 trading days (1 year) of history before backtest start
    
    if start_date is None:
        # Backtest period starts 'days' ago
        backtest_start = datetime.now() - timedelta(days=days)
        start_date = backtest_start.strftime('%Y-%m-%d')
        # Load data starting 400 days before backtest start to have enough history for signals
        data_start = (backtest_start - timedelta(days=400)).strftime('%Y-%m-%d')
    else:
        # Start loading data 400 days before backtest start to have enough history
        backtest_start = datetime.strptime(start_date, '%Y-%m-%d')
        data_start = (backtest_start - timedelta(days=400)).strftime('%Y-%m-%d')
    
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    print(f"Backtest: {start_date} to {end_date}")
    if top_n_long or top_n_short:
        print(f"Portfolio: Top {top_n_long or 0} Longs / Top {top_n_short or 0} Shorts")
    print(f"Capital: ${capital:,.0f}\n")
    
    # Load data (no downloading!)
    print("Loading data...")
    data, tickers = get_clean_sp500_data(start_date=data_start, end_date=end_date)
    
    # Calculate signals based on strategy
    if strategy == 'combined':
        signal_calc = CombinedSignal(
            meanrev_weight=meanrev_weight,
            momentum_weight=momentum_weight,
            sector_weight=sector_weight,
            value_weight=value_weight
        )
        signals, meanrev_signals, momentum_signals, sector_signals, value_signals = signal_calc.calculate(data)
        
        # Evaluate signal quality
        quality_results = signal_calc.evaluate_signals(signals, meanrev_signals, momentum_signals,
                                                       sector_signals, value_signals)
        quality = quality_results['combined']
        
    elif strategy == 'momentum':
        signal_calc = MomentumSignal()
        signals = signal_calc.calculate(data)
        
        from signals.signal_momentum import calculate_signal_quality as calc_quality_mom
        quality = calc_quality_mom(signals)
        
    elif strategy == 'sector':
        signal_calc = SectorRelativeSignal()
        signals = signal_calc.calculate(data)
        
        from signals.signal_sector_relative import calculate_signal_quality as calc_quality_sector
        quality = calc_quality_sector(signals)
        
    elif strategy == 'value':
        signal_calc = ValueSignal()
        signals = signal_calc.calculate(data)
        
        from signals.signal_value import calculate_signal_quality as calc_quality_value
        quality = calc_quality_value(signals)
        
    else:  # meanrev
        signal_calc = MeanReversionSignal()
        signals = signal_calc.calculate(data)
        quality = calculate_signal_quality(signals)
    
    # Filter data and signals to backtest period only
    backtest_start_dt = pd.Timestamp(start_date)
    data = data[data.index.get_level_values('Date') >= backtest_start_dt]
    signals = signals[signals.index.get_level_values('Date') >= backtest_start_dt]
    
    print(f"Trading days: {len(data.index.get_level_values('Date').unique())}")
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
        description='Stock Picking Strategy with Multiple Signals'
    )
    parser.add_argument(
        '--backtest',
        action='store_true',
        help='Run historical backtest instead of live recommendations'
    )
    parser.add_argument(
        '--days',
        type=int,
        default=90,
        help='Number of days for backtest period (default: 90)'
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
        '--strategy',
        type=str,
        default='combined',
        choices=['meanrev', 'momentum', 'sector', 'value', 'combined'],
        help='Signal strategy: meanrev, momentum, sector, value, or combined (default: combined)'
    )
    parser.add_argument(
        '--meanrev-weight',
        type=float,
        default=0.5,
        help='Weight for mean reversion in combined strategy (default: 0.5)'
    )
    parser.add_argument(
        '--momentum-weight',
        type=float,
        default=0.5,
        help='Weight for momentum in combined strategy (default: 0.5)'
    )
    parser.add_argument(
        '--sector-weight',
        type=float,
        default=0.0,
        help='Weight for sector relative in combined strategy (default: 0.0)'
    )
    parser.add_argument(
        '--value-weight',
        type=float,
        default=0.0,
        help='Weight for value in combined strategy (default: 0.0)'
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
                top_n_short=args.top_n_short,
                strategy=args.strategy,
                meanrev_weight=args.meanrev_weight,
                momentum_weight=args.momentum_weight,
                sector_weight=args.sector_weight,
                value_weight=args.value_weight
            )
        else:
            run_live_strategy(
                portfolio_value=args.portfolio_value,
                days=args.days,
                top_n_long=args.top_n_long,
                top_n_short=args.top_n_short,
                strategy=args.strategy,
                meanrev_weight=args.meanrev_weight,
                momentum_weight=args.momentum_weight,
                sector_weight=args.sector_weight,
                value_weight=args.value_weight
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
