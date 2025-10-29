#!/usr/bin/env python3
"""
Download and save historical stock data.

This script downloads 10 years of S&P 500 data and saves it locally.
Run this once, then use run_strategy.py for backtesting.
"""
import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta
from data import get_sp500_tickers, fetch_stock_data, clean_data, filter_by_liquidity, save_sp500_data


def download_data():
    """Download 10 years of S&P 500 data."""
    
    print("=" * 70)
    print("DOWNLOADING 10 YEARS OF S&P 500 DATA")
    print("=" * 70)
    print("This will take several minutes...\n")
    
    # Calculate date range (10 years)
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=3650)).strftime('%Y-%m-%d')
    
    print(f"Date range: {start_date} to {end_date}\n")
    
    # Get tickers
    print("Fetching S&P 500 ticker list...")
    tickers = get_sp500_tickers()
    print(f"Found {len(tickers)} stocks\n")
    
    # Fetch data
    print("Downloading price data (this may take 5-10 minutes)...")
    data = fetch_stock_data(tickers, start_date, end_date)
    
    # Clean data
    print("\nCleaning data...")
    data = clean_data(data)
    
    # Filter by liquidity
    print("\nFiltering for liquid stocks...")
    liquid_tickers = filter_by_liquidity(data)
    data = data[data.index.get_level_values('Ticker').isin(liquid_tickers)]
    
    # Save to file
    print("\nSaving data...")
    save_sp500_data(data, liquid_tickers)
    
    print("\n" + "=" * 70)
    print("DOWNLOAD COMPLETE!")
    print("=" * 70)
    print(f"Downloaded: {len(liquid_tickers)} stocks")
    print(f"Date range: {data.index.get_level_values('Date').min()} to {data.index.get_level_values('Date').max()}")
    print(f"Trading days: {len(data.index.get_level_values('Date').unique())}")
    print(f"Total data points: {len(data):,}")
    print("\nYou can now run: python3 run_strategy.py --backtest")


if __name__ == '__main__':
    download_data()
