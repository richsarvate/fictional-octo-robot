"""
Data module for fetching and managing stock price data.
"""
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import os
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Tuple


DATA_DIR = Path('./stock_data')
DATA_DIR.mkdir(exist_ok=True)


def get_sp500_tickers() -> List[str]:
    """
    Get list of S&P 500 tickers.
    
    Returns:
        List of ticker symbols
    """
    try:
        # Fetch S&P 500 tickers from Wikipedia with user agent
        import urllib.request
        url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
        req = urllib.request.Request(url, headers={'User-Agent': 'Mozilla/5.0'})
        tables = pd.read_html(urllib.request.urlopen(req))
        sp500_table = tables[0]
        tickers = sp500_table['Symbol'].tolist()
        
        # Clean tickers (replace dots with dashes for yfinance)
        tickers = [ticker.replace('.', '-') for ticker in tickers]
        
        return tickers
    except Exception as e:
        print(f"Warning: Could not fetch S&P 500 list from Wikipedia: {e}")
        print("Using sample of major tickers instead...")
        
        # Fallback to a sample of major S&P 500 stocks
        return [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'TSLA', 'BRK-B',
            'UNH', 'JNJ', 'XOM', 'JPM', 'V', 'PG', 'MA', 'HD', 'CVX', 'MRK',
            'ABBV', 'PEP', 'KO', 'AVGO', 'COST', 'WMT', 'MCD', 'CSCO', 'ABT',
            'TMO', 'ACN', 'DHR', 'VZ', 'ADBE', 'NFLX', 'NKE', 'DIS', 'CRM',
            'CMCSA', 'TXN', 'PM', 'WFC', 'NEE', 'RTX', 'UPS', 'LOW', 'ORCL',
            'BMY', 'QCOM', 'HON', 'AMD', 'INTU', 'UNP', 'BA', 'SBUX', 'CAT',
            'LMT', 'GE', 'DE', 'INTC', 'PLD', 'SPGI', 'GILD', 'BLK', 'AXP',
            'C', 'MMM', 'MDLZ', 'ISRG', 'AMT', 'CVS', 'TJX', 'SYK', 'ADP',
            'BKNG', 'VRTX', 'ZTS', 'ADI', 'TMUS', 'MO', 'CI', 'REGN', 'CB',
            'PGR', 'DUK', 'SO', 'BSX', 'NOC', 'GS', 'BDX', 'SCHW', 'EOG',
            'MMC', 'CL', 'ITW', 'SLB', 'AON', 'FI', 'USB', 'APD', 'CME',
            'ICE', 'PNC', 'MS', 'CCI', 'TGT', 'NSC', 'MCO', 'WM', 'HUM'
        ]


def fetch_stock_data(tickers: List[str], start_date: str, end_date: str) -> pd.DataFrame:
    """
    Fetch OHLCV data for multiple tickers.
    
    Args:
        tickers: List of ticker symbols
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format
        
    Returns:
        DataFrame with MultiIndex (date, ticker) and OHLCV columns
    """
    print(f"Fetching data for {len(tickers)} stocks from {start_date} to {end_date}...")
    
    # Download data for all tickers
    data = yf.download(tickers, start=start_date, end=end_date, group_by='ticker', 
                      auto_adjust=True, progress=True)
    
    # Restructure data to have (date, ticker) MultiIndex
    dfs = []
    
    for ticker in tickers:
        try:
            if len(tickers) == 1:
                ticker_data = data
            else:
                ticker_data = data[ticker]
            
            # Add ticker column
            ticker_data = ticker_data.copy()
            ticker_data['Ticker'] = ticker
            dfs.append(ticker_data)
        except (KeyError, AttributeError):
            print(f"Warning: No data for {ticker}")
            continue
    
    # Combine all dataframes
    combined = pd.concat(dfs)
    combined = combined.reset_index()
    combined = combined.set_index(['Date', 'Ticker'])
    
    # Rename columns to lowercase
    combined.columns = [col.lower() for col in combined.columns]
    
    return combined


def filter_by_liquidity(data: pd.DataFrame, min_market_cap: float = 1e9, 
                       min_volume: float = 1e6) -> List[str]:
    """
    Filter stocks by liquidity (market cap and volume).
    
    Args:
        data: DataFrame with price and volume data
        min_market_cap: Minimum market cap in dollars (default: $1B)
        min_volume: Minimum average daily volume in dollars (default: $1M)
        
    Returns:
        List of tickers that pass liquidity filters
    """
    # Calculate average dollar volume over last 60 days
    recent_data = data.groupby('Ticker').tail(60)
    
    # Get latest close price for market cap estimation
    latest_prices = data.groupby('Ticker')['close'].last()
    
    # Calculate average volume
    avg_volume = recent_data.groupby('Ticker')['volume'].mean()
    avg_dollar_volume = latest_prices * avg_volume
    
    # Filter by volume
    liquid_tickers = avg_dollar_volume[avg_dollar_volume >= min_volume].index.tolist()
    
    print(f"Filtered to {len(liquid_tickers)} liquid stocks (min ${min_volume:,.0f} daily volume)")
    
    return liquid_tickers


def clean_data(data: pd.DataFrame, min_data_points: int = 252) -> pd.DataFrame:
    """
    Clean data by removing stocks with insufficient history or data quality issues.
    
    Args:
        data: Raw price data
        min_data_points: Minimum number of data points required (default: 252 = 1 year)
        
    Returns:
        Cleaned DataFrame
    """
    # Count data points per ticker
    data_counts = data.groupby('Ticker').size()
    valid_tickers = data_counts[data_counts >= min_data_points].index.tolist()
    
    # Filter to valid tickers
    cleaned = data[data.index.get_level_values('Ticker').isin(valid_tickers)]
    
    # Remove rows with any NaN values
    cleaned = cleaned.dropna()
    
    # Remove stocks with zero or negative prices
    cleaned = cleaned[cleaned['close'] > 0]
    
    print(f"Cleaned data: {len(valid_tickers)} stocks with sufficient history")
    
    return cleaned


def get_clean_sp500_data(start_date: str = '2010-01-01', 
                         end_date: str = None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load clean S&P 500 data from saved file.
    
    Args:
        start_date: Start date in 'YYYY-MM-DD' format
        end_date: End date in 'YYYY-MM-DD' format (default: today)
        
    Returns:
        Tuple of (cleaned DataFrame, list of valid tickers)
    """
    if end_date is None:
        end_date = datetime.now().strftime('%Y-%m-%d')
    
    # Load from saved file
    data_file = DATA_DIR / 'sp500_data.pkl'
    
    if not data_file.exists():
        raise FileNotFoundError(
            f"Data file not found: {data_file}\n"
            f"Please run: python3 download.py"
        )
    
    print(f"Loading data from {data_file}...")
    with open(data_file, 'rb') as f:
        data, tickers = pickle.load(f)
    
    # Filter to requested date range
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)
    
    data = data[
        (data.index.get_level_values('Date') >= start_dt) &
        (data.index.get_level_values('Date') <= end_dt)
    ]
    
    print(f"Loaded {len(tickers)} stocks")
    print(f"Date range: {data.index.get_level_values('Date').min()} to {data.index.get_level_values('Date').max()}")
    print(f"Total data points: {len(data):,}")
    
    return data, tickers


def save_sp500_data(data: pd.DataFrame, tickers: List[str]):
    """
    Save cleaned data to file.
    
    Args:
        data: Cleaned DataFrame
        tickers: List of tickers
    """
    data_file = DATA_DIR / 'sp500_data.pkl'
    
    with open(data_file, 'wb') as f:
        pickle.dump((data, tickers), f)
    
    file_size = data_file.stat().st_size / (1024 * 1024)
    print(f"\nData saved to {data_file} ({file_size:.2f} MB)")


if __name__ == '__main__':
    # Test the data module
    print("Testing data module...\n")
    
    # Get 2 years of data for testing
    start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    data, tickers = get_clean_sp500_data(start_date=start)
    
    print(f"\nData shape: {data.shape}")
    print(f"\nSample data:")
    print(data.head(10))
    
    print(f"\nColumns: {data.columns.tolist()}")
    print(f"\nUnique tickers: {len(tickers)}")
