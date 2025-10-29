"""
Sector relative signal calculator.

Calculates buy/sell signals based on stock performance relative to sector peers.
Identifies sector leaders (outperformers) and laggards (underperformers).
"""
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
import os
from pathlib import Path


class SectorRelativeSignal:
    """
    Sector relative signal based on performance vs sector peers.
    
    Buy stocks that are outperforming their sector (sector leaders).
    Sell stocks that are underperforming their sector (sector laggards).
    """
    
    def __init__(self, lookback_window: int = 20, zscore_window: int = 120):
        """
        Initialize sector relative signal.
        
        Args:
            lookback_window: Window for calculating relative performance (default: 20 days)
            zscore_window: Window for z-score normalization (default: 120 days)
        """
        self.lookback_window = lookback_window
        self.zscore_window = zscore_window
        self.sector_map = None
        self.cache_dir = Path('data/data_cache')
        self.cache_file = self.cache_dir / 'sector_map.pkl'
    
    def _load_cached_sectors(self) -> dict:
        """
        Load sector data from cache if available.
        
        Returns:
            Dictionary mapping ticker to sector, or None if cache doesn't exist
        """
        if self.cache_file.exists():
            print(f"Loading sector data from cache: {self.cache_file}")
            with open(self.cache_file, 'rb') as f:
                sector_map = pickle.load(f)
            print(f"Loaded {len(sector_map)} tickers from cache")
            return sector_map
        return None
    
    def _save_sectors_to_cache(self, sector_map: dict):
        """
        Save sector data to cache.
        
        Args:
            sector_map: Dictionary mapping ticker to sector
        """
        # Create cache directory if it doesn't exist
        self.cache_dir.mkdir(exist_ok=True)
        
        with open(self.cache_file, 'wb') as f:
            pickle.dump(sector_map, f)
        print(f"Saved sector data to cache: {self.cache_file}")
    
    def _fetch_sector_data(self, tickers: list) -> dict:
        """
        Fetch sector information for all tickers.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary mapping ticker to sector
        """
        print(f"Fetching sector data for {len(tickers)} stocks...")
        
        sector_map = {}
        batch_size = 50
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i+batch_size]
            for ticker in batch:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    sector = info.get('sector', 'Unknown')
                    sector_map[ticker] = sector
                except Exception as e:
                    sector_map[ticker] = 'Unknown'
                    
            if (i + batch_size) % 100 == 0:
                print(f"  Processed {min(i + batch_size, len(tickers))}/{len(tickers)} tickers...")
        
        print(f"Successfully mapped {len([s for s in sector_map.values() if s != 'Unknown'])} tickers to sectors")
        return sector_map
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate sector relative signal for all stocks.
        
        Args:
            data: DataFrame with MultiIndex (Date, Ticker) and 'close' column
            
        Returns:
            DataFrame with signal z-scores for each (date, ticker)
        """
        print("Calculating sector relative signals...")
        
        # Get unique tickers
        tickers = data.index.get_level_values('Ticker').unique().tolist()
        
        # Try to load from cache first
        if self.sector_map is None:
            self.sector_map = self._load_cached_sectors()
        
        # If cache exists, check if we have all tickers
        if self.sector_map is not None:
            missing_tickers = [t for t in tickers if t not in self.sector_map]
            if missing_tickers:
                print(f"Cache missing {len(missing_tickers)} tickers, fetching...")
                new_sectors = self._fetch_sector_data(missing_tickers)
                self.sector_map.update(new_sectors)
                self._save_sectors_to_cache(self.sector_map)
        else:
            # No cache, fetch all sector data
            self.sector_map = self._fetch_sector_data(tickers)
            self._save_sectors_to_cache(self.sector_map)
        
        # Calculate returns for each stock
        returns_data = []
        
        for ticker in tickers:
            ticker_data = data.xs(ticker, level='Ticker')
            ticker_returns = ticker_data['close'].pct_change(self.lookback_window)
            
            returns_df = pd.DataFrame({
                'return': ticker_returns,
                'Ticker': ticker,
                'sector': self.sector_map.get(ticker, 'Unknown')
            }, index=ticker_data.index)
            
            returns_data.append(returns_df)
        
        # Combine all returns
        all_returns = pd.concat(returns_data)
        all_returns = all_returns.reset_index()
        
        # Calculate sector average returns for each date
        sector_avg = all_returns.groupby(['Date', 'sector'])['return'].mean().reset_index()
        sector_avg.columns = ['Date', 'sector', 'sector_return']
        
        # Merge sector averages back
        all_returns = all_returns.merge(sector_avg, on=['Date', 'sector'], how='left')
        
        # Calculate relative performance (stock return - sector return)
        all_returns['relative_perf'] = all_returns['return'] - all_returns['sector_return']
        
        # Calculate signals for each ticker
        signals = []
        
        for ticker in tickers:
            ticker_data = all_returns[all_returns['Ticker'] == ticker].copy()
            ticker_data = ticker_data.set_index('Date').sort_index()
            
            # Calculate z-score of relative performance
            rel_perf = ticker_data['relative_perf']
            rel_perf_mean = rel_perf.rolling(window=self.zscore_window).mean()
            rel_perf_std = rel_perf.rolling(window=self.zscore_window).std()
            signal_zscore = (rel_perf - rel_perf_mean) / rel_perf_std
            
            # Create result dataframe
            result = pd.DataFrame({
                'close': data.xs(ticker, level='Ticker')['close'],
                'return': ticker_data['return'],
                'sector_return': ticker_data['sector_return'],
                'relative_perf': ticker_data['relative_perf'],
                'signal': rel_perf,
                'signal_zscore': signal_zscore,
                'sector': ticker_data['sector'],
                'Ticker': ticker
            })
            
            result = result.reset_index()
            result = result.set_index(['Date', 'Ticker'])
            
            signals.append(result)
        
        # Combine all signals
        combined = pd.concat(signals)
        
        print(f"Calculated sector relative signals for {len(tickers)} stocks")
        
        # Print sector distribution
        sector_counts = pd.Series(self.sector_map).value_counts()
        print(f"\nSector distribution:")
        for sector, count in sector_counts.head(10).items():
            print(f"  {sector}: {count} stocks")
        
        return combined
    
    def get_current_signals(self, data: pd.DataFrame, date: str = None) -> pd.DataFrame:
        """
        Get signals for a specific date (latest by default).
        
        Args:
            data: DataFrame with signals
            date: Date to get signals for (default: latest date)
            
        Returns:
            DataFrame with tickers and their signals for that date
        """
        if date is None:
            date = data.index.get_level_values('Date').max()
        
        # Get signals for specified date
        signals = data.xs(date, level='Date')[['signal_zscore', 'sector', 'relative_perf']]
        signals = signals.sort_values('signal_zscore', ascending=False)
        
        return signals
    
    def get_top_signals(self, signals: pd.DataFrame, n: int = 50) -> dict:
        """
        Get top buy and sell signals.
        
        Args:
            signals: DataFrame with signal_zscore column
            n: Number of top signals to return
            
        Returns:
            Dictionary with 'buy' and 'sell' DataFrames
        """
        # Top buy signals (highest positive z-scores = strongest sector outperformers)
        top_buys = signals.nlargest(n, 'signal_zscore')
        
        # Top sell signals (most negative z-scores = worst sector underperformers)
        top_sells = signals.nsmallest(n, 'signal_zscore')
        
        return {
            'buy': top_buys,
            'sell': top_sells
        }
    
    def get_sector_leaders(self, signals: pd.DataFrame, n_per_sector: int = 3) -> pd.DataFrame:
        """
        Get top performers from each sector.
        
        Args:
            signals: DataFrame with signals and sector info
            n_per_sector: Number of top stocks per sector
            
        Returns:
            DataFrame with top performers from each sector
        """
        leaders = []
        
        for sector in signals['sector'].unique():
            if sector == 'Unknown':
                continue
                
            sector_stocks = signals[signals['sector'] == sector]
            top_in_sector = sector_stocks.nlargest(n_per_sector, 'signal_zscore')
            leaders.append(top_in_sector)
        
        if len(leaders) > 0:
            return pd.concat(leaders).sort_values('signal_zscore', ascending=False)
        else:
            return pd.DataFrame()


def calculate_signal_quality(data: pd.DataFrame, forward_returns_days: int = 5) -> dict:
    """
    Calculate signal quality metrics (Information Coefficient).
    
    Args:
        data: DataFrame with signal_zscore and price data
        forward_returns_days: Days to calculate forward returns
        
    Returns:
        Dictionary with quality metrics
    """
    results = []
    
    for ticker in data.index.get_level_values('Ticker').unique():
        ticker_data = data.xs(ticker, level='Ticker').copy()
        
        # Calculate forward returns
        ticker_data['forward_return'] = ticker_data['close'].pct_change(forward_returns_days).shift(-forward_returns_days)
        
        # Drop NaN
        ticker_data = ticker_data.dropna()
        
        if len(ticker_data) > 0:
            results.append(ticker_data[['signal_zscore', 'forward_return']])
    
    if len(results) == 0:
        return {
            'information_coefficient': 0.0,
            'win_rate': 0.5,
            'sample_size': 0
        }
    
    combined = pd.concat(results)
    
    if len(combined) == 0:
        return {
            'information_coefficient': 0.0,
            'win_rate': 0.5,
            'sample_size': 0
        }
    
    # Calculate correlation (Information Coefficient)
    ic = combined['signal_zscore'].corr(combined['forward_return'])
    
    # Calculate win rate (signal direction matches return direction)
    correct_direction = ((combined['signal_zscore'] > 0) & (combined['forward_return'] > 0)) | \
                       ((combined['signal_zscore'] < 0) & (combined['forward_return'] < 0))
    win_rate = correct_direction.sum() / len(correct_direction)
    
    return {
        'information_coefficient': ic if not pd.isna(ic) else 0.0,
        'win_rate': win_rate,
        'sample_size': len(combined)
    }


if __name__ == '__main__':
    # Test the signal module
    from data import get_clean_sp500_data
    from datetime import datetime, timedelta
    
    print("Testing sector relative signal...\n")
    
    # Get 2 years of data
    start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    data, tickers = get_clean_sp500_data(start_date=start)
    
    # Calculate signals
    signal_calc = SectorRelativeSignal()
    signals = signal_calc.calculate(data)
    
    print(f"\nSignal data shape: {signals.shape}")
    print(f"\nSample signals:")
    print(signals[['signal_zscore', 'sector', 'relative_perf']].tail(10))
    
    # Get current signals
    current = signal_calc.get_current_signals(signals)
    print(f"\nCurrent signals (latest date):")
    print(current.head(10))
    
    # Get top signals
    top = signal_calc.get_top_signals(current, n=10)
    print(f"\nTop 10 BUY signals (sector outperformers):")
    print(top['buy'])
    print(f"\nTop 10 SELL signals (sector underperformers):")
    print(top['sell'])
    
    # Get sector leaders
    leaders = signal_calc.get_sector_leaders(current, n_per_sector=2)
    print(f"\nTop 2 performers from each sector:")
    print(leaders)
    
    # Calculate signal quality
    quality = calculate_signal_quality(signals)
    print(f"\nSignal Quality Metrics:")
    print(f"Information Coefficient: {quality['information_coefficient']:.4f}")
    print(f"Win Rate: {quality['win_rate']:.2%}")
