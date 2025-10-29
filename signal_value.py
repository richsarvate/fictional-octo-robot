"""
Value signal calculator.

Calculates buy/sell signals based on valuation metrics (P/E ratio) relative to sector peers.
Buy undervalued stocks, sell overvalued stocks.
"""
import pandas as pd
import numpy as np
import yfinance as yf
import pickle
from pathlib import Path
from datetime import datetime, timedelta


class ValueSignal:
    """
    Value signal based on P/E ratio relative to sector.
    
    Buy stocks with low P/E ratio compared to sector peers (undervalued).
    Sell stocks with high P/E ratio compared to sector peers (overvalued).
    """
    
    def __init__(self, zscore_window: int = 120, use_forward_pe: bool = False):
        """
        Initialize value signal.
        
        Args:
            zscore_window: Window for z-score normalization (default: 120 days)
            use_forward_pe: Use forward P/E instead of trailing P/E (default: False)
        """
        self.zscore_window = zscore_window
        self.use_forward_pe = use_forward_pe
        self.sector_map = None
        self.cache_dir = Path('data_cache')
        self.sector_cache_file = self.cache_dir / 'sector_map.pkl'
        self.pe_cache_file = self.cache_dir / 'pe_ratios.pkl'
    
    def _load_cached_sectors(self) -> dict:
        """Load sector data from cache."""
        if self.sector_cache_file.exists():
            with open(self.sector_cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def _load_cached_pe_ratios(self) -> dict:
        """Load P/E ratio data from cache."""
        if self.pe_cache_file.exists():
            print(f"Loading P/E ratios from cache...")
            with open(self.pe_cache_file, 'rb') as f:
                cache_data = pickle.load(f)
                # Check if cache is recent (within 7 days)
                if 'timestamp' in cache_data:
                    cache_age = datetime.now() - cache_data['timestamp']
                    if cache_age.days < 7:
                        print(f"Using cached P/E ratios (age: {cache_age.days} days)")
                        return cache_data['pe_ratios']
                    else:
                        print(f"Cache is {cache_age.days} days old, will refresh")
        return None
    
    def _save_pe_ratios_to_cache(self, pe_ratios: dict):
        """Save P/E ratio data to cache."""
        self.cache_dir.mkdir(exist_ok=True)
        cache_data = {
            'pe_ratios': pe_ratios,
            'timestamp': datetime.now()
        }
        with open(self.pe_cache_file, 'wb') as f:
            pickle.dump(cache_data, f)
        print(f"Saved P/E ratios to cache")
    
    def _fetch_pe_ratios(self, tickers: list) -> dict:
        """
        Fetch P/E ratios for all tickers.
        
        Args:
            tickers: List of stock ticker symbols
            
        Returns:
            Dictionary mapping ticker to P/E ratio
        """
        print(f"Fetching P/E ratios for {len(tickers)} stocks...")
        
        pe_ratios = {}
        batch_size = 50
        
        for i in range(0, len(tickers), batch_size):
            batch = tickers[i:i + batch_size]
            print(f"Processing batch {i//batch_size + 1}/{(len(tickers)-1)//batch_size + 1}...")
            
            for ticker in batch:
                try:
                    stock = yf.Ticker(ticker)
                    info = stock.info
                    
                    # Get P/E ratio (forward or trailing)
                    if self.use_forward_pe and 'forwardPE' in info:
                        pe = info.get('forwardPE')
                    else:
                        pe = info.get('trailingPE')
                    
                    # Validate P/E ratio (should be positive and reasonable)
                    if pe is not None and 0 < pe < 500:  # Filter out extreme values
                        pe_ratios[ticker] = pe
                    else:
                        pe_ratios[ticker] = None
                        
                except Exception as e:
                    print(f"  Warning: Could not fetch P/E for {ticker}: {e}")
                    pe_ratios[ticker] = None
        
        valid_count = sum(1 for v in pe_ratios.values() if v is not None)
        print(f"Successfully fetched P/E ratios for {valid_count}/{len(tickers)} stocks")
        
        return pe_ratios
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate value signal for all stocks.
        
        Args:
            data: DataFrame with MultiIndex (Date, Ticker) and 'close' column
            
        Returns:
            DataFrame with signal z-scores for each (date, ticker)
        """
        print("Calculating value signals...")
        
        # Get unique tickers
        tickers = data.index.get_level_values('Ticker').unique().tolist()
        
        # Load sector mapping (reuse from sector relative signal)
        if self.sector_map is None:
            self.sector_map = self._load_cached_sectors()
            if self.sector_map is None:
                print("Error: Sector mapping not found. Run sector relative signal first.")
                # Create dummy sector map
                self.sector_map = {t: 'Unknown' for t in tickers}
        
        # Try to load P/E ratios from cache
        pe_ratios = self._load_cached_pe_ratios()
        
        if pe_ratios is None:
            # Fetch fresh P/E ratios
            pe_ratios = self._fetch_pe_ratios(tickers)
            self._save_pe_ratios_to_cache(pe_ratios)
        else:
            # Check if we have all tickers
            missing_tickers = [t for t in tickers if t not in pe_ratios]
            if missing_tickers:
                print(f"Fetching P/E ratios for {len(missing_tickers)} new tickers...")
                new_pe_ratios = self._fetch_pe_ratios(missing_tickers)
                pe_ratios.update(new_pe_ratios)
                self._save_pe_ratios_to_cache(pe_ratios)
        
        # Calculate sector average P/E ratios
        sector_pe = {}
        sector_counts = {}
        
        for ticker, pe in pe_ratios.items():
            if pe is not None:
                sector = self.sector_map.get(ticker, 'Unknown')
                if sector not in sector_pe:
                    sector_pe[sector] = []
                sector_pe[sector].append(pe)
        
        # Calculate mean and std for each sector
        sector_stats = {}
        for sector, pe_list in sector_pe.items():
            if len(pe_list) > 0:
                sector_stats[sector] = {
                    'mean': np.mean(pe_list),
                    'std': np.std(pe_list) if len(pe_list) > 1 else 1.0
                }
        
        print(f"Calculated sector P/E statistics for {len(sector_stats)} sectors")
        
        # Calculate relative P/E (z-score vs sector)
        signals = []
        
        for ticker in tickers:
            ticker_data = data.xs(ticker, level='Ticker').copy()
            pe = pe_ratios.get(ticker)
            sector = self.sector_map.get(ticker, 'Unknown')
            
            if pe is not None and sector in sector_stats:
                # Calculate z-score: (stock_pe - sector_mean) / sector_std
                # Negative z-score = undervalued = BUY signal
                sector_mean = sector_stats[sector]['mean']
                sector_std = sector_stats[sector]['std']
                
                # Invert the z-score so low P/E = positive signal
                relative_pe = -(pe - sector_mean) / sector_std
                
                # Create signal (constant across all dates since P/E is point-in-time)
                ticker_data['pe_ratio'] = pe
                ticker_data['sector_pe_mean'] = sector_mean
                ticker_data['relative_pe'] = relative_pe
                ticker_data['signal'] = relative_pe
                ticker_data['signal_zscore'] = relative_pe  # Already normalized
            else:
                # No P/E data available
                ticker_data['pe_ratio'] = np.nan
                ticker_data['sector_pe_mean'] = np.nan
                ticker_data['relative_pe'] = np.nan
                ticker_data['signal'] = 0
                ticker_data['signal_zscore'] = 0
            
            ticker_data['Ticker'] = ticker
            ticker_data = ticker_data.reset_index()
            ticker_data = ticker_data.set_index(['Date', 'Ticker'])
            
            signals.append(ticker_data)
        
        # Combine all signals
        combined = pd.concat(signals)
        
        # Count valid signals
        valid_signals = combined[combined['signal_zscore'] != 0].index.get_level_values('Ticker').nunique()
        print(f"Generated value signals for {valid_signals}/{len(tickers)} stocks with P/E data")
        
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
        signals = data.xs(date, level='Date')[['signal_zscore', 'pe_ratio', 'sector_pe_mean']]
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
        # Top buy signals (highest positive z-scores = most undervalued)
        top_buys = signals.nlargest(n, 'signal_zscore')
        
        # Top sell signals (most negative z-scores = most overvalued)
        top_sells = signals.nsmallest(n, 'signal_zscore')
        
        return {
            'buy': top_buys,
            'sell': top_sells
        }


def calculate_signal_quality(data: pd.DataFrame, forward_returns_days: int = 20) -> dict:
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
        
        # Drop NaN and zero signals
        ticker_data = ticker_data.dropna()
        ticker_data = ticker_data[ticker_data['signal_zscore'] != 0]
        
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
    
    print("Testing value signal...\n")
    
    # Get recent data (P/E ratios don't change daily, so we don't need much history)
    start = (datetime.now() - timedelta(days=60)).strftime('%Y-%m-%d')
    data, tickers = get_clean_sp500_data(start_date=start)
    
    # Calculate signals
    signal_calc = ValueSignal()
    signals = signal_calc.calculate(data)
    
    print(f"\nSignal data shape: {signals.shape}")
    print(f"\nSample signals:")
    print(signals[signals['signal_zscore'] != 0].tail(10))
    
    # Get current signals
    current = signal_calc.get_current_signals(signals)
    print(f"\nCurrent signals (latest date):")
    print(current.head(10))
    
    # Get top signals
    top = signal_calc.get_top_signals(current, n=10)
    print(f"\nTop 10 BUY signals (most undervalued):")
    print(top['buy'])
    print(f"\nTop 10 SELL signals (most overvalued):")
    print(top['sell'])
    
    # Calculate signal quality
    quality = calculate_signal_quality(signals)
    print(f"\nSignal Quality Metrics:")
    print(f"Information Coefficient: {quality['information_coefficient']:.4f}")
    print(f"Win Rate: {quality['win_rate']:.2%}")
    print(f"Sample Size: {quality['sample_size']}")
