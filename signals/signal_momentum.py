"""
Momentum signal calculator.

Calculates buy/sell signals based on price momentum over multiple timeframes.
"""
import pandas as pd
import numpy as np


class MomentumSignal:
    """
    Momentum signal based on recent price trends.
    
    Buy when price momentum is positive (trending up).
    Sell when price momentum is negative (trending down).
    """
    
    def __init__(self, short_window: int = 10, medium_window: int = 30, 
                 long_window: int = 60, zscore_window: int = 120):
        """
        Initialize momentum signal.
        
        Args:
            short_window: Short-term momentum window (default: 10 days)
            medium_window: Medium-term momentum window (default: 30 days)
            long_window: Long-term momentum window (default: 60 days)
            zscore_window: Window for z-score normalization (default: 120 days)
        """
        self.short_window = short_window
        self.medium_window = medium_window
        self.long_window = long_window
        self.zscore_window = zscore_window
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate momentum signal for all stocks.
        
        Args:
            data: DataFrame with MultiIndex (Date, Ticker) and 'close' column
            
        Returns:
            DataFrame with signal z-scores for each (date, ticker)
        """
        print("Calculating momentum signals...")
        
        # Calculate for each ticker separately
        signals = []
        
        for ticker in data.index.get_level_values('Ticker').unique():
            ticker_data = data.xs(ticker, level='Ticker')
            ticker_signal = self._calculate_single(ticker_data['close'], ticker)
            signals.append(ticker_signal)
        
        # Combine all signals
        combined = pd.concat(signals)
        
        print(f"Calculated momentum signals for {len(data.index.get_level_values('Ticker').unique())} stocks")
        
        return combined
    
    def _calculate_single(self, prices: pd.Series, ticker: str) -> pd.DataFrame:
        """
        Calculate momentum signal for a single stock.
        
        Args:
            prices: Series of closing prices
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with momentum signal and z-score
        """
        # Calculate returns over different timeframes
        short_return = prices.pct_change(self.short_window)
        medium_return = prices.pct_change(self.medium_window)
        long_return = prices.pct_change(self.long_window)
        
        # Combine momentum signals with equal weighting
        # Positive returns = positive signal (buy)
        momentum = (short_return + medium_return + long_return) / 3
        
        # Normalize to z-score
        momentum_mean = momentum.rolling(window=self.zscore_window).mean()
        momentum_std = momentum.rolling(window=self.zscore_window).std()
        momentum_zscore = (momentum - momentum_mean) / momentum_std
        
        # Create result dataframe
        result = pd.DataFrame({
            'close': prices,
            'short_return': short_return,
            'medium_return': medium_return,
            'long_return': long_return,
            'signal': momentum,
            'signal_zscore': momentum_zscore,
            'Ticker': ticker
        })
        
        # Set index
        result = result.reset_index()
        result = result.set_index(['Date', 'Ticker'])
        
        return result
    
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
        signals = data.xs(date, level='Date')[['signal_zscore']]
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
        # Top buy signals (highest positive z-scores = strongest momentum)
        top_buys = signals.nlargest(n, 'signal_zscore')
        
        # Top sell signals (most negative z-scores = weakest momentum)
        top_sells = signals.nsmallest(n, 'signal_zscore')
        
        return {
            'buy': top_buys,
            'sell': top_sells
        }


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
    
    print("Testing momentum signal...\n")
    
    # Get 2 years of data
    start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    data, tickers = get_clean_sp500_data(start_date=start)
    
    # Calculate signals
    signal_calc = MomentumSignal()
    signals = signal_calc.calculate(data)
    
    print(f"\nSignal data shape: {signals.shape}")
    print(f"\nSample signals:")
    print(signals.tail(10))
    
    # Get current signals
    current = signal_calc.get_current_signals(signals)
    print(f"\nCurrent signals (latest date):")
    print(current.head(10))
    
    # Get top signals
    top = signal_calc.get_top_signals(current, n=10)
    print(f"\nTop 10 BUY signals (strongest momentum):")
    print(top['buy'])
    print(f"\nTop 10 SELL signals (weakest momentum):")
    print(top['sell'])
    
    # Calculate signal quality
    quality = calculate_signal_quality(signals)
    print(f"\nSignal Quality Metrics:")
    print(f"Information Coefficient: {quality['information_coefficient']:.4f}")
    print(f"Win Rate: {quality['win_rate']:.2%}")
