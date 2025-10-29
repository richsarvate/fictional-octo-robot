"""
Mean reversion signal calculator.

Calculates buy/sell signals based on price deviation from moving average.
"""
import pandas as pd
import numpy as np


class MeanReversionSignal:
    """
    Mean reversion signal based on deviation from moving average.
    
    Buy when price is below average (oversold).
    Sell when price is above average (overbought).
    """
    
    def __init__(self, ma_window: int = 20, smooth_window: int = 5, 
                 zscore_window: int = 252):
        """
        Initialize mean reversion signal.
        
        Args:
            ma_window: Moving average window (default: 20 days)
            smooth_window: Smoothing window (default: 5 days)
            zscore_window: Window for z-score normalization (default: 252 days)
        """
        self.ma_window = ma_window
        self.smooth_window = smooth_window
        self.zscore_window = zscore_window
    
    def calculate(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Calculate mean reversion signal for all stocks.
        
        Args:
            data: DataFrame with MultiIndex (Date, Ticker) and 'close' column
            
        Returns:
            DataFrame with signal z-scores for each (date, ticker)
        """
        print("Calculating mean reversion signals...")
        
        # Calculate for each ticker separately
        signals = []
        
        for ticker in data.index.get_level_values('Ticker').unique():
            ticker_data = data.xs(ticker, level='Ticker')
            ticker_signal = self._calculate_single(ticker_data['close'], ticker)
            signals.append(ticker_signal)
        
        # Combine all signals
        combined = pd.concat(signals)
        
        print(f"Calculated signals for {len(data.index.get_level_values('Ticker').unique())} stocks")
        
        return combined
    
    def _calculate_single(self, prices: pd.Series, ticker: str) -> pd.DataFrame:
        """
        Calculate signal for a single stock.
        
        Args:
            prices: Series of closing prices
            ticker: Stock ticker symbol
            
        Returns:
            DataFrame with signal and z-score
        """
        # Calculate moving average
        ma = prices.rolling(window=self.ma_window).mean()
        
        # Calculate percentage deviation from MA
        deviation = (prices - ma) / ma
        
        # Smooth deviation over smooth_window days
        # Invert so negative deviation (below MA) = positive signal (buy)
        signal = -deviation.rolling(window=self.smooth_window).mean()
        
        # Normalize to z-score
        signal_mean = signal.rolling(window=self.zscore_window).mean()
        signal_std = signal.rolling(window=self.zscore_window).std()
        signal_zscore = (signal - signal_mean) / signal_std
        
        # Create result dataframe
        result = pd.DataFrame({
            'close': prices,
            'ma': ma,
            'deviation': deviation,
            'signal': signal,
            'signal_zscore': signal_zscore,
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
        # Top buy signals (highest positive z-scores = most oversold)
        top_buys = signals.nlargest(n, 'signal_zscore')
        
        # Top sell signals (most negative z-scores = most overbought)
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
    
    combined = pd.concat(results)
    
    # Calculate correlation (Information Coefficient)
    ic = combined['signal_zscore'].corr(combined['forward_return'])
    
    # Calculate win rate (signal direction matches return direction)
    correct_direction = ((combined['signal_zscore'] > 0) & (combined['forward_return'] > 0)) | \
                       ((combined['signal_zscore'] < 0) & (combined['forward_return'] < 0))
    win_rate = correct_direction.sum() / len(correct_direction)
    
    return {
        'information_coefficient': ic,
        'win_rate': win_rate,
        'sample_size': len(combined)
    }


if __name__ == '__main__':
    # Test the signal module
    from data import get_clean_sp500_data
    from datetime import datetime, timedelta
    
    print("Testing mean reversion signal...\n")
    
    # Get 2 years of data
    start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    data, tickers = get_clean_sp500_data(start_date=start)
    
    # Calculate signals
    signal_calc = MeanReversionSignal()
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
    print(f"\nTop 10 BUY signals (most oversold):")
    print(top['buy'])
    print(f"\nTop 10 SELL signals (most overbought):")
    print(top['sell'])
    
    # Calculate signal quality
    quality = calculate_signal_quality(signals)
    print(f"\nSignal Quality Metrics:")
    print(f"Information Coefficient: {quality['information_coefficient']:.4f}")
    print(f"Win Rate: {quality['win_rate']:.2%}")
