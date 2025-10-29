"""
Combined signal calculator.

Combines multiple signals (mean reversion, momentum, sector relative, and value) into a single composite signal.
"""
import pandas as pd
import numpy as np
from signals.signal_meanrev import MeanReversionSignal, calculate_signal_quality as calc_quality_meanrev
from signals.signal_momentum import MomentumSignal, calculate_signal_quality as calc_quality_momentum
from signals.signal_sector_relative import SectorRelativeSignal, calculate_signal_quality as calc_quality_sector
from signals.signal_value import ValueSignal, calculate_signal_quality as calc_quality_value


class CombinedSignal:
    """
    Combined signal that blends mean reversion, momentum, sector relative, and value signals.
    
    Uses weighted average of individual signals to create composite signal.
    """
    
    def __init__(self, meanrev_weight: float = 0.25, momentum_weight: float = 0.25,
                 sector_weight: float = 0.25, value_weight: float = 0.25,
                 meanrev_params: dict = None, momentum_params: dict = None,
                 sector_params: dict = None, value_params: dict = None):
        """
        Initialize combined signal.
        
        Args:
            meanrev_weight: Weight for mean reversion signal (default: 0.25)
            momentum_weight: Weight for momentum signal (default: 0.25)
            sector_weight: Weight for sector relative signal (default: 0.25)
            value_weight: Weight for value signal (default: 0.25)
            meanrev_params: Parameters for mean reversion signal (optional)
            momentum_params: Parameters for momentum signal (optional)
            sector_params: Parameters for sector relative signal (optional)
            value_params: Parameters for value signal (optional)
        """
        self.meanrev_weight = meanrev_weight
        self.momentum_weight = momentum_weight
        self.sector_weight = sector_weight
        self.value_weight = value_weight
        
        # Normalize weights to sum to 1
        total = meanrev_weight + momentum_weight + sector_weight + value_weight
        self.meanrev_weight /= total
        self.momentum_weight /= total
        self.sector_weight /= total
        self.value_weight /= total
        
        # Initialize signal calculators
        meanrev_params = meanrev_params or {}
        momentum_params = momentum_params or {}
        sector_params = sector_params or {}
        value_params = value_params or {}
        
        self.meanrev_calc = MeanReversionSignal(**meanrev_params)
        self.momentum_calc = MomentumSignal(**momentum_params)
        self.sector_calc = SectorRelativeSignal(**sector_params)
        self.value_calc = ValueSignal(**value_params)
        
        print(f"Combined Signal Weights: Mean Rev={self.meanrev_weight:.1%}, " +
              f"Momentum={self.momentum_weight:.1%}, Sector={self.sector_weight:.1%}, " +
              f"Value={self.value_weight:.1%}")
    
    def calculate(self, data: pd.DataFrame) -> tuple:
        """
        Calculate combined signal for all stocks.
        
        Args:
            data: DataFrame with MultiIndex (Date, Ticker) and 'close' column
            
        Returns:
            Tuple of (combined_signals, meanrev_signals, momentum_signals, sector_signals, value_signals)
        """
        print("\n" + "=" * 70)
        print("CALCULATING COMBINED SIGNALS (4 SIGNALS)")
        print("=" * 70)
        
        # Calculate individual signals
        meanrev_signals = self.meanrev_calc.calculate(data)
        momentum_signals = self.momentum_calc.calculate(data)
        sector_signals = self.sector_calc.calculate(data)
        value_signals = self.value_calc.calculate(data)
        
        # Combine signals
        print("\nCombining all signals...")
        combined = self._combine_signals(meanrev_signals, momentum_signals, 
                                         sector_signals, value_signals)
        
        return combined, meanrev_signals, momentum_signals, sector_signals, value_signals
    
    def _combine_signals(self, meanrev_signals: pd.DataFrame, 
                         momentum_signals: pd.DataFrame,
                         sector_signals: pd.DataFrame,
                         value_signals: pd.DataFrame) -> pd.DataFrame:
        """
        Combine all four signals.
        
        Args:
            meanrev_signals: Mean reversion signal DataFrame
            momentum_signals: Momentum signal DataFrame
            sector_signals: Sector relative signal DataFrame
            value_signals: Value signal DataFrame
            
        Returns:
            Combined signal DataFrame
        """
        # Extract just the signal z-scores
        mr_zscore = meanrev_signals[['signal_zscore']].rename(
            columns={'signal_zscore': 'meanrev_zscore'}
        )
        mom_zscore = momentum_signals[['signal_zscore']].rename(
            columns={'signal_zscore': 'momentum_zscore'}
        )
        sector_zscore = sector_signals[['signal_zscore']].rename(
            columns={'signal_zscore': 'sector_zscore'}
        )
        value_zscore = value_signals[['signal_zscore']].rename(
            columns={'signal_zscore': 'value_zscore'}
        )
        
        # Merge on index (Date, Ticker)
        combined = mr_zscore.join(mom_zscore, how='inner')
        combined = combined.join(sector_zscore, how='inner')
        combined = combined.join(value_zscore, how='inner')
        
        # Calculate weighted average
        combined['signal_zscore'] = (
            self.meanrev_weight * combined['meanrev_zscore'] + 
            self.momentum_weight * combined['momentum_zscore'] +
            self.sector_weight * combined['sector_zscore'] +
            self.value_weight * combined['value_zscore']
        )
        
        # Add close price for reference
        combined = combined.join(meanrev_signals[['close']])
        
        print(f"Combined signals for {len(combined.index.get_level_values('Ticker').unique())} stocks")
        
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
        signals = data.xs(date, level='Date')
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
        # Top buy signals (highest positive z-scores)
        top_buys = signals.nlargest(n, 'signal_zscore')
        
        # Top sell signals (most negative z-scores)
        top_sells = signals.nsmallest(n, 'signal_zscore')
        
        return {
            'buy': top_buys,
            'sell': top_sells
        }
    
    def evaluate_signals(self, combined_signals: pd.DataFrame,
                        meanrev_signals: pd.DataFrame,
                        momentum_signals: pd.DataFrame,
                        sector_signals: pd.DataFrame,
                        value_signals: pd.DataFrame) -> dict:
        """
        Evaluate quality of all signals.
        
        Args:
            combined_signals: Combined signal DataFrame
            meanrev_signals: Mean reversion signal DataFrame
            momentum_signals: Momentum signal DataFrame
            sector_signals: Sector relative signal DataFrame
            value_signals: Value signal DataFrame
            
        Returns:
            Dictionary with quality metrics for all signals
        """
        print("\n" + "=" * 70)
        print("SIGNAL QUALITY EVALUATION")
        print("=" * 70)
        
        # Calculate quality for each signal type
        mr_quality = calc_quality_meanrev(meanrev_signals)
        mom_quality = calc_quality_momentum(momentum_signals)
        sector_quality = calc_quality_sector(sector_signals)
        value_quality = calc_quality_value(value_signals)
        
        # For combined signal, we need to prepare data in same format
        combined_quality = self._calculate_combined_quality(combined_signals)
        
        results = {
            'mean_reversion': mr_quality,
            'momentum': mom_quality,
            'sector_relative': sector_quality,
            'value': value_quality,
            'combined': combined_quality
        }
        
        # Print results
        print("\nMean Reversion Signal:")
        print(f"  IC: {mr_quality['information_coefficient']:.4f}")
        print(f"  Win Rate: {mr_quality['win_rate']:.1%}")
        
        print("\nMomentum Signal:")
        print(f"  IC: {mom_quality['information_coefficient']:.4f}")
        print(f"  Win Rate: {mom_quality['win_rate']:.1%}")
        
        print("\nSector Relative Signal:")
        print(f"  IC: {sector_quality['information_coefficient']:.4f}")
        print(f"  Win Rate: {sector_quality['win_rate']:.1%}")
        
        print("\nValue Signal:")
        print(f"  IC: {value_quality['information_coefficient']:.4f}")
        print(f"  Win Rate: {value_quality['win_rate']:.1%}")
        
        print("\nCombined Signal (All 4):")
        print(f"  IC: {combined_quality['information_coefficient']:.4f}")
        print(f"  Win Rate: {combined_quality['win_rate']:.1%}")
        
        return results
    
    def _calculate_combined_quality(self, data: pd.DataFrame, 
                                    forward_returns_days: int = 5) -> dict:
        """
        Calculate signal quality for combined signal.
        
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
            ticker_data['forward_return'] = ticker_data['close'].pct_change(
                forward_returns_days).shift(-forward_returns_days)
            
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
        
        # Calculate win rate
        correct_direction = (
            ((combined['signal_zscore'] > 0) & (combined['forward_return'] > 0)) | 
            ((combined['signal_zscore'] < 0) & (combined['forward_return'] < 0))
        )
        win_rate = correct_direction.sum() / len(correct_direction)
        
        return {
            'information_coefficient': ic if not pd.isna(ic) else 0.0,
            'win_rate': win_rate,
            'sample_size': len(combined)
        }


if __name__ == '__main__':
    # Test the combined signal module
    from data import get_clean_sp500_data
    from datetime import datetime, timedelta
    
    print("Testing combined signal...\n")
    
    # Get 2 years of data
    start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    data, tickers = get_clean_sp500_data(start_date=start)
    
    # Calculate combined signals
    signal_calc = CombinedSignal(meanrev_weight=0.6, momentum_weight=0.4)
    combined, meanrev, momentum = signal_calc.calculate(data)
    
    print(f"\nCombined signal data shape: {combined.shape}")
    print(f"\nSample combined signals:")
    print(combined.tail(10))
    
    # Get current signals
    current = signal_calc.get_current_signals(combined)
    print(f"\nCurrent combined signals (latest date):")
    print(current.head(10))
    
    # Get top signals
    top = signal_calc.get_top_signals(current, n=10)
    print(f"\nTop 10 BUY signals:")
    print(top['buy'])
    print(f"\nTop 10 SELL signals:")
    print(top['sell'])
    
    # Evaluate signal quality
    quality = signal_calc.evaluate_signals(combined, meanrev, momentum)
