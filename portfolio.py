"""
Portfolio construction and position sizing.

Converts signals into actual portfolio positions with constraints.
"""
import pandas as pd
import numpy as np
from typing import Dict, List


class Portfolio:
    """
    Portfolio manager for mean reversion strategy.
    
    Handles position sizing, constraints, and rebalancing.
    """
    
    def __init__(self, 
                 min_position: float = 0.001,
                 max_position: float = 0.02,
                 max_sector_weight: float = 0.25,
                 max_turnover: float = 0.50,
                 transaction_cost: float = 0.001,
                 top_n_long: int = None,
                 top_n_short: int = None):
        """
        Initialize portfolio manager.
        
        Args:
            min_position: Minimum position size (default: 0.1%)
            max_position: Maximum position size (default: 2%)
            max_sector_weight: Maximum sector weight (default: 25%)
            max_turnover: Maximum weekly turnover (default: 50%)
            transaction_cost: Transaction cost per trade (default: 10 bps)
            top_n_long: Only buy top N longs (None = all longs)
            top_n_short: Only short top N shorts (None = all shorts)
        """
        self.min_position = min_position
        self.max_position = max_position
        self.max_sector_weight = max_sector_weight
        self.max_turnover = max_turnover
        self.transaction_cost = transaction_cost
        self.top_n_long = top_n_long
        self.top_n_short = top_n_short
    
    def calculate_positions(self, signals: pd.DataFrame, 
                          current_positions: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate target portfolio positions from signals.
        
        Args:
            signals: DataFrame with signal_zscore for each ticker
            current_positions: Current portfolio positions (for turnover constraint)
            
        Returns:
            DataFrame with target positions (weights) for each ticker
        """
        # If top_n specified, only take top longs and shorts
        if self.top_n_long is not None or self.top_n_short is not None:
            return self._calculate_top_n_positions(signals, current_positions)
        
        # Original logic: use all stocks
        # Start with equal weight base
        num_stocks = len(signals)
        base_weight = 1.0 / num_stocks
        
        # Adjust by signal strength
        # Formula: weight = base_weight * (1 + signal_zscore / 2)
        signal_adj = 1 + (signals['signal_zscore'] / 2)
        positions = base_weight * signal_adj
        
        # Apply position limits
        positions = positions.clip(lower=self.min_position, upper=self.max_position)
        
        # Normalize to sum to 1.0
        positions = positions / positions.sum()
        
        # Create result DataFrame
        result = pd.DataFrame({
            'weight': positions,
            'signal_zscore': signals['signal_zscore']
        })
        
        # Apply turnover constraint if current positions provided
        if current_positions is not None:
            result = self._apply_turnover_constraint(result, current_positions)
        
        return result
    
    def _calculate_top_n_positions(self, signals: pd.DataFrame,
                                   current_positions: pd.DataFrame = None) -> pd.DataFrame:
        """
        Calculate positions using only top N longs and shorts.
        
        Args:
            signals: DataFrame with signal_zscore for each ticker
            current_positions: Current portfolio positions (for turnover constraint)
            
        Returns:
            DataFrame with target positions (weights) for top longs/shorts only
        """
        result = pd.DataFrame(index=signals.index)
        result['signal_zscore'] = signals['signal_zscore']
        result['weight'] = 0.0
        
        # Get top N longs (highest positive z-scores)
        if self.top_n_long:
            long_signals = signals[signals['signal_zscore'] > 0].nlargest(self.top_n_long, 'signal_zscore')
            # Equal weight among longs: 50% / N
            long_weight = 0.50 / self.top_n_long
            result.loc[long_signals.index, 'weight'] = long_weight
        
        # Get top N shorts (lowest negative z-scores)
        if self.top_n_short:
            short_signals = signals[signals['signal_zscore'] < 0].nsmallest(self.top_n_short, 'signal_zscore')
            # Equal weight among shorts: 50% / N
            short_weight = 0.50 / self.top_n_short
            result.loc[short_signals.index, 'weight'] = short_weight
        
        # Filter to only stocks with positions
        result = result[result['weight'] > 0]
        
        # Don't apply turnover constraint for top-N strategy
        # (we want clean rotation of the top picks each week)
        
        return result
    
    def _apply_turnover_constraint(self, target: pd.DataFrame,
                                   current: pd.DataFrame) -> pd.DataFrame:
        """
        Apply maximum turnover constraint.
        
        Args:
            target: Target positions
            current: Current positions
            
        Returns:
            Adjusted target positions
        """
        # Skip turnover constraint on first rebalance (no current positions)
        if current.empty:
            return target
        
        # Align current and target
        all_tickers = target.index.union(current.index)
        target_aligned = target.reindex(all_tickers, fill_value=0)
        current_aligned = current.reindex(all_tickers, fill_value=0)
        
        # Calculate turnover
        changes = (target_aligned['weight'] - current_aligned.get('weight', 0)).abs()
        total_turnover = changes.sum()
        
        # If turnover exceeds max, scale down the changes
        if total_turnover > self.max_turnover:
            scale_factor = self.max_turnover / total_turnover
            adjusted_weight = current_aligned.get('weight', 0) + changes * scale_factor
            target_aligned['weight'] = adjusted_weight
            
            print(f"Applied turnover constraint: {total_turnover:.1%} -> {self.max_turnover:.1%}")
        
        return target_aligned
    
    def calculate_trades(self, target: pd.DataFrame, current: pd.DataFrame,
                        portfolio_value: float) -> pd.DataFrame:
        """
        Calculate required trades to reach target positions.
        
        Args:
            target: Target positions (weights)
            current: Current positions (weights)
            portfolio_value: Total portfolio value in dollars
            
        Returns:
            DataFrame with trade details
        """
        # Align indices
        all_tickers = target.index.union(current.index)
        target_aligned = target.reindex(all_tickers, fill_value=0)
        current_aligned = current.reindex(all_tickers, fill_value=0)
        
        # Calculate dollar changes
        target_dollars = target_aligned['weight'] * portfolio_value
        current_dollars = current_aligned.get('weight', 0) * portfolio_value
        trade_dollars = target_dollars - current_dollars
        
        # Create trades DataFrame
        trades = pd.DataFrame({
            'current_weight': current_aligned.get('weight', 0),
            'target_weight': target_aligned['weight'],
            'current_dollars': current_dollars,
            'target_dollars': target_dollars,
            'trade_dollars': trade_dollars,
            'action': ['BUY' if x > 100 else 'SELL' if x < -100 else 'HOLD' 
                      for x in trade_dollars]
        })
        
        # Filter to only actual trades
        trades = trades[trades['action'] != 'HOLD']
        trades = trades.sort_values('trade_dollars', ascending=False)
        
        # Calculate costs
        total_cost = trades['trade_dollars'].abs().sum() * self.transaction_cost
        
        print(f"\nTrade Summary:")
        print(f"Total trades: {len(trades)}")
        print(f"Total turnover: ${trades['trade_dollars'].abs().sum():,.0f}")
        print(f"Estimated costs: ${total_cost:,.0f}")
        
        return trades
    
    def format_recommendations(self, positions: pd.DataFrame, 
                              portfolio_value: float = 1_000_000) -> str:
        """
        Format portfolio recommendations for display.
        
        Args:
            positions: Target positions with weights and signals
            portfolio_value: Portfolio value in dollars (default: $1M)
            
        Returns:
            Formatted string with recommendations
        """
        # Calculate dollar amounts
        positions['dollars'] = positions['weight'] * portfolio_value
        
        # Separate long and short based on signal
        long_positions = positions[positions['signal_zscore'] > 0].copy()
        short_positions = positions[positions['signal_zscore'] < 0].copy()
        
        # Sort by signal strength
        long_positions = long_positions.sort_values('signal_zscore', ascending=False)
        short_positions = short_positions.sort_values('signal_zscore', ascending=True)
        
        # Format output
        output = []
        output.append(f"=== Portfolio Recommendations ===\n")
        output.append(f"Portfolio Value: ${portfolio_value:,.0f}\n")
        
        output.append(f"\nBUY (Long Positions): {len(long_positions)} stocks")
        output.append("-" * 70)
        output.append(f"{'Ticker':<8} {'Weight':<8} {'Dollars':<15} {'Z-Score':<10} {'Signal'}")
        output.append("-" * 70)
        
        for ticker, row in long_positions.head(20).iterrows():
            signal_strength = self._signal_description(row['signal_zscore'])
            output.append(
                f"{ticker:<8} {row['weight']*100:>6.2f}%  "
                f"${row['dollars']:>12,.0f}  "
                f"{row['signal_zscore']:>8.2f}  {signal_strength}"
            )
        
        if len(long_positions) > 20:
            output.append(f"... and {len(long_positions)-20} more long positions")
        
        output.append(f"\nSELL/SHORT (Short Positions): {len(short_positions)} stocks")
        output.append("-" * 70)
        output.append(f"{'Ticker':<8} {'Weight':<8} {'Dollars':<15} {'Z-Score':<10} {'Signal'}")
        output.append("-" * 70)
        
        for ticker, row in short_positions.head(20).iterrows():
            signal_strength = self._signal_description(row['signal_zscore'])
            output.append(
                f"{ticker:<8} {row['weight']*100:>6.2f}%  "
                f"${row['dollars']:>12,.0f}  "
                f"{row['signal_zscore']:>8.2f}  {signal_strength}"
            )
        
        if len(short_positions) > 20:
            output.append(f"... and {len(short_positions)-20} more short positions")
        
        # Summary stats
        output.append("\n" + "=" * 70)
        output.append(f"Total Long Weight: {long_positions['weight'].sum()*100:.1f}%")
        output.append(f"Total Short Weight: {short_positions['weight'].sum()*100:.1f}%")
        output.append(f"Net Exposure: {(long_positions['weight'].sum() - short_positions['weight'].sum())*100:.1f}%")
        
        return "\n".join(output)
    
    def _signal_description(self, zscore: float) -> str:
        """Get human-readable signal description."""
        if zscore > 2:
            return "Very strong oversold"
        elif zscore > 1:
            return "Strong oversold"
        elif zscore > 0.5:
            return "Oversold"
        elif zscore > -0.5:
            return "Neutral"
        elif zscore > -1:
            return "Overbought"
        elif zscore > -2:
            return "Strong overbought"
        else:
            return "Very strong overbought"


if __name__ == '__main__':
    # Test portfolio module
    from data import get_clean_sp500_data
    from signal_meanrev import MeanReversionSignal
    from datetime import datetime, timedelta
    
    print("Testing portfolio construction...\n")
    
    # Get data
    start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    data, tickers = get_clean_sp500_data(start_date=start)
    
    # Calculate signals
    signal_calc = MeanReversionSignal()
    signals = signal_calc.calculate(data)
    
    # Get current signals
    current_signals = signal_calc.get_current_signals(signals)
    
    # Build portfolio
    portfolio = Portfolio()
    positions = portfolio.calculate_positions(current_signals)
    
    print(f"\nPortfolio positions shape: {positions.shape}")
    print(f"\nTop 10 positions:")
    print(positions.sort_values('weight', ascending=False).head(10))
    
    # Format recommendations
    recommendations = portfolio.format_recommendations(positions)
    print("\n" + recommendations)
