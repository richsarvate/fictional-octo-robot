"""
Backtesting framework for mean reversion strategy.

Simulates historical trading and calculates performance metrics.
"""
import pandas as pd
import numpy as np
from datetime import datetime
from typing import Dict, List, Tuple
import matplotlib.pyplot as plt


class Backtest:
    """
    Backtest engine for portfolio strategies.
    """
    
    def __init__(self, initial_capital: float = 1_000_000,
                 transaction_cost: float = 0.001,
                 rebalance_frequency: str = 'W-MON'):
        """
        Initialize backtest engine.
        
        Args:
            initial_capital: Starting portfolio value (default: $1M)
            transaction_cost: Cost per trade (default: 10 bps)
            rebalance_frequency: Rebalancing frequency (default: weekly on Monday)
        """
        self.initial_capital = initial_capital
        self.transaction_cost = transaction_cost
        self.rebalance_frequency = rebalance_frequency
        
        self.portfolio_history = []
        self.trade_history = []
        
    def run(self, data: pd.DataFrame, signals: pd.DataFrame, 
            portfolio_manager, verbose: bool = True) -> pd.DataFrame:
        """
        Run backtest simulation.
        
        Args:
            data: Price data with MultiIndex (Date, Ticker)
            signals: Signal data with MultiIndex (Date, Ticker)
            portfolio_manager: Portfolio object for position sizing
            verbose: Print detailed rebalancing actions
            
        Returns:
            DataFrame with daily portfolio values and metrics
        """
        print("Running backtest...")
        
        # Get rebalance dates
        dates = signals.index.get_level_values('Date').unique().sort_values()
        # Convert to datetime series with datetime index for resampling
        date_series = pd.Series(dates, index=pd.DatetimeIndex(dates))
        rebalance_dates = date_series.resample(self.rebalance_frequency).first().dropna()
        
        print(f"Rebalancing on {len(rebalance_dates)} dates from {dates[0]} to {dates[-1]}")
        print(f"Starting capital: ${self.initial_capital:,.2f}\n")
        
        # Initialize portfolio
        current_positions = pd.DataFrame()
        portfolio_value = self.initial_capital
        cash = self.initial_capital
        
        # Track daily values
        daily_values = []
        rebalance_count = 0
        
        # Iterate through all dates
        for date in dates:
            # Get prices for this date
            try:
                day_prices = data.xs(date, level='Date')['close']
            except KeyError:
                continue
            
            # Check if it's a rebalance date
            if date in rebalance_dates.values:
                rebalance_count += 1
                
                if verbose:
                    print(f"\nRebalance #{rebalance_count} ({date.strftime('%Y-%m-%d')}): ", end="")
                
                # Get signals for this date
                day_signals = signals.xs(date, level='Date')[['signal_zscore']]
                
                # Calculate target positions
                target_positions = portfolio_manager.calculate_positions(
                    day_signals, current_positions
                )
                
                # Calculate trades needed
                if not current_positions.empty:
                    trades = self._execute_rebalance(
                        current_positions, target_positions, day_prices, portfolio_value,
                        verbose=verbose
                    )
                    self.trade_history.extend(trades)
                    
                    if verbose and trades:
                        buys = [t for t in trades if t['value'] > 0]
                        sells = [t for t in trades if t['value'] < 0]
                        print(f"{len(buys)} buys, {len(sells)} sells, Portfolio = ${portfolio_value:,.0f}")
                        
                        # Show stocks bought and sold
                        if buys:
                            buy_tickers = [t['ticker'] for t in sorted(buys, key=lambda x: x['value'], reverse=True)]
                            print(f"  Bought: {', '.join(buy_tickers)}")
                        if sells:
                            sell_tickers = [t['ticker'] for t in sorted(sells, key=lambda x: x['value'])]
                            print(f"  Sold: {', '.join(sell_tickers)}")
                else:
                    # Initial positions
                    if verbose:
                        long_pos = target_positions[target_positions['signal_zscore'] > 0]
                        short_pos = target_positions[target_positions['signal_zscore'] < 0]
                        print(f"{len(target_positions)} positions")
                        if len(long_pos) > 0:
                            print(f"  Long: {', '.join(long_pos.index.tolist())}")
                        if len(short_pos) > 0:
                            print(f"  Short: {', '.join(short_pos.index.tolist())}")
                
                # Update positions
                current_positions = target_positions.copy()
                current_positions['shares'] = (
                    current_positions['weight'] * portfolio_value / day_prices
                )
            
            # Calculate portfolio value
            if not current_positions.empty:
                # Get current prices for holdings
                holdings_prices = day_prices.reindex(current_positions.index)
                
                # Calculate position values
                position_values = current_positions['shares'] * holdings_prices
                portfolio_value = position_values.sum()
            
            # Record daily value
            daily_values.append({
                'date': date,
                'portfolio_value': portfolio_value,
                'num_positions': len(current_positions)
            })
            
            # Show weekly summary (on Fridays)
            if verbose and date.weekday() == 4:  # Friday
                week_return = (portfolio_value / self.initial_capital - 1) * 100
                print(f"  Week ending {date.strftime('%Y-%m-%d')}: ${portfolio_value:,.0f} ({week_return:+.2f}%)")
        
        # Convert to DataFrame
        results = pd.DataFrame(daily_values)
        results = results.set_index('date')
        
        # Calculate returns
        results['daily_return'] = results['portfolio_value'].pct_change()
        results['cumulative_return'] = (
            results['portfolio_value'] / self.initial_capital - 1
        )
        
        if verbose:
            print(f"\nCompleted: {len(results)} days, {rebalance_count} rebalances")
        
        return results
    
    def _execute_rebalance(self, current: pd.DataFrame, target: pd.DataFrame,
                          prices: pd.Series, portfolio_value: float,
                          verbose: bool = True) -> List[Dict]:
        """
        Execute rebalancing trades.
        
        Args:
            current: Current positions
            target: Target positions
            prices: Current prices
            portfolio_value: Total portfolio value
            verbose: Print trade details
            
        Returns:
            List of trade dictionaries
        """
        trades = []
        
        # Calculate changes
        all_tickers = current.index.union(target.index)
        
        for ticker in all_tickers:
            current_weight = current.loc[ticker, 'weight'] if ticker in current.index else 0
            target_weight = target.loc[ticker, 'weight'] if ticker in target.index else 0
            
            if ticker not in prices.index:
                continue
            
            change = target_weight - current_weight
            
            if abs(change) > 0.0001:  # Only trade if meaningful change
                trade_value = change * portfolio_value
                trades.append({
                    'ticker': ticker,
                    'change': change,
                    'value': trade_value,
                    'price': prices[ticker],
                    'cost': abs(trade_value) * self.transaction_cost
                })
        
        # Verbose output is now handled in the main run() method
        
        return trades
    
    def calculate_metrics(self, results: pd.DataFrame) -> Dict:
        """
        Calculate performance metrics.
        
        Args:
            results: Backtest results DataFrame
            
        Returns:
            Dictionary of performance metrics
        """
        returns = results['daily_return'].dropna()
        
        # Basic metrics
        total_return = results['cumulative_return'].iloc[-1]
        annual_return = (1 + total_return) ** (252 / len(results)) - 1
        
        # Volatility
        daily_vol = returns.std()
        annual_vol = daily_vol * np.sqrt(252)
        
        # Sharpe ratio (assuming 0% risk-free rate)
        sharpe = annual_return / annual_vol if annual_vol > 0 else 0
        
        # Drawdown
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max
        max_drawdown = drawdown.min()
        
        # Win rate
        win_rate = (returns > 0).sum() / len(returns)
        
        # Calmar ratio
        calmar = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        
        return {
            'total_return': total_return,
            'annual_return': annual_return,
            'annual_volatility': annual_vol,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_drawdown,
            'win_rate': win_rate,
            'calmar_ratio': calmar,
            'num_days': len(results),
            'final_value': results['portfolio_value'].iloc[-1]
        }
    
    def plot_performance(self, results: pd.DataFrame, 
                        benchmark: pd.DataFrame = None,
                        save_path: str = None):
        """
        Plot backtest performance.
        
        Args:
            results: Backtest results
            benchmark: Optional benchmark returns (e.g., S&P 500)
            save_path: Path to save plot (optional)
        """
        fig, axes = plt.subplots(3, 1, figsize=(12, 10))
        
        # Plot 1: Cumulative returns
        axes[0].plot(results.index, results['cumulative_return'] * 100, 
                    label='Strategy', linewidth=2)
        
        if benchmark is not None:
            axes[0].plot(benchmark.index, benchmark['cumulative_return'] * 100,
                        label='Benchmark', linewidth=2, alpha=0.7)
        
        axes[0].set_title('Cumulative Returns', fontsize=14, fontweight='bold')
        axes[0].set_ylabel('Return (%)')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Drawdown
        returns = results['daily_return'].dropna()
        cumulative = (1 + returns).cumprod()
        running_max = cumulative.expanding().max()
        drawdown = (cumulative - running_max) / running_max * 100
        
        axes[1].fill_between(drawdown.index, drawdown, 0, alpha=0.3, color='red')
        axes[1].plot(drawdown.index, drawdown, color='red', linewidth=1)
        axes[1].set_title('Drawdown', fontsize=14, fontweight='bold')
        axes[1].set_ylabel('Drawdown (%)')
        axes[1].grid(True, alpha=0.3)
        
        # Plot 3: Rolling Sharpe (60-day)
        rolling_returns = results['daily_return'].rolling(60)
        rolling_sharpe = (rolling_returns.mean() / rolling_returns.std()) * np.sqrt(252)
        
        axes[2].plot(rolling_sharpe.index, rolling_sharpe, linewidth=2)
        axes[2].axhline(y=0, color='black', linestyle='--', alpha=0.3)
        axes[2].set_title('Rolling 60-Day Sharpe Ratio', fontsize=14, fontweight='bold')
        axes[2].set_ylabel('Sharpe Ratio')
        axes[2].set_xlabel('Date')
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Plot saved to {save_path}")
        
        plt.show()
    
    def print_summary(self, metrics: Dict):
        """
        Print formatted performance summary.
        
        Args:
            metrics: Dictionary of performance metrics
        """
        print("\n" + "=" * 70)
        print("BACKTEST PERFORMANCE SUMMARY")
        print("=" * 70)
        print(f"Initial Capital:      ${self.initial_capital:,.0f}")
        print(f"Final Value:          ${metrics['final_value']:,.0f}")
        print(f"Total Return:         {metrics['total_return']*100:,.2f}%")
        print(f"Annual Return:        {metrics['annual_return']*100:,.2f}%")
        print(f"Annual Volatility:    {metrics['annual_volatility']*100:,.2f}%")
        print(f"Sharpe Ratio:         {metrics['sharpe_ratio']:.2f}")
        print(f"Calmar Ratio:         {metrics['calmar_ratio']:.2f}")
        print(f"Max Drawdown:         {metrics['max_drawdown']*100:,.2f}%")
        print(f"Win Rate:             {metrics['win_rate']*100:,.1f}%")
        print(f"Days Simulated:       {metrics['num_days']:,}")
        print("=" * 70)


if __name__ == '__main__':
    # Test backtest module
    from data import get_clean_sp500_data
    from signal_meanrev import MeanReversionSignal
    from portfolio import Portfolio
    from datetime import datetime, timedelta
    
    print("Running backtest test...\n")
    
    # Get 2 years of data
    start = (datetime.now() - timedelta(days=730)).strftime('%Y-%m-%d')
    data, tickers = get_clean_sp500_data(start_date=start)
    
    # Calculate signals
    signal_calc = MeanReversionSignal()
    signals = signal_calc.calculate(data)
    
    # Create portfolio manager
    portfolio_mgr = Portfolio()
    
    # Run backtest
    backtest = Backtest(initial_capital=1_000_000)
    results = backtest.run(data, signals, portfolio_mgr)
    
    # Calculate metrics
    metrics = backtest.calculate_metrics(results)
    
    # Print summary
    backtest.print_summary(metrics)
    
    # Plot results
    backtest.plot_performance(results)
