# Renaissance - Quantitative Stock Trading Strategy

A multi-signal quantitative trading system for S&P 500 stocks using mean reversion, momentum, sector relative, and value strategies.

## Project Structure

```
Renaissance/
├── run_strategy.py          # Main entry point
├── requirements.txt         # Python dependencies
│
├── signals/                 # Signal calculation modules
│   ├── signal_meanrev.py    # Mean reversion signal
│   ├── signal_momentum.py   # Momentum signal
│   ├── signal_sector_relative.py  # Sector relative strength
│   ├── signal_value.py      # Value (P/E ratio) signal
│   └── signal_combined.py   # Combined multi-signal strategy
│
├── strategies/              # Backtesting and portfolio management
│   ├── backtest.py          # Backtesting engine
│   └── portfolio.py         # Portfolio construction and rebalancing
│
├── data/                    # Data management
│   ├── data.py              # Data loading and cleaning
│   ├── download.py          # Data download script
│   ├── stock_data/          # Cached stock price data
│   └── data_cache/          # Cached sector and P/E data
│
├── docs/                    # Documentation
│   ├── README.md            # Project documentation
│   ├── analytics.md         # Backtest performance results
│   └── design_doc_simple_quant_model.md  # Design specifications
│
└── tests/                   # Unit tests (future)

```

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Download Data (First Time Only)
```bash
python3 data/download.py
```

### 3. Run a Backtest
```bash
# Value strategy on recent month
python3 run_strategy.py --backtest --strategy value --top-n-long 10 --top-n-short 10 --days 30

# Combined strategy with custom weights
python3 run_strategy.py --backtest --strategy combined --top-n-long 10 --top-n-short 10 \
    --meanrev-weight 0.2 --momentum-weight 0.1 --sector-weight 0.2 --value-weight 0.5 \
    --start-date 2025-09-01 --end-date 2025-10-01

# Random month test
python3 run_strategy.py --backtest --strategy value --random-month --top-n-long 10 --top-n-short 10
```

## Available Strategies

| Strategy | Description |
|----------|-------------|
| `meanrev` | Mean reversion - buy oversold, sell overbought |
| `momentum` | Momentum - buy recent winners, sell recent losers |
| `sector` | Sector relative - buy sector leaders, sell sector laggards |
| `value` | Value - buy low P/E vs sector, sell high P/E |
| `combined` | Weighted combination of all 4 signals |

## Performance Summary

See [analytics.md](docs/analytics.md) for detailed backtest results across multiple time periods.

**Best Strategies:**
- **Value**: Consistent winner across bull and bear markets (+3-4% monthly returns)
- **Combined (Val:50% MR:20% Sec:20% Mom:10%)**: Best risk-adjusted combined strategy

## Command Line Options

```
--backtest              Run historical backtest (vs live recommendations)
--strategy STRATEGY     Signal strategy: meanrev, momentum, sector, value, combined
--days DAYS             Number of days for backtest (default: 30)
--start-date DATE       Start date YYYY-MM-DD (overrides --days)
--end-date DATE         End date YYYY-MM-DD (default: today)
--top-n-long N          Only buy top N long positions
--top-n-short N         Only short top N short positions
--portfolio-value VAL   Portfolio value in dollars (default: 1,000,000)
--random-month          Pick a random month from available data

# Combined strategy weights:
--meanrev-weight W      Mean reversion weight (default: 0.5)
--momentum-weight W     Momentum weight (default: 0.5)
--sector-weight W       Sector relative weight (default: 0.0)
--value-weight W        Value weight (default: 0.0)
```

## Signal Details

### Mean Reversion
- 20-day return vs 120-day moving average
- Z-score normalized
- **Best for**: Choppy/sideways markets

### Momentum  
- 10-day, 20-day, 60-day returns
- Z-score normalized
- **Best for**: Trending markets

### Sector Relative
- 20-day relative performance vs sector peers
- Z-score normalized
- **Best for**: Sector rotation strategies

### Value
- P/E ratio vs sector average
- Inverted (low P/E = buy signal)
- **Best for**: Long-term value investing

## Development

### Adding New Signals
1. Create new file in `signals/` directory
2. Implement `calculate()` method returning MultiIndex DataFrame
3. Add signal quality evaluation function
4. Update `signal_combined.py` if integrating into combined strategy

### Running Tests
```bash
python3 -m pytest tests/
```

## License

MIT License - See LICENSE file

## Contributing

Pull requests welcome! Please add tests for new features.
