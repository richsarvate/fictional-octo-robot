# Mean Reversion Stock Picking Strategy

A simplified quantitative stock picking model inspired by Renaissance Technologies. Uses mean reversion signals to identify trading opportunities in the S&P 500.

## Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

### Get Live Recommendations

```bash
# Run strategy for current date
python run_strategy.py

# Specify portfolio value
python run_strategy.py --portfolio-value 100000
```

### Run Backtest

```bash
# Run historical backtest (default: 2 years)
python run_strategy.py --backtest

# Backtest longer period (10 years)
python run_strategy.py --backtest --days 3650
```

## Project Structure

```
Renaissance/
├── design_doc_simple_quant_model.md  # Strategy design document
├── requirements.txt                   # Python dependencies
├── data.py                           # Data fetching and cleaning
├── signal_meanrev.py                 # Mean reversion signal calculator
├── portfolio.py                      # Portfolio construction
├── backtest.py                       # Backtesting framework
└── run_strategy.py                   # Main script
```

## How It Works

### 1. Mean Reversion Signal
- Calculate 20-day moving average for each stock
- Measure price deviation from average
- Generate buy signals for oversold stocks (below average)
- Generate sell signals for overbought stocks (above average)

### 2. Portfolio Construction
- Equal weight base across ~400 liquid S&P 500 stocks
- Adjust position sizes by signal strength
- Apply constraints: 0.1% - 2% per stock
- Rebalance weekly

### 3. Expected Performance
- Annual Return: 10-14% (vs S&P 500 ~10%)
- Sharpe Ratio: 0.6-1.0
- Max Drawdown: ~12%
- Win Rate: 60-65%

## Usage Examples

### Weekly Trading Routine

Every Monday morning:
```bash
python run_strategy.py --portfolio-value 1000000
```

Review the output and execute trades in your broker.

### Test Individual Modules

```bash
# Test data fetching
python data.py

# Test signal calculation
python signal_meanrev.py

# Test portfolio construction
python portfolio.py

# Test backtesting
python backtest.py
```

## Requirements

- Python 3.9+
- Minimum recommended portfolio: $50,000-$100,000
- Internet connection for data fetching

## Strategy Details

See [design_doc_simple_quant_model.md](design_doc_simple_quant_model.md) for complete strategy documentation.

## Disclaimer

This is educational software for learning quantitative trading concepts. Not financial advice. Use at your own risk.

## Next Steps

Once mean reversion is working well, the strategy can be extended with:
- Momentum signal (trend following)
- Sector relative signal (pairs trading)  
- Value signal (fundamental ratios)
- Machine learning enhancements

But start simple. Get one thing working really well first.
