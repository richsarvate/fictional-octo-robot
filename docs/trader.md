# trader.py Design Document

## Purpose
Simple on-demand script to generate trading recommendations using the Value signal strategy.

## Behavior
1. Downloads fresh S&P 500 data (90 days for signal calculations)
2. Calculates Value signals based on P/E ratios vs sector averages
3. Generates Top 10 Long + Top 10 Short recommendations
4. Checks if market is open (using pytz timezone handling)
5. Outputs actionable portfolio recommendations with position sizes

## Key Functions
- Uses ValueSignal.calculate() for P/E-based signals
- Portfolio.calculate_positions() for position sizing
- get_clean_sp500_data() for fresh data download
- Market hours check before generating recommendations

## Output
- Clear list of Top 10 Longs and Top 10 Shorts
- Position weights and dollar amounts for $1M portfolio
- Z-scores showing signal strength
- Ready for manual execution or Alpaca API integration

## Architecture
- Shares core modules with run_strategy.py and alpaca_trader.py
- All scripts use signals/ and strategies/ modules for calculations
- Difference is output only: trader.py displays, alpaca_trader.py executes
