# Simple Quant Stock Picker - Mean Reversion

## The Idea
Use math to pick stocks in the S&P 500. When stock prices deviate too far from their average, bet on them returning to normal.

Inspired by Renaissance Technologies' mean reversion strategies, but ultra-simplified.

## The Strategy

**Single Signal: Mean Reversion**

Buy when a stock's price drops below its 20-day average. Sell when it rises above.

```python
# Calculate 20-day moving average
price_ma20 = close.rolling(20).mean()

# Calculate deviation from average
deviation = (close - price_ma20) / price_ma20

# Smooth over 5 days and invert (negative deviation = buy signal)
signal = -deviation.rolling(5).mean()

# Normalize to z-score for position sizing
signal_zscore = (signal - signal.rolling(252).mean()) / signal.rolling(252).std()
```

**Why this works:**
- Markets overreact to short-term news
- Prices oscillate around fair value
- Mean reversion is one of the most reliable patterns in finance

## Portfolio Rules

- **Universe**: S&P 500, $1B+ market cap (~400 stocks)
- **Position Size**: 0.1% - 2% per stock (based on signal strength)
- **Rebalance**: Weekly (every Monday)
- **Limits**: Max 25% per sector, max 15% drawdown → stop trading

**Position Sizing:**
```python
base_weight = 1 / num_stocks  # Equal weight starting point
position = base_weight * (1 + signal_zscore / 2)  # Adjust by signal
position = clip(position, 0.001, 0.02)  # Apply 0.1% - 2% limits
```

## Expected Returns

- **Annual Return**: 10-14% (vs S&P 500 ~10%)
- **Sharpe Ratio**: 0.6-1.0
- **Max Loss**: ~12%
- **Win Rate**: 60-65% of trades

Note: Single signal = more conservative than multi-signal approaches

## Build Plan

1. Get daily price/volume data (Yahoo Finance)
2. Code mean reversion signal
3. Build portfolio logic with constraints
4. Backtest 2010-2025 + optimize parameters

## How to Use

**Weekly Routine (Every Monday):**
```bash
python run_strategy.py
```

**Script outputs:**
```
=== Portfolio Recommendations (2025-10-29) ===

BUY (Long Positions):
AAPL   1.8%  ($18,000)  z-score: +2.3 (strong oversold)
MSFT   1.5%  ($15,000)  z-score: +1.2 (oversold)
JPM    1.2%  ($12,000)  z-score: +0.8 (slight oversold)
... (~200 long positions)

SELL/SHORT (Short Positions):
TSLA   1.6%  ($16,000)  z-score: -2.1 (strong overbought)
NVDA   1.3%  ($13,000)  z-score: -1.4 (overbought)
... (~200 short positions)

---
Trades This Week: 47 adjustments
Estimated Cost: $235 (commissions)
Portfolio Turnover: 4.7%
```

**Execution Options:**
1. **Manual**: Copy trades into your broker (30 min/week)
2. **Automated**: Script places orders via broker API (fully hands-off)

**Minimum Account Size:** $50,000-$100,000 realistic minimum (need diversification across many stocks)

**Time Commitment:** 
- Manual: ~30 min every Monday
- Automated: Set it and forget it


**Paper Trading:** 
add Alpaca integration
Token: d59a15de-0f0e-4a44-9c8f-3e9ba52c488f