# Alpaca Integration Design

## Purpose
Execute Value strategy trades automatically via Alpaca API.

## Architecture
- Shares core modules (signals/, strategies/) with trader.py and run_strategy.py
- Does NOT call trader.py script - imports ValueSignal and Portfolio directly
- Uses same calculation logic, adds Alpaca API execution layer

## Behavior
1. Calculate portfolio recommendations using ValueSignal module
2. Connect to Alpaca API (paper or live)
3. Get current account positions and cash balance
4. Calculate trades needed:
   - Close positions not in new portfolio
   - Open/adjust positions to match target allocations
5. Submit market orders for all trades
6. Wait for fills and log results

## Key Functions
- `get_current_positions()` - Fetch Alpaca portfolio
- `calculate_rebalance_orders()` - Diff current vs target portfolio
- `execute_trades()` - Submit orders to Alpaca
- `wait_for_fills()` - Monitor order execution

## Safety Features
- Dry-run mode (print orders without executing)
- Position size limits (max 10% per stock)
- Verify account buying power before trading
- Log all trades to file with timestamp