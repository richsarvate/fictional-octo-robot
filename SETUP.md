# Renaissance Trading System Setup Guide

This guide will help you set up the Renaissance trading system on a new machine.

## Prerequisites

- Python 3.8+
- Git
- Alpaca trading account (paper or live)

## Setup Steps

### 1. Clone the Repository

```bash
git clone https://github.com/richsarvate/fictional-octo-robot.git
cd fictional-octo-robot
```

### 2. Install Python Dependencies

```bash
pip install -r requirements.txt
```

### 3. Configure Alpaca API Credentials

Create a file `scripts/config.py` with your Alpaca API credentials:

```python
ALPACA_CONFIG = {
    'api_key': 'YOUR_API_KEY_HERE',
    'secret_key': 'YOUR_SECRET_KEY_HERE',
    'base_url': 'https://paper-api.alpaca.markets',  # or https://api.alpaca.markets for live
    'api_version': 'v2'
}
```

**IMPORTANT:** This file is gitignored and should NEVER be committed to the repository.

### 4. Download Initial Data

Run the data download script to populate the stock data cache:

```bash
python data/download.py
```

This will download S&P 500 stock data and cache it locally.

### 5. Test the Trader (Dry Run)

Test the trading system without executing any trades:

```bash
python scripts/alpaca_trader.py --dry-run
```

This will show you what trades would be executed without actually placing orders.

### 6. Set Up Automated Trading (Optional)

To run the trader automatically at market close (4:30 PM ET), set up a cron job:

```bash
# Open crontab editor
crontab -e

# Add this line (adjust path as needed):
30 16 * * 1-5 cd /home/ubuntu/GitHubProjects/fictional-octo-robot && /usr/bin/python3 scripts/alpaca_trader.py --paper >> logs/cron/rebalance_$(date +\%Y\%m\%d).log 2>&1
```

This will:
- Run at 4:30 PM ET on weekdays
- Execute on paper trading account
- Log output to `logs/cron/`

### 7. Run the Trader

Once you're confident everything is working:

**Paper Trading (recommended):**
```bash
python scripts/alpaca_trader.py --paper
```

**Live Trading (DANGEROUS - real money!):**
```bash
python scripts/alpaca_trader.py --live
```

You'll be prompted to type 'CONFIRM' when using live trading.

## File Structure

```
Renaissance/
├── data/               # Data downloading and processing
│   ├── download.py    # Download S&P 500 data
│   ├── data.py        # Data utilities
│   └── data_cache/    # Cached data (gitignored)
├── signals/           # Trading signal generators
│   ├── signal_value.py          # Value-based signals
│   ├── signal_momentum.py       # Momentum signals
│   ├── signal_meanrev.py        # Mean reversion signals
│   └── signal_sector_relative.py # Sector relative signals
├── strategies/        # Portfolio management
│   ├── portfolio.py   # Portfolio construction
│   └── backtest.py    # Backtesting framework
├── scripts/           # Executable scripts
│   ├── alpaca_trader.py  # Main trader (THIS IS WHAT YOU RUN)
│   ├── config.py         # API credentials (CREATE THIS - gitignored)
│   └── setup_cron.sh     # Cron setup helper
├── docs/              # Documentation
│   ├── alpaca_integration.md  # Trading system docs
│   └── design_doc_simple_quant_model.md
├── logs/              # Trade logs (gitignored)
├── requirements.txt   # Python dependencies
└── README.md         # Project overview
```

## Key Features

### Zombie Position Cleanup
The trader automatically detects and closes positions worth less than $10. For fractional positions that can't be closed automatically (e.g., sub-penny positions), it will provide clear guidance to close them manually via the Alpaca dashboard.

### Current Strategy
- **Value-based long/short equity**: Longs undervalued stocks, shorts overvalued stocks
- **10 long + 10 short positions** (configurable with `--top-n`)
- **Market-neutral** approach (equal long/short exposure)
- **Daily rebalancing** at 4:30 PM ET

### Error Handling
- Handles partial fills and insufficient borrow availability
- Skips orders that are too small to execute
- Retries with available quantity when possible
- Logs all trades for audit trail

## Monitoring

### Check Logs
```bash
# Latest trade log
ls -lt logs/trades_*.log | head -1 | xargs cat

# Latest cron log (if using automated trading)
ls -lt logs/cron/*.log | head -1 | xargs cat
```

### View Current Positions
The trader shows your current positions when run. You can also view them on the Alpaca dashboard:
- Paper: https://app.alpaca.markets/paper/dashboard/overview
- Live: https://app.alpaca.markets/live/dashboard/overview

## Troubleshooting

### "Missing Alpaca credentials" Error
Make sure you've created `scripts/config.py` with your API keys (see step 3).

### "No module named 'alpaca'" Error
Install the Alpaca Python SDK:
```bash
pip install alpaca-py
```

### Zombie Positions Not Closing
Some positions are too small to close programmatically (< 0.001 shares). Close these manually via the Alpaca dashboard.

### Orders Failing with "insufficient qty available"
This happens when trying to short a stock with limited borrow availability. The trader will automatically retry with available quantity or skip if too small.

## Important Notes

1. **Always test with paper trading first** before using real money
2. **The config.py file with API keys is gitignored** - never commit it
3. **Logs are gitignored** - they're for local debugging only
4. **Data cache is gitignored** - it will be regenerated on first run
5. **Review trades before going live** - run with `--dry-run` first

## Support

For issues or questions:
- Check the documentation in `docs/`
- Review recent logs in `logs/`
- Check Alpaca dashboard for position status

---

**Last Updated:** November 21, 2025
