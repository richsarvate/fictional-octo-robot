#!/bin/bash
# Setup cron job for daily rebalancing

# Get the absolute path to the project
PROJECT_DIR="/home/ubuntu/GitHubProjects/Renaissance"

# Create logs directory
mkdir -p "$PROJECT_DIR/logs/cron"

# Create the cron job entry
# Runs daily at 4:30 PM ET (after market close at 4:00 PM ET)
CRON_JOB="30 16 * * 1-5 cd $PROJECT_DIR && /usr/bin/python3 scripts/alpaca_trader.py --paper >> logs/cron/rebalance_\$(date +\%Y\%m\%d).log 2>&1"

# Check if cron job already exists
if crontab -l 2>/dev/null | grep -q "alpaca_trader.py"; then
    echo "Cron job already exists. Removing old one..."
    crontab -l 2>/dev/null | grep -v "alpaca_trader.py" | crontab -
fi

# Add the new cron job
(crontab -l 2>/dev/null; echo "$CRON_JOB") | crontab -

echo "✓ Cron job installed!"
echo ""
echo "Schedule: Monday-Friday at 4:30 PM ET (after market close)"
echo "Command: cd $PROJECT_DIR && python3 scripts/alpaca_trader.py --paper"
echo "Logs: $PROJECT_DIR/logs/cron/rebalance_YYYYMMDD.log"
echo ""
echo "To view current cron jobs:"
echo "  crontab -l"
echo ""
echo "To remove this cron job:"
echo "  crontab -l | grep -v 'alpaca_trader.py' | crontab -"
