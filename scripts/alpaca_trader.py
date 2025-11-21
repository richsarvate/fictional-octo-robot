#!/usr/bin/env python3
"""
Alpaca integration - Execute Value strategy trades via Alpaca API.

Usage:
    python scripts/alpaca_trader.py --dry-run      # Preview trades without executing
    python scripts/alpaca_trader.py --paper        # Execute on paper account
    python scripts/alpaca_trader.py --live         # Execute on live account (DANGEROUS!)

Environment variables:
    ALPACA_API_KEY_ID     - Your Alpaca API key
    ALPACA_SECRET_KEY     - Your Alpaca secret key
"""
import sys
import os
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from data.data import get_clean_sp500_data
from signals.signal_value import ValueSignal
from strategies.portfolio import Portfolio

# Import Alpaca config
try:
    from scripts.config import ALPACA_CONFIG
    print(f"[INFO] Loaded Alpaca config from scripts/config.py")
except ImportError:
    ALPACA_CONFIG = None
    print("[INFO] No config.py found, will use environment variables")


class AlpacaTrader:
    """Alpaca API integration for automated trading."""
    
    def __init__(self, paper: bool = True, dry_run: bool = False):
        """
        Initialize Alpaca trader.
        
        Args:
            paper: Use paper trading account (default: True)
            dry_run: Print orders without executing (default: False)
        """
        self.paper = paper
        self.dry_run = dry_run
        self.api = None
        
        # Always connect to Alpaca (even in dry-run) to fetch real positions
        try:
            from alpaca.trading.client import TradingClient
            from alpaca.trading.enums import OrderSide, TimeInForce
            from alpaca.trading.requests import MarketOrderRequest
            
            # Get API credentials: 1) from config.py, 2) from environment
            api_key = os.getenv('ALPACA_API_KEY_ID')
            secret_key = os.getenv('ALPACA_SECRET_KEY')
            
            # Try loading from config.py if not in environment
            if not api_key and ALPACA_CONFIG:
                api_key = ALPACA_CONFIG.get('api_key')
                secret_key = ALPACA_CONFIG.get('secret_key')
                print("Loaded credentials from config.py")
            
            if not api_key or not secret_key:
                raise ValueError(
                    "Missing Alpaca credentials. Set ALPACA_API_KEY_ID and ALPACA_SECRET_KEY "
                    "environment variables or add to scripts/config.py"
                )
            
            self.api = TradingClient(api_key, secret_key, paper=paper)
            self.OrderSide = OrderSide
            self.TimeInForce = TimeInForce
            self.MarketOrderRequest = MarketOrderRequest
            
        except ImportError:
            raise ImportError("alpaca-py not installed. Run: pip install alpaca-py")
    
    def get_current_positions(self) -> Dict[str, dict]:
        """
        Get current positions from Alpaca account.
        
        Returns:
            Dict mapping ticker -> {qty: float, market_value: float, side: str}
        """
        positions = {}
        for pos in self.api.get_all_positions():
            positions[pos.symbol] = {
                'qty': float(pos.qty),
                'market_value': float(pos.market_value),
                'side': 'long' if float(pos.qty) > 0 else 'short'
            }
        return positions
    
    def get_account_info(self) -> dict:
        """Get account info including buying power."""
        account = self.api.get_account()
        return {
            'buying_power': float(account.buying_power),
            'portfolio_value': float(account.portfolio_value)
        }
    
    def calculate_rebalance_orders(
        self, 
        current_positions: Dict[str, dict],
        target_positions: Dict[str, float],
        portfolio_value: float
    ) -> List[dict]:
        """
        Calculate orders needed to rebalance from current to target portfolio.
        
        Args:
            current_positions: Current positions {ticker: {qty, market_value, side}}
            target_positions: Target positions {ticker: weight (0-1)}
            portfolio_value: Total portfolio value
            
        Returns:
            List of order dicts: [{ticker, side, dollars, action, reason}]
        """
        orders = []
        
        # Close positions not in target
        for ticker in current_positions:
            if ticker not in target_positions:
                orders.append({
                    'ticker': ticker,
                    'side': 'sell' if current_positions[ticker]['side'] == 'long' else 'buy',
                    'dollars': abs(current_positions[ticker]['market_value']),
                    'action': 'CLOSE',
                    'reason': 'Not in target portfolio'
                })
        
        # Open or adjust positions in target
        for ticker, target_weight in target_positions.items():
            target_dollars = target_weight * portfolio_value
            
            if ticker in current_positions:
                current_dollars = current_positions[ticker]['market_value']
                diff = target_dollars - current_dollars
                
                if abs(diff) > 100:  # Only rebalance if > $100 difference
                    orders.append({
                        'ticker': ticker,
                        'side': 'buy' if diff > 0 else 'sell',
                        'dollars': abs(diff),
                        'action': 'ADJUST',
                        'reason': f'Rebalance: ${current_dollars:,.0f} -> ${target_dollars:,.0f}'
                    })
            else:
                # New position
                # For shorts, we sell first (borrow and sell)
                side = 'buy' if target_weight > 0 else 'sell'
                orders.append({
                    'ticker': ticker,
                    'side': side,
                    'dollars': abs(target_dollars),
                    'action': 'OPEN',
                    'reason': f'New {"long" if side == "buy" else "short"}: ${abs(target_dollars):,.0f}'
                })
        
        return orders
    
    def _get_stock_price(self, ticker: str) -> float:
        """
        Get current stock price for position sizing.
        
        Args:
            ticker: Stock symbol
            
        Returns:
            Current price (ask for buys, bid for sells)
        """
        from alpaca.data.historical import StockHistoricalDataClient
        from alpaca.data.requests import StockLatestQuoteRequest
        
        # Get API credentials
        api_key = os.getenv('ALPACA_API_KEY_ID')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        if not api_key and ALPACA_CONFIG:
            api_key = ALPACA_CONFIG.get('api_key')
            secret_key = ALPACA_CONFIG.get('secret_key')
        
        # Initialize data client with auth
        data_client = StockHistoricalDataClient(api_key, secret_key)
        quote_request = StockLatestQuoteRequest(symbol_or_symbols=ticker)
        quote = data_client.get_stock_latest_quote(quote_request)[ticker]
        
        # Try ask price, fallback to bid, then mid price
        price = float(quote.ask_price)
        if price == 0:
            price = float(quote.bid_price)
        if price == 0:
            price = (float(quote.ask_price) + float(quote.bid_price)) / 2
        
        return price
    
    def execute_trades(self, orders: List[dict], dry_run: bool = None) -> List[dict]:
        """
        Execute trades via Alpaca API.
        
        Args:
            orders: List of order dicts from calculate_rebalance_orders()
            dry_run: Override instance dry_run setting
            
        Returns:
            List of execution results
        """
        if dry_run is None:
            dry_run = self.dry_run
        
        if dry_run:
            print("\n[DRY RUN] Would execute these orders:")
            print("-" * 70)
            for order in orders:
                print(f"{order['action']:6} {order['ticker']:6} {order['side']:4} ${order['dollars']:>10,.0f}  ({order['reason']})")
            print("-" * 70)
            return []
        
        results = []
        for order in orders:
            try:
                # Create market order
                side = self.OrderSide.BUY if order['side'] == 'buy' else self.OrderSide.SELL
                
                # Determine if this is a short order (selling for OPEN/ADJUST)
                is_short_order = (order['side'] == 'sell' and 
                                 order['action'] in ['OPEN', 'ADJUST'] and
                                 order['dollars'] > 0)  # Not a close
                
                if is_short_order:
                    # For short positions, MUST use qty (whole shares) not notional
                    # Alpaca doesn't support fractional shares for shorts
                    
                    # Get current price
                    price = self._get_stock_price(order['ticker'])
                    
                    # Check for zero/invalid price
                    if price == 0:
                        raise ValueError(f"No valid quote available for {order['ticker']}")
                    
                    # Calculate whole shares
                    qty = int(order['dollars'] / price)
                    
                    # Skip orders that round to 0 shares
                    if qty == 0:
                        print(f"  ⊘ Skipping {order['ticker']}: adjustment too small (${order['dollars']:.2f} = 0 shares at ${price:.2f}/share)")
                        results.append({
                            'ticker': order['ticker'],
                            'status': 'skipped',
                            'reason': 'qty_too_small',
                            'side': order['side'],
                            'dollars': order['dollars']
                        })
                        continue
                    
                    order_request = self.MarketOrderRequest(
                        symbol=order['ticker'],
                        qty=qty,
                        side=side,
                        time_in_force=self.TimeInForce.DAY
                    )
                else:
                    # For longs and closes, use notional (dollar-based)
                    notional_rounded = round(order['dollars'], 2)
                    
                    # Skip orders with invalid or near-zero notional
                    if notional_rounded <= 1.0:
                        print(f"  ⊘ Skipping {order['ticker']}: notional too small (${notional_rounded:.2f})")
                        results.append({
                            'ticker': order['ticker'],
                            'status': 'skipped',
                            'reason': 'notional_too_small',
                            'side': order['side'],
                            'dollars': order['dollars']
                        })
                        continue
                    
                    order_request = self.MarketOrderRequest(
                        symbol=order['ticker'],
                        notional=notional_rounded,
                        side=side,
                        time_in_force=self.TimeInForce.DAY
                    )
                
                submitted = self.api.submit_order(order_request)
                results.append({
                    'ticker': order['ticker'],
                    'status': 'submitted',
                    'order_id': submitted.id,
                    'side': order['side'],
                    'dollars': order['dollars']
                })
                print(f"✓ Submitted: {order['action']} {order['ticker']} {order['side']} ${order['dollars']:,.0f}")
                
            except Exception as e:
                error_str = str(e)
                
                # Handle insufficient borrow availability
                if "insufficient qty available" in error_str:
                    try:
                        # Extract available qty from error message
                        # Format: "insufficient qty available for order (requested: X, available: Y)"
                        import re
                        match = re.search(r'available:\s*([\d.]+)', error_str)
                        if match:
                            available_qty = float(match.group(1))
                            available_qty_int = int(available_qty)  # Round down to whole shares
                            
                            # Skip if available < 1 share
                            if available_qty_int < 1:
                                print(f"  ⊘ Skipping {order['ticker']}: insufficient borrow (< 1 share available)")
                                results.append({
                                    'ticker': order['ticker'],
                                    'status': 'skipped',
                                    'reason': 'insufficient_borrow',
                                    'side': order['side'],
                                    'dollars': order['dollars']
                                })
                                continue
                            
                            # Get target qty
                            price = self._get_stock_price(order['ticker'])
                            target_qty = int(order['dollars'] / price)
                            
                            # Retry if we can get at least 50% of target
                            if available_qty_int >= target_qty * 0.5 and available_qty_int > 0:
                                print(f"  ↻ Retrying {order['ticker']} with {available_qty_int} shares (available) vs {target_qty} (requested)")
                                
                                retry_order = self.MarketOrderRequest(
                                    symbol=order['ticker'],
                                    qty=available_qty_int,
                                    side=side,
                                    time_in_force=self.TimeInForce.DAY
                                )
                                
                                submitted = self.api.submit_order(retry_order)
                                results.append({
                                    'ticker': order['ticker'],
                                    'status': 'submitted_partial',
                                    'order_id': submitted.id,
                                    'side': order['side'],
                                    'dollars': available_qty_int * price,
                                    'note': f'Partial fill: {available_qty_int}/{target_qty} shares'
                                })
                                print(f"✓ Submitted (partial): {order['action']} {order['ticker']} {available_qty_int} shares")
                                continue
                            else:
                                print(f"  ⊘ Skipping {order['ticker']}: only {available_qty_int} shares available (< 50% of {target_qty})")
                    except Exception as retry_error:
                        print(f"  ⚠ Retry failed for {order['ticker']}: {retry_error}")
                
                # Log failure
                results.append({
                    'ticker': order['ticker'],
                    'status': 'failed',
                    'error': error_str,
                    'side': order['side'],
                    'dollars': order['dollars']
                })
                print(f"✗ Failed: {order['ticker']} - {error_str}")
        
        return results
    
    def log_trades(self, orders: List[dict], results: List[dict], filename: str = None):
        """Log trades to file."""
        if filename is None:
            filename = f"trades_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        
        log_dir = Path('./logs')
        log_dir.mkdir(exist_ok=True)
        log_path = log_dir / filename
        
        with open(log_path, 'w') as f:
            f.write(f"Alpaca Trade Log - {datetime.now()}\n")
            f.write("=" * 70 + "\n\n")
            
            f.write("ORDERS:\n")
            for order in orders:
                f.write(f"  {order['action']:6} {order['ticker']:6} {order['side']:4} ${order['dollars']:>10,.0f}\n")
                f.write(f"    Reason: {order['reason']}\n")
            
            f.write("\n" + "=" * 70 + "\n\n")
            
            if results:
                f.write("RESULTS:\n")
                for result in results:
                    status = result['status']
                    f.write(f"  {result['ticker']:6} - {status}\n")
                    if status == 'submitted':
                        f.write(f"    Order ID: {result['order_id']}\n")
                    elif status == 'failed':
                        f.write(f"    Error: {result['error']}\n")
        
        print(f"\nTrade log saved: {log_path}")


def main():
    parser = argparse.ArgumentParser(description='Alpaca automated trader')
    parser.add_argument('--dry-run', action='store_true', help='Preview trades without executing')
    parser.add_argument('--paper', action='store_true', default=True, help='Use paper trading account (default)')
    parser.add_argument('--live', action='store_true', help='Use LIVE trading account (DANGEROUS!)')
    parser.add_argument('--top-n', type=int, default=10, help='Number of longs and shorts (default: 10)')
    args = parser.parse_args()
    
    # Safety check
    if args.live and not args.dry_run:
        confirm = input("⚠️  WARNING: You are about to trade on a LIVE account. Type 'CONFIRM' to proceed: ")
        if confirm != 'CONFIRM':
            print("Aborted.")
            return
    
    paper = not args.live
    trader = AlpacaTrader(paper=paper, dry_run=args.dry_run)
    
    print("=" * 70)
    print("ALPACA TRADER - Value Strategy")
    print("=" * 70)
    print(f"Mode: {'DRY RUN' if args.dry_run else ('PAPER' if paper else 'LIVE')}")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # Get account info
    account = trader.get_account_info()
    print(f"Portfolio Value: ${account['portfolio_value']:,.0f}")
    print(f"Buying Power: ${account['buying_power']:,.0f}")
    
    # Get current positions
    print("\nFetching current positions...")
    current_positions = trader.get_current_positions()
    print(f"Current positions: {len(current_positions)}")
    if current_positions:
        for ticker, pos in list(current_positions.items())[:5]:
            print(f"  {ticker}: ${pos['market_value']:,.0f} ({pos['side']})")
        if len(current_positions) > 5:
            print(f"  ... and {len(current_positions) - 5} more")
    
    # Calculate target portfolio
    print("\nCalculating target portfolio...")
    print("Loading data (90 days)...")
    from datetime import timedelta
    end_date = datetime.now().strftime('%Y-%m-%d')
    start_date = (datetime.now() - timedelta(days=90)).strftime('%Y-%m-%d')
    data, tickers = get_clean_sp500_data(start_date=start_date, end_date=end_date)
    
    print("Calculating Value signals...")
    signal_calc = ValueSignal()
    signals = signal_calc.calculate(data)
    current_signals = signal_calc.get_current_signals(signals)
    
    print(f"Building portfolio (Top {args.top_n} longs + {args.top_n} shorts)...")
    portfolio = Portfolio(top_n_long=args.top_n, top_n_short=args.top_n)
    target_positions_df = portfolio.calculate_positions(current_signals)
    
    # Convert to dict {ticker: weight}
    # Note: Portfolio assigns positive weights for both longs and shorts
    # We need to check signal_zscore to determine if it's actually a short
    target_positions = {}
    for ticker, row in target_positions_df.iterrows():
        weight = row['weight']
        # If signal is negative, it's a short position (make weight negative)
        if row['signal_zscore'] < 0:
            weight = -weight
        target_positions[ticker] = weight
    
    print(f"Target positions: {len(target_positions)}")
    longs = {t: w for t, w in target_positions.items() if w > 0}
    shorts = {t: w for t, w in target_positions.items() if w < 0}
    print(f"  Longs: {len(longs)}, total weight: {sum(longs.values())*100:.1f}%")
    print(f"  Shorts: {len(shorts)}, total weight: {sum(abs(w) for w in shorts.values())*100:.1f}%")
    
    # Calculate rebalance orders
    print("\nCalculating rebalance orders...")
    orders = trader.calculate_rebalance_orders(
        current_positions,
        target_positions,
        account['portfolio_value']
    )
    
    if not orders:
        print("No rebalancing needed. Portfolio is up to date.")
        return
    
    print(f"\nRebalance summary: {len(orders)} orders")
    
    # Group by action
    opens = [o for o in orders if o['action'] == 'OPEN']
    closes = [o for o in orders if o['action'] == 'CLOSE']
    adjusts = [o for o in orders if o['action'] == 'ADJUST']
    
    if opens:
        print(f"  Open: {len(opens)} new positions")
    if closes:
        print(f"  Close: {len(closes)} positions")
    if adjusts:
        print(f"  Adjust: {len(adjusts)} positions")
    
    # Execute trades
    results = trader.execute_trades(orders)
    
    # Log
    trader.log_trades(orders, results)
    
    # Print final portfolio summary
    print("\n" + "=" * 70)
    print("Final Portfolio Summary")
    print("=" * 70)
    
    final_account = trader.get_account_info()
    final_positions = trader.get_current_positions()
    
    print(f"Portfolio Value: ${final_account['portfolio_value']:,.2f}")
    print(f"Buying Power: ${final_account['buying_power']:,.2f}")
    print(f"Total Positions: {len(final_positions)}")
    
    # Breakdown by long/short
    longs = {t: p for t, p in final_positions.items() if p['side'] == 'long' and abs(p['market_value']) > 10}
    shorts = {t: p for t, p in final_positions.items() if p['side'] == 'short'}
    
    long_value = sum(p['market_value'] for p in longs.values())
    short_value = sum(abs(p['market_value']) for p in shorts.values())
    
    print(f"Long Positions: {len(longs)} (${long_value:,.2f})")
    print(f"Short Positions: {len(shorts)} (${short_value:,.2f})")
    print(f"Total Deployed: ${long_value + short_value:,.2f}")
    
    # Cleanup zombie positions (< $10 value)
    if not args.dry_run:
        zombies = {t: p for t, p in final_positions.items() if abs(p['market_value']) < 10}
        if zombies:
            print(f"\n🧹 Cleaning up {len(zombies)} zombie position(s):")
            for ticker, pos in zombies.items():
                try:
                    # Determine order side (sell longs, buy shorts)
                    side = trader.OrderSide.SELL if pos['side'] == 'long' else trader.OrderSide.BUY
                    qty = abs(pos['qty'])
                    
                    # Skip if quantity rounds to 0 (sub-penny positions)
                    if qty < 0.001:
                        print(f"  ⊘ Skipping {ticker}: position too small ({qty:.6f} shares, ${abs(pos['market_value']):.4f})")
                        print(f"    → Manual intervention needed: close via Alpaca dashboard")
                        continue
                    
                    # For fractional shares, round to avoid API errors
                    # Alpaca supports fractional shares for longs but not shorts
                    if pos['side'] == 'short':
                        # Shorts require whole shares - round up to ensure full close
                        qty = int(qty) + (1 if qty % 1 > 0 else 0)
                        if qty == 0:
                            print(f"  ⊘ Skipping {ticker}: fractional short position ({abs(pos['qty']):.6f} shares)")
                            print(f"    → Manual intervention needed: close via Alpaca dashboard")
                            continue
                    
                    # Submit close order
                    order_request = trader.MarketOrderRequest(
                        symbol=ticker,
                        qty=qty,
                        side=side,
                        time_in_force=trader.TimeInForce.DAY
                    )
                    
                    submitted = trader.api.submit_order(order_request)
                    print(f"  ✓ Closed {ticker}: {qty:.4f} shares (${abs(pos['market_value']):.2f})")
                    
                except Exception as e:
                    print(f"  ✗ Failed to close {ticker}: {e}")
    
    print("\n" + "=" * 70)
    print("Done.")


if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
    except Exception as e:
        print(f"\n✗ Error: {e}")
        raise
