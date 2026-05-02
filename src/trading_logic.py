import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_fetch
from datetime import datetime, timedelta

def generate_trading_signals(
    predictions: pd.DataFrame,
    threshold: float = 0.002,
    allow_short: bool = True
) -> pd.DataFrame:
    
    signals = predictions.copy()
    
    if "target_a_close" in signals.columns:
        pred_col = "target_a_close"
    elif "target_b_close" in signals.columns:
        pred_col = "target_b_close"
    else:
        raise ValueError("Could not find predicted return column")
    
    signals['signal'] = 0

    signals.loc[signals[pred_col] > threshold, 'signal'] = 1
    if allow_short:
        signals.loc[signals[pred_col] < -threshold, 'signal'] = -1

    signals['predicted_return'] = signals[pred_col]
    signals['direction'] = np.where(signals['signal'] == 1, 'LONG',
                          np.where(signals['signal'] == -1, 'SHORT', 'FLAT'))

    return signals

def backtest_strategy(
    signals: pd.DataFrame,
    price_data: pd.DataFrame,
    ticker: str = 'CL=F',
    multiplier: int = 1000,
    commission_per_contract: float = 2.5,
    initial_capital: float = 100_000,
    stop_loss_pct: float = 0.10,
    risk_per_trade: float = 0.01
):
    # print(f"Backtesting {ticker} with {stop_loss_pct*100}% stop-loss...")

    prices = price_data[price_data['Ticker'] == ticker].copy()
    if prices.empty:
        raise ValueError(f"No data for {ticker}")

    prices = prices[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    prices['timestamp'] = pd.to_datetime(prices['timestamp']).dt.tz_localize(None)
    prices = prices.set_index('timestamp').sort_index()

    df = signals.copy()
    if 'timestamp' not in df.columns:
        df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

    df = df.merge(prices, on='timestamp', how='left')
    df = df.dropna(subset=['Close']).reset_index(drop=True)

    capital = float(initial_capital)
    position = 0                    # net contracts (positive = long, negative = short)
    entry_prices = []               # List to track all entry prices for averaging
    equity_curve = []
    trade_log = []
    total_commissions = 0.0
    count_stop_loss=0

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row['Close'])
        signal = int(row.get('signal', 0))

        realized_pnl = 0.0
        commission = 0.0

        # === Stop Loss using Average Entry Price ===
        if position != 0 and len(entry_prices) > 0:
            avg_entry = sum(entry_prices) / len(entry_prices)
            stop_price = avg_entry * (1 - stop_loss_pct) if position > 0 else avg_entry * (1 + stop_loss_pct)
            
            hit_stop = (position > 0 and price <= stop_price) or (position < 0 and price >= stop_price)
            
            if hit_stop:
                count_stop_loss += 1
                realized_pnl = (price - avg_entry) * position * multiplier
                commission = commission_per_contract * abs(position)
                capital += realized_pnl - commission
                position = 0
                entry_prices = []
                trade_log.append({'timestamp': row['timestamp'], 'action': 'Stop Loss', 'price': price, 'pnl': realized_pnl})

        # Close on opposite or flat signal
        if position != 0 and ((signal == 0) or (signal != np.sign(position))):
            avg_entry = sum(entry_prices) / len(entry_prices) if entry_prices else 0
            realized_pnl = (price - avg_entry) * position * multiplier
            commission = commission_per_contract * abs(position)
            capital += realized_pnl - commission
            position = 0
            entry_prices = []

        # Open / Add to position with capital-aware sizing
        if position == 0 and signal != 0:
            risk_amount = capital * risk_per_trade
            stop_distance = price * stop_loss_pct
            contracts = int(risk_amount / (stop_distance * multiplier))
            contracts = max(1, min(contracts, 20))

            position = contracts if signal == 1 else -contracts
            entry_prices = [price] * abs(contracts)   # All contracts at this price
            commission = commission_per_contract * abs(position)
            capital -= commission

            action = 'Buy Long' if signal == 1 else 'Sell Short'
            trade_log.append({'timestamp': row['timestamp'], 'action': action, 'price': price, 'size': abs(position)})

        # Calculate equity
        avg_entry = sum(entry_prices) / len(entry_prices) if entry_prices else 0
        unrealized = (price - avg_entry) * position * multiplier if position != 0 else 0.0
        current_equity = capital + unrealized

        equity_curve.append({
            'timestamp': row['timestamp'],
            'equity': current_equity,
            'realized_pnl': realized_pnl,
            'position': position,
            'signal': signal,
            'close': price,
            'avg_entry': avg_entry
        })

        total_commissions += commission

    result = pd.DataFrame(equity_curve)
    result['cum_return'] = result['equity'] / initial_capital - 1
    result['peak'] = result['equity'].cummax()
    result['drawdown_pct'] = (result['equity'] - result['peak']) / result['peak']


    return result, pd.DataFrame(trade_log)

def calculate_performance_metrics(result: pd.DataFrame, initial_capital: float = 100000) -> dict:
    """
    Calculate key risk-adjusted metrics from backtest results.
    """
    final_equity = result['equity'].iloc[-1]
    total_return = (final_equity / initial_capital) - 1
    max_dd = result['drawdown_pct'].min()
    
    # Risk-adjusted ratios
    return_to_dd = total_return / abs(max_dd) if max_dd != 0 else 0
    adjusted_return_to_dd = total_return / abs(max_dd) if max_dd != 0 else 0
    
    # Sharpe Ratio (annualized, assuming 252 trading days)
    daily_returns = result['equity'].pct_change().dropna()
    sharpe = (daily_returns.mean() / daily_returns.std()) * np.sqrt(252) if daily_returns.std() != 0 else 0
    
    metrics = {
        'Total Return (%)': total_return * 100,
        'Max Drawdown (%)': max_dd * 100,
        'Return / Drawdown Ratio': return_to_dd,
        'Sharpe Ratio': sharpe,
        'Final Equity': final_equity,
        'Number of Trades': len(result),
        'Score': (total_return / abs(max_dd)) * sharpe * 0.7
    }


    if metrics['Score']>1.9 and max_dd>-0.5:
    
        print("\n" + "="*60)
        print("PERFORMANCE METRICS")
        print(f"Total Return          : {metrics['Total Return (%)']:.2f}%")
        print(f"Max Drawdown          : {metrics['Max Drawdown (%)']:.2f}%")
        print(f"Return / DD Ratio     : {metrics['Return / Drawdown Ratio']:.2f}")
        print(f"Sharpe Ratio          : {metrics['Sharpe Ratio']:.2f}")
        print(f"Final Equity          : ${metrics['Final Equity']:,.2f}")
        print(f"Number of Trades      : {metrics['Number of Trades']}")
        print(f"Score                 : {metrics['Score']:.2f}")
        print("="*60)
    
    return metrics



tickers = ['CL=F', 'NG=F', 'GC=F', 'ES=F', '^OVX', '^VIX']
all_dfs = []

for ticker in tickers:
    all_dfs.append(data_fetch.generate_table(ticker_sym=ticker))


final_df=pd.concat(all_dfs, ignore_index=True)
preds=pd.DataFrame(pd.read_csv('data\preds_lgbm_option_b.csv', parse_dates=['timestamp']))

trading_signals = generate_trading_signals(
    predictions=preds,
    threshold=0.0031,      
    allow_short=True
)

result, trade_log_df = backtest_strategy(
    signals=trading_signals,
    price_data=final_df,           
    ticker='CL=F',
    initial_capital=100_000,
    multiplier=1000,               
    commission_per_contract=1.5,
    stop_loss_pct=0.005,
    risk_per_trade=0.01
)

risk_levels = [0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.7, 0.8, 0.9, 1]
stop_loss_levels = [0.03, 0.05, 0.1, 0.2]
score_results = []
for risk in risk_levels:
    for stop in stop_loss_levels:
        result, _ = backtest_strategy(
            signals=trading_signals,
            price_data=final_df,
            ticker='CL=F',
            risk_per_trade=risk,
            stop_loss_pct=stop
        )
        calculate_performance_metrics(result)




if not trade_log_df.empty:
    buys  = trade_log_df[trade_log_df['action'].str.contains('BUY|Long', case=False, na=False)]
    sells = trade_log_df[trade_log_df['action'].str.contains('SELL|Short', case=False, na=False)]
else:
    buys = sells = pd.DataFrame()
    print("Warning: No trades were logged. Check backtest_strategy logic.")


oil = final_df[final_df['Ticker'] == 'CL=F'].copy()
oil = oil[(oil['timestamp'] >= result['timestamp'].min()) & 
          (oil['timestamp'] <= result['timestamp'].max())]

# Filter trades
buys  = trade_log_df[trade_log_df['action'].str.contains('Buy|Long', case=False, na=False)]
sells = trade_log_df[trade_log_df['action'].str.contains('Sell|Short', case=False, na=False)]

buys   = trade_log_df[trade_log_df['action'].str.contains('Buy|Long', case=False, na=False)]
sells  = trade_log_df[trade_log_df['action'].str.contains('Sell|Short', case=False, na=False)]
stops  = trade_log_df[trade_log_df['action'].str.contains('Stop Loss', case=False, na=False)]

plt.figure(figsize=(14, 8))

# Primary axis: Equity Curve
ax1 = plt.gca()
ax1.plot(result['timestamp'], result['equity'], 
         label='Strategy Equity', color='blue', linewidth=2.5)
ax1.set_ylabel('Equity ($)', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')
ax1.set_xlabel('Date')
ax1.grid(True, alpha=0.3)

# Secondary axis: Oil Price
ax2 = ax1.twinx()
ax2.plot(oil['timestamp'], oil['Close'], 
         label='Crude Oil Close Price', color='red', linewidth=1.8, alpha=0.85)
ax2.set_ylabel('Oil Price ($)', color='red')
ax2.tick_params(axis='y', labelcolor='red')

# === Trade Markers ===
# Buy - Green circles
if not buys.empty:
    buy_eq = result[result['timestamp'].isin(buys['timestamp'])]['equity']
    ax1.scatter(buys['timestamp'], buy_eq, 
                color='green', s=130, label='BUY', zorder=5, marker='o', edgecolors='black')

# Sell - Red circles
if not sells.empty:
    sell_eq = result[result['timestamp'].isin(sells['timestamp'])]['equity']
    ax1.scatter(sells['timestamp'], sell_eq, 
                color='red', s=130, label='SELL', zorder=5, marker='o', edgecolors='black')

# Stop Loss - Orange triangles (pointing down)
if not stops.empty:
    stop_eq = result[result['timestamp'].isin(stops['timestamp'])]['equity']
    ax1.scatter(stops['timestamp'], stop_eq, 
                color='orange', s=160, label='STOP LOSS', zorder=6, 
                marker='v', edgecolors='darkred', linewidth=1.5)

plt.title('Strategy Equity Curve vs Crude Oil Price\nwith Buy / Sell / Stop Loss Markers')
# Combined legend
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()