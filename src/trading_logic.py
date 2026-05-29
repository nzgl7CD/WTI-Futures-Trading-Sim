import pandas as pd
import numpy as np
import matplotlib.pyplot as plt




def generate_trading_signals(
    predictions: pd.DataFrame,
    strong_threshold_long: float = 0.002,
    strong_threshold_short: float = -0.002,
    weak_threshold_long: float = 0.001,
    weak_threshold_short: float = -0.001,
    strong_position_size: float = 0.8,
    weak_position_size: float = 0.4,
    allow_short: bool = True
) -> pd.DataFrame:
    
    signals = predictions.copy()
    
    if "target_a_close" in signals.columns:
        pred_col = "target_a_close"
    elif "target_b_close" in signals.columns:
        pred_col = "target_b_close"
    elif "target_open_t10" in signals.columns:
        pred_col = "target_open_t10"
    elif "target_dir_t10" in signals.columns:
        pred_col = "target_dir_t10"
    else:
        raise ValueError("Could not find predicted return column")
    
    signals['signal'] = 0
    signals['position_size'] = 0.0
    
    # Long signals
    signals.loc[signals[pred_col] > strong_threshold_long, ['signal', 'position_size']] = [1, strong_position_size]
    signals.loc[(signals[pred_col] > weak_threshold_long) & 
                (signals[pred_col] <= strong_threshold_long), ['signal', 'position_size']] = [1, weak_position_size]

    # Short signals
    if allow_short:
        signals.loc[signals[pred_col] < strong_threshold_short, ['signal', 'position_size']] = [-1, strong_position_size]
        signals.loc[(signals[pred_col] < weak_threshold_short) & 
                    (signals[pred_col] >= strong_threshold_short), ['signal', 'position_size']] = [-1, weak_position_size]

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
    execution_price: str = 'Open',
    prediction_horizon: int = 10,     
    min_holding_days: int = 5          
):
    print(f"Backtesting {ticker} | Horizon: {prediction_horizon}d | Min hold: {min_holding_days}d")

    # Prepare prices
    prices = price_data[price_data['Ticker'] == ticker].copy()
    if prices.empty:
        raise ValueError(f"No data for {ticker}")

    prices = prices[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    prices['timestamp'] = pd.to_datetime(prices['timestamp']).dt.tz_localize(None)
    prices = prices.set_index('timestamp').sort_index()

    # Prepare signals
    df = signals.copy()
    if 'timestamp' not in df.columns:
        df = df.reset_index()
    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

    df = df.merge(prices, on='timestamp', how='left')
    df = df.dropna(subset=[execution_price]).reset_index(drop=True)

    capital = float(initial_capital)
    position = 0                     # net contracts (positive = long)
    entry_prices = []                # list to calculate average entry
    equity_curve = []
    trade_log = []
    total_commissions = 0.0
    count_stop_loss = 0

    for i in range(len(df)):
        row = df.iloc[i]
        exec_price = float(row[execution_price])
        mtm_price = float(row['Close'])
        signal = int(row.get('signal', 0))
        desired_position_size = float(row.get('position_size', 0.0))   # 0.0 to 1.0

        realized_pnl = 0.0
        commission = 0.0

        # === Stop Loss ===
        if position != 0 and len(entry_prices) > 0:
            avg_entry = sum(entry_prices) / len(entry_prices)
            stop_price = avg_entry * (1 - stop_loss_pct) if position > 0 else avg_entry * (1 + stop_loss_pct)
            
            hit_stop = (position > 0 and mtm_price <= stop_price) or (position < 0 and mtm_price >= stop_price)
            
            if hit_stop:
                count_stop_loss += 1
                realized_pnl = (mtm_price - avg_entry) * position * multiplier
                commission = commission_per_contract * abs(position)
                capital += realized_pnl - commission
                trade_log.append({'timestamp': row['timestamp'], 'action': 'Stop Loss', 'price': mtm_price, 'pnl': realized_pnl})
                position = 0
                entry_prices = []

        # === Close on opposite / flat signal after min holding ===
        if position != 0:
            days_held = (row['timestamp'] - trade_log[-1]['timestamp']).days if trade_log else 999
            if days_held >= min_holding_days and (signal == 0 or signal != np.sign(position)):
                avg_entry = sum(entry_prices) / len(entry_prices)
                realized_pnl = (mtm_price - avg_entry) * position * multiplier
                commission = commission_per_contract * abs(position)
                capital += realized_pnl - commission
                trade_log.append({'timestamp': row['timestamp'], 'action': 'Exit', 'price': mtm_price, 'pnl': realized_pnl})
                position = 0
                entry_prices = []

        # === Adjust toward target position ===
        if desired_position_size > 0:
            risk_amount = capital * desired_position_size
            stop_distance = exec_price * stop_loss_pct or (exec_price * 0.05)
            target_contracts = int(risk_amount / (stop_distance * multiplier))
            target_contracts = max(1, min(target_contracts, 30))

            # Capital check
            contract_value = exec_price * multiplier
            max_affordable = int((capital * 0.95) / contract_value)
            target_contracts = min(target_contracts, max_affordable)

            target_position = target_contracts if signal == 1 else -target_contracts

            # How many contracts to trade
            contracts_to_trade = target_position - position

            if contracts_to_trade != 0:
                trade_size = abs(contracts_to_trade)
                commission = commission_per_contract * trade_size

                if contracts_to_trade > 0:
                    action = 'Buy Long' if position <= 0 else 'Add Long'
                    capital -= commission
                else:
                    action = 'Sell Short' if position >= 0 else 'Reduce Short'
                    capital += commission   # we receive money from selling

                position = target_position
                entry_prices.extend([exec_price] * trade_size)

                trade_log.append({
                    'timestamp': row['timestamp'],
                    'action': action,
                    'price': exec_price,
                    'size': trade_size,
                    'position_size': desired_position_size
                })

        # Mark-to-market
        avg_entry = sum(entry_prices) / len(entry_prices) if entry_prices else 0.0
        unrealized = (mtm_price - avg_entry) * position * multiplier if position != 0 else 0.0
        current_equity = capital + unrealized

        equity_curve.append({
            'timestamp': row['timestamp'],
            'equity': current_equity,
            'position': position,
            'signal': signal,
            'close': mtm_price,
            'avg_entry': avg_entry
        })

        total_commissions += commission

    result = pd.DataFrame(equity_curve)
    result = result.dropna(subset=['equity']).reset_index(drop=True)

    result['cum_return'] = result['equity'] / initial_capital - 1
    result['peak'] = result['equity'].cummax()
    result['drawdown_pct'] = (result['equity'] - result['peak']) / result['peak']

    print("\n" + "="*80)
    print("BACKTEST SUMMARY")
    print(f"Initial Capital     : ${initial_capital:,.0f}")
    print(f"Final Equity        : ${result['equity'].iloc[-1]:,.2f}")
    print(f"Total Return        : {result['cum_return'].iloc[-1]:.2%}")
    print(f"Max Drawdown        : {result['drawdown_pct'].min():.2%}")
    print(f"Stop Loss Triggers  : {count_stop_loss}")
    print(f"Total Commissions   : ${total_commissions:,.2f}")
    print("="*80)

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
        print(f"Score                 : {metrics['Score']:.2f}")
        print("="*60)
        
    return metrics


preds=pd.DataFrame(pd.read_csv('data\preds_lgbm_option_direction_t10.csv', parse_dates=['timestamp']))
final_df=pd.DataFrame(pd.read_csv("data\CL_futures_raw.csv", parse_dates=['timestamp']))

trading_signals = generate_trading_signals(
    predictions=preds,
    strong_threshold_long=0.65,
    weak_threshold_long=0.6,
    strong_threshold_short=0.35,
    weak_threshold_short=0.4,
    strong_position_size=0.15,
    weak_position_size=0.01,
    allow_short=True
)

result, trade_log_df = backtest_strategy(
    signals=trading_signals,
    price_data=final_df,
    ticker='CL=F',
    initial_capital=100_000,
    prediction_horizon=10,
    min_holding_days=8,         
    stop_loss_pct=0.15,
    execution_price='Close'       
)

calculate_performance_metrics(result)
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

if not buys.empty:
    buy_eq = result[result['timestamp'].isin(buys['timestamp'])]['equity']
    ax1.scatter(buys['timestamp'], buy_eq, 
                color='green', s=130, label='BUY', zorder=5, marker='o', edgecolors='black')

# Sell - Red circles
if not sells.empty:
    sell_eq = result[result['timestamp'].isin(sells['timestamp'])]['equity']
    ax1.scatter(sells['timestamp'], sell_eq, 
                color='red', s=130, label='SELL', zorder=5, marker='o', edgecolors='black')

if not stops.empty:
    stop_eq = result[result['timestamp'].isin(stops['timestamp'])]['equity']
    ax1.scatter(stops['timestamp'], stop_eq, 
                color='orange', s=160, label='STOP LOSS', zorder=6, 
                marker='v', edgecolors='darkred', linewidth=1.5)

plt.title('Strategy Equity Curve vs Crude Oil Price\nwith Buy / Sell / Stop Loss Markers')

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')

plt.tight_layout()
plt.show()