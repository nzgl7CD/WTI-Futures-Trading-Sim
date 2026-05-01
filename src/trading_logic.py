import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import data_fetch
from datetime import datetime, timedelta

def generate_trading_signals(
    predictions: pd.DataFrame,
    threshold: float = 0.0002,      # 0.2% predicted move as default TODO: test with 0.0015, 0.001, 0.0005 
    position_size: float = 1.0,    
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
    signals['position_size'] = position_size
    signals['position'] = signals['signal'] * signals['position_size']

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
    stop_loss_pct: float = 0.10
):
    """
    Robust backtester - Fixed for column names and merge issues.
    """
    print(f"Backtesting on {ticker} ...")

    prices = price_data[price_data['Ticker'] == ticker].copy()
    if prices.empty:
        raise ValueError(f"No data found for {ticker}. Available: {price_data['Ticker'].unique()}")

    prices = prices[['timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']].copy()
    prices['timestamp'] = pd.to_datetime(prices['timestamp']).dt.tz_localize(None)
    prices = prices.set_index('timestamp').sort_index()

    df = signals.copy()
    if isinstance(df.index, pd.DatetimeIndex):
        df = df.reset_index()                    # Make 'timestamp' a column
    elif 'timestamp' not in df.columns:
        raise ValueError("Signals must have 'timestamp' column or DatetimeIndex")

    df['timestamp'] = pd.to_datetime(df['timestamp']).dt.tz_localize(None)

    df = df.merge(prices, on='timestamp', how='left')

    df = df.dropna(subset=['Close']).reset_index(drop=True)

    if len(df) == 0:
        raise ValueError("No matching dates between signals and price data!")

    print(f"Successfully merged {len(df)} rows for backtesting.")

    # ====================== Backtest Loop ======================
    capital = float(initial_capital)
    position = 0
    entry_price = 0.0
    equity_curve = []
    total_commissions = 0.0

    for i in range(len(df)):
        row = df.iloc[i]
        price = float(row['Close'])
        signal = int(row.get('signal', 0))

        realized_pnl = 0.0
        commission = 0.0
        stop_loss_triggered = False

        if position != 0:
            if position > 0:
                stop_price = entry_price * (1 - stop_loss_pct)
                if float(row['Low']) <= stop_price:
                    realized_pnl = (stop_price - entry_price) * position * multiplier
                    commission = commission_per_contract * abs(position)
                    capital += realized_pnl - commission
                    position = 0
                    entry_price = 0.0
                    stop_loss_triggered = True
            else:
                stop_price = entry_price * (1 + stop_loss_pct)
                if float(row['High']) >= stop_price:
                    realized_pnl = (entry_price - stop_price) * abs(position) * multiplier
                    commission = commission_per_contract * abs(position)
                    capital += realized_pnl - commission
                    position = 0
                    entry_price = 0.0
                    stop_loss_triggered = True

        if not stop_loss_triggered and position != 0 and (signal == 0 or signal != np.sign(position)):
            if position > 0:
                realized_pnl = (price - entry_price) * position * multiplier
                action = 'Sell Long'
            else:
                realized_pnl = (entry_price - price) * abs(position) * multiplier
                action = 'Cover Short'

            commission = commission_per_contract * abs(position)
            capital += realized_pnl - commission
            position = 0
            entry_price = 0.0

        if not stop_loss_triggered:
            if signal == 1 and position == 0:
                size = float(row.get('position_size', row.get('position', 1.0)))
                position = size
                entry_price = price
                commission = commission_per_contract * abs(position)
                capital -= commission

            elif signal == -1 and position == 0:
                size = float(row.get('position_size', abs(row.get('position', 1.0))))
                position = -size
                entry_price = price
                commission = commission_per_contract * abs(position)
                capital -= commission

        # Calculate equity
        unrealized_pnl = (price - entry_price) * position * multiplier if position != 0 else 0.0
        current_equity = capital + unrealized_pnl

        equity_curve.append({
            'timestamp': row['timestamp'],
            'equity': current_equity,
            'position': position,
            'signal': signal,
            'close': price,
            'unrealized_pnl': unrealized_pnl,
            'realized_pnl': realized_pnl,
            'commission': commission,
            'stop_loss': stop_loss_triggered
        })

        total_commissions += commission

    result = pd.DataFrame(equity_curve)
    result['cum_return'] = result['equity'] / initial_capital - 1
    result['peak'] = result['equity'].cummax()
    result['drawdown_pct'] = (result['equity'] - result['peak']) / result['peak']

    # Summary
    print("\n" + "="*60)
    print("BACKTEST SUMMARY")
    print(f"Initial Capital     : ${initial_capital:,.0f}")
    print(f"Final Equity        : ${result['equity'].iloc[-1]:,.2f}")
    print(f"Total Return        : {result['cum_return'].iloc[-1]:.2%}")
    print(f"Max Drawdown        : {result['drawdown_pct'].min():.2%}")
    print(f"Total Commissions   : ${total_commissions:,.2f}")
    print("="*60)

    return result, pd.DataFrame()

tickers = ['CL=F', 'NG=F', 'GC=F', 'ES=F', '^OVX', '^VIX']
all_dfs = []

for ticker in tickers:
    all_dfs.append(data_fetch.generate_table(ticker_sym=ticker))
final_df=pd.concat(all_dfs, ignore_index=True)
print(final_df.columns)
# preds = create_mock_preds(periods=300, start_date='2024-01-01')
preds=pd.DataFrame(pd.read_csv('data\preds_lgbm_option_a.csv', parse_dates=['timestamp']))

trading_signals = generate_trading_signals(
    predictions=preds,
    threshold=0.00015,      
    position_size=1,
    allow_short=True
)

backtest_results, trade_log = backtest_strategy(
    signals=trading_signals,
    price_data=final_df,           
    ticker='CL=F',
    initial_capital=100_000,
    multiplier=1000,               
    commission_per_contract=1.5 # We can adjust if needed
)

# print(preds.head(10))
# print(trading_signals[['predicted_return', 'signal', 'direction']].head(10))

plt.figure(figsize=(12, 6))
plt.plot(backtest_results['timestamp'], backtest_results['equity'])
plt.title('Strategy Equity Curve - Crude Oil')
plt.ylabel('Equity ($)')
plt.xlabel('Date')
plt.grid(True)
plt.show()


# Test
# print(trading_signals[['predicted_return', 'signal', 'direction']].head(10))