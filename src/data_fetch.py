import yfinance as yf
import pandas as pd
import os
import ta  
import numpy as np             

os.makedirs('data', exist_ok=True)

tickers = ['CL=F']        # Will add more later: 'NG=F', 'GC=F'

df = yf.download(tickers, 
                   start='2015-01-01', 
                   end='2025-12-31',
                   progress=False,
                   interval='1d')


if isinstance(df.columns, pd.MultiIndex):
    df.columns = [col[0] for col in df.columns]

df = df.reset_index()
df.rename(columns={'Datetime': 'timestamp', 'Date': 'timestamp'}, inplace=True)
df['timestamp'] = pd.to_datetime(df['timestamp'])

# Momentum
df['rsi'] = ta.momentum.RSIIndicator(df['Close'], window=14).rsi()
df['macd'] = ta.trend.MACD(df['Close']).macd()

# Volatility
df['atr'] = ta.volatility.AverageTrueRange(df['High'], df['Low'], df['Close'], window=14).average_true_range()
df['bollinger_pband'] = ta.volatility.BollingerBands(df['Close']).bollinger_pband()

# Trend
df['sma_20'] = ta.trend.SMAIndicator(df['Close'], window=20).sma_indicator()
df['ema_12'] = ta.trend.EMAIndicator(df['Close'], window=12).ema_indicator()

# Volume
df['obv'] = ta.volume.OnBalanceVolumeIndicator(df['Close'], df['Volume']).on_balance_volume()

# Simple returns and lags
df['return'] = df['Close'].pct_change()
df['log_return'] = np.log(df['Close'] / df['Close'].shift(1))

for lag in [1, 2, 3, 5]:
    df[f'return_lag_{lag}'] = df['return'].shift(lag)

# Time features
df['dayofweek'] = df['timestamp'].dt.dayofweek
df['month'] = df['timestamp'].dt.month

# df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

filename = 'data/CL_futures_raw.csv'
df.to_csv(filename, index=False)

print('done')