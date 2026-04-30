import yfinance as yf
import pandas as pd
import os
import ta  
import numpy as np             

os.makedirs('data', exist_ok=True)


def generate_table(ticker_sym,start_date='2005-01-01',end_date='2025-12-31'):
       # Will add more later: 'NG=F', 'GC=F'

    df = yf.download(ticker_sym, 
                    start=start_date, 
                    end=end_date,
                    progress=False,
                    interval='1d')

    # vix_df=yf.download('^VIX', 
    #                 start=start_date, 
    #                 end=end_date,
    #                 progress=False,
    #                 interval='1d')
    
    # ovx_df=yf.download('^OVX', 
    #                 start=start_date, 
    #                 end=end_date,
    #                 progress=False,
    #                 interval='1d')
    

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()
    df['Ticker']=ticker_sym
    # df['vix_closing']=vix_df['Close']
    # df['ovx_closing']=ovx_df['Close']

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
    return df

tickers = ['CL=F', 'NG=F', 'GC=F', 'ES=F', '^OVX', '^VIX']
all_dfs = []
for ticker in tickers:
    all_dfs.append(generate_table(ticker_sym=ticker))
final_df=pd.concat(all_dfs, ignore_index=True)
# print(final_df.head(50))
filename = 'data/CL_futures_raw.csv'
final_df.to_csv(filename, index=False)

print('done')