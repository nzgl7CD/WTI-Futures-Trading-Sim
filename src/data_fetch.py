import yfinance as yf
import pandas as pd
import os
import ta  
import numpy as np             
from datetime import date
import matplotlib.pyplot as plt
import meteostat as ms
import eia
import requests

os.makedirs('data', exist_ok=True)

def generate_table(ticker_sym,start_date='2005-01-01',end_date='2026-05-01'):
       # Will add more later: 'NG=F', 'GC=F'

    df = yf.download(ticker_sym, 
                    start=start_date, 
                    end=end_date,
                    progress=False,
                    interval='1d')

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [col[0] for col in df.columns]

    df = df.reset_index()
    df['Ticker']=ticker_sym

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

def add_hdd_cdd_features(df: pd.DataFrame, temp_col: str, prefix: str, base_temp: float = 15.5) -> pd.DataFrame:
    """
    Add Heating Degree Days (HDD) and Cooling Degree Days (CDD) + rolling sums.
    
    df: DataFrame with weather data
    temp_col: column name like 'new_york_temp'
    prefix: 'new_york' or 'moscow'
    base_temp: 15.5°C is standard for Europe/US gas demand
    """
    df = df.copy()
    
    # HDD and CDD
    df[f'{prefix}_HDD'] = (base_temp - df[temp_col]).clip(lower=0)
    df[f'{prefix}_CDD'] = (df[temp_col] - base_temp).clip(lower=0)
    
    # Rolling sums - very important for gas demand
    for days in [7, 14, 30]:
        df[f'{prefix}_HDD_{days}d'] = df[f'{prefix}_HDD'].rolling(window=days, min_periods=1).sum()
        df[f'{prefix}_CDD_{days}d'] = df[f'{prefix}_CDD'].rolling(window=days, min_periods=1).sum()
    df[f'{prefix}_HDD_14d_vs_30d'] = df[f'{prefix}_HDD_14d'] / (df[f'{prefix}_HDD_30d'] + 1)  # ratio
    
    return df


def get_ng_storage_data(start_date: str = '2025-01-01') -> pd.DataFrame:
    """
    Fetch weekly US Natural Gas Storage using EIA API (recommended)
    """
    API_KEY = "YOUR_API_KEY_HERE"   # ← Put your real key here
    
    url = "https://api.eia.gov/v2/natural-gas/stor/wkly/data/"
    
    params = {
        "api_key": "py7iN2Q56STMQTeed4o9sxFavIP9VBSuzzZub6YQ",
        "frequency": "weekly",
        "data[0]": "value",
        "facets[series][]": "NW2_EPG0_SWO_R48_BCF",   # Working Gas - Lower 48 States
        "start": start_date,
        "sort[0][column]": "period",
        "sort[0][direction]": "asc",
        "offset": 0,
        "length": 5000
    }

    response = requests.get(url, params=params)
    
    if response.status_code != 200:
        print(f"API Error {response.status_code}: {response.text[:400]}")
        return pd.DataFrame()

    data = response.json()

    # Handle the actual structure returned by EIA
    if 'response' in data and 'data' in data['response']:
        df = pd.DataFrame(data['response']['data'])
    else:
        print("Unexpected API format. Raw keys:", list(data.keys()))
        return pd.DataFrame()

    # Clean column names
    df = df.rename(columns={
        'period': 'timestamp',
        'value': 'ng_storage_bcf'
    })

    df['timestamp'] = pd.to_datetime(df['timestamp'])
    df = df.sort_values('timestamp').reset_index(drop=True)
    df['ng_storage_bcf'] = df['ng_storage_bcf'].astype(int)
    # # Add useful features
    df['ng_storage_change_wow'] = df['ng_storage_bcf'].pct_change()
    df['ng_storage_5yr_avg'] = df['ng_storage_bcf'].rolling(window=260, min_periods=100).mean()
    df['ng_storage_vs_5yr'] = df['ng_storage_bcf'] / df['ng_storage_5yr_avg']


    return df
   

tickers = ['CL=F', 'NG=F', 'GC=F', 'ES=F', '^OVX', '^VIX']
all_dfs = []
for ticker in tickers:
    all_dfs.append(generate_table(ticker_sym=ticker))
df=pd.concat(all_dfs, ignore_index=True)

START = date(2005, 1, 1)
END = date(2026, 5, 1)

POINT_NEW_YORK = ms.Point(40.7128, -74.0060,10) #United States - New York

POINT_MOSCOW = ms.Point(55.7558, 37.6173, 150) #Russia - Moscow

stations_new_york = ms.stations.nearby(POINT_NEW_YORK, limit=4)
stations_moscow = ms.stations.nearby(POINT_MOSCOW, limit=4)

ts_new_york = ms.daily(stations_new_york, START, END)
df_new_york = ms.interpolate(ts_new_york, POINT_NEW_YORK).fetch().reset_index()

ts_moscow = ms.daily(stations_moscow, START, END)
df_moscow = ms.interpolate(ts_moscow, POINT_MOSCOW).fetch().reset_index()

for column in df_new_york.columns:
    if column == 'time':
        df_new_york.rename(columns={'time': 'timestamp'}, inplace=True)
    else:
        df_new_york.rename(columns={column: f"new_york_{column}"}, inplace=True)

for column in df_moscow.columns:
    if column == 'time':
        df_moscow.rename(columns={'time': 'timestamp'}, inplace=True)
    else:
        df_moscow.rename(columns={column: f"moscow_{column}"}, inplace=True)
for column in df_moscow.columns:
    if df_moscow[column].isnull().any():
        print(f"Column '{column}' in Moscow data has missing values.")
    else:
        print(f"Column '{column}' in Moscow data has no missing values.")

df_new_york = add_hdd_cdd_features(df_new_york, temp_col='new_york_temp', prefix='new_york')
df_moscow = add_hdd_cdd_features(df_moscow, temp_col='moscow_temp', prefix='moscow')

 
storage_df = get_ng_storage_data(start_date="2005-01-01")

final_df = df.merge(df_new_york, on='timestamp', how='left').merge(df_moscow, on='timestamp', how='left').merge(storage_df, on='timestamp', how='left')


filename = 'data/CL_futures_raw.csv'
final_df.to_csv(filename, index=False)
