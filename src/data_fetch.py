import yfinance as yf
import pandas as pd
import os
import ta  
import numpy as np             
from datetime import date, datetime, timedelta
import matplotlib.pyplot as plt
import meteostat as ms
import eia
import requests
os.makedirs('data', exist_ok=True)


api_key = "py7iN2Q56STMQTeed4o9sxFavIP9VBSuzzZub6YQ"
api = eia.API(api_key)

def generate_table(start_date='2000-01-01', end_date='2026-05-29'):
    """Generate features with ticker-specific column names."""
    
    tickers = ['CL=F', 'NG=F', 'GC=F', 'ES=F', '^OVX', '^VIX', 'DX-Y.NYB']
    all_dfs = []
    for ticker in tickers:
            

        df = yf.download(ticker, 
                        start=start_date, 
                        end=end_date,
                        progress=False,
                        interval='1d')

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [col[0] for col in df.columns]

        df = df.reset_index()
        df['Ticker'] = ticker
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
        df.reset_index(drop=True, inplace=True)
        all_dfs.append(df)

    return pd.concat(all_dfs, ignore_index=True)

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

def weather_data_pipeline(start_date: date=date(2020, 5, 1), end_date: date=date(2026, 5, 29)) -> pd.DataFrame:


    START=start_date
    END = end_date
    

    # ==================== CITY COORDINATES ====================

    POINT_NEW_YORK = ms.Point(40.7128, -74.0060, 10)   # United States - New York
    POINT_MOSCOW   = ms.Point(55.7558, 37.6173, 150)  # Russia - Moscow
    POINT_BEIJING  = ms.Point(39.9042, 116.4074, 10)  # China - Beijing
    POINT_SHANGHAI = ms.Point(31.2304, 121.4737, 10)  # China - Shanghai
    POINT_CALGARY  = ms.Point(51.0447, -114.0719, 10) # Canada - Calgary
    POINT_TORONTO  = ms.Point(43.6532, -79.3832, 10)  # Canada - Toronto

    # ==================== FETCH STATIONS ====================

    stations_new_york = ms.stations.nearby(POINT_NEW_YORK, limit=4)
    stations_moscow   = ms.stations.nearby(POINT_MOSCOW, limit=4)
    stations_beijing  = ms.stations.nearby(POINT_BEIJING, limit=4)
    stations_shanghai = ms.stations.nearby(POINT_SHANGHAI, limit=4)
    stations_calgary  = ms.stations.nearby(POINT_CALGARY, limit=4)
    stations_toronto  = ms.stations.nearby(POINT_TORONTO, limit=4)

    # ==================== FETCH DAILY DATA ====================

    ts_new_york = ms.daily(stations_new_york, START, END)
    ts_moscow   = ms.daily(stations_moscow, START, END)
    ts_beijing  = ms.daily(stations_beijing, START, END)
    ts_shanghai = ms.daily(stations_shanghai, START, END)
    ts_calgary  = ms.daily(stations_calgary, START, END)
    ts_toronto  = ms.daily(stations_toronto, START, END)

    # ==================== INTERPOLATE & CREATE DATAFRAMES ====================

    df_new_york = ms.interpolate(ts_new_york, POINT_NEW_YORK).fetch().reset_index()
    df_moscow   = ms.interpolate(ts_moscow, POINT_MOSCOW).fetch().reset_index()
    df_beijing  = ms.interpolate(ts_beijing, POINT_BEIJING).fetch().reset_index()
    df_shanghai = ms.interpolate(ts_shanghai, POINT_SHANGHAI).fetch().reset_index()
    df_calgary  = ms.interpolate(ts_calgary, POINT_CALGARY).fetch().reset_index()
    df_toronto  = ms.interpolate(ts_toronto, POINT_TORONTO).fetch().reset_index()

    # ==================== RENAME COLUMNS ====================

    for weather_df, prefix in [
        (df_new_york, "new_york"),
        (df_moscow, "moscow"),
        (df_beijing, "beijing"),
        (df_shanghai, "shanghai"),
        (df_calgary, "calgary"),
        (df_toronto, "toronto")
    ]:
        for column in weather_df.columns:
            if column == 'time':
                weather_df.rename(columns={'time': 'timestamp'}, inplace=True)
            else:
                weather_df.rename(columns={column: f"{prefix}_{column}"}, inplace=True)

    # ==================== MISSING VALUES CHECK ====================

    for name, weather_df in [
        ("New York", df_new_york),
        ("Moscow", df_moscow),
        ("Beijing", df_beijing),
        ("Shanghai", df_shanghai),
        ("Calgary", df_calgary),
        ("Toronto", df_toronto)
    ]:
        for column in weather_df.columns:
            if weather_df[column].isnull().any():
                print(f"Column '{column}' in {name} data has missing values.")
            else:
                print(f"Column '{column}' in {name} data has no missing values.")

    # ==================== ADD HDD / CDD FEATURES ====================

    for weather_df, temp_col, prefix in [
        (df_new_york, 'new_york_temp', 'new_york'),
        (df_moscow, 'moscow_temp', 'moscow'),
        (df_beijing, 'beijing_temp', 'beijing'),
        (df_shanghai, 'shanghai_temp', 'shanghai'),
        (df_calgary, 'calgary_temp', 'calgary'),
        (df_toronto, 'toronto_temp', 'toronto')
    ]:
        weather_df = add_hdd_cdd_features(weather_df, temp_col=temp_col, prefix=prefix)



    return (df_new_york
                .merge(df_moscow,    on='timestamp', how='left')
                .merge(df_beijing,   on='timestamp', how='left')
                .merge(df_shanghai,  on='timestamp', how='left')
                .merge(df_calgary,   on='timestamp', how='left')
                .merge(df_toronto,   on='timestamp', how='left')
                
    )


def get_ng_storage_data(start_date: str = '2025-01-01') -> pd.DataFrame:
    """
    Fetch weekly US Natural Gas Storage using EIA API (recommended)
    """    
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
        df= pd.DataFrame(data['response']['data'])
    else:
        print("Unexpected API format. Raw keys:", list(data.keys()))
        return pd.DataFrame()
    

    # Clean column names
    df = df.rename(columns={
        'period': 'timestamp',
        'value': 'ng_storage_bcf'
    })

    df['timestamp'] = pd.to_datetime(df['timestamp'])


    df['ng_storage_bcf'] = df['ng_storage_bcf'].astype(int)
    # # Add useful features
    df['ng_storage_change_wow'] = df['ng_storage_bcf'].pct_change()
    df['ng_storage_5yr_avg'] = df['ng_storage_bcf'].rolling(window=260, min_periods=100).mean()
    df['ng_storage_vs_5yr'] = df['ng_storage_bcf'] / df['ng_storage_5yr_avg']


    return df


def get_trading_data(ticker_sym,start_date='2024-01-01',end_date='2026-05-01'):

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

    df.reset_index(drop=True, inplace=True)
    return df
   
df=generate_table(start_date='2000-01-01', end_date='2026-05-29')   
weather_df = weather_data_pipeline(start_date=date(2000,1,1), end_date=date(2026,5,29))
ng_storage_df = get_ng_storage_data('2000-01-01')
final_df = (df.merge(weather_df, on='timestamp', how='left')
                .merge(ng_storage_df, on='timestamp', how='left')
            )
filename = 'data/CL_futures_raw.csv'
final_df.to_csv(filename, index=False)

trading_decision_df = generate_table(start_date=(datetime.today() - timedelta(days=40)).strftime('%Y-%m-%d'), end_date=datetime.today().strftime('%Y-%m-%d'))
trading_decision_weather_df = weather_data_pipeline(start_date=(date.today() - timedelta(days=40)), end_date=date.today())
trading_decision_ng_storage_df = get_ng_storage_data(start_date=(datetime.today() - timedelta(days=40)).strftime('%Y-%m-%d'))

final_trading_decision_df = (trading_decision_df.merge(trading_decision_weather_df, on='timestamp', how='left')
                .merge(trading_decision_ng_storage_df, on='timestamp', how='left')
            )

final_trading_decision_df.to_csv('data/trading_decision_data.csv', index=False)

# trading_data=get_trading_data(ticker_sym='CL=F', start_date='2024-01-01', end_date='2026-05-01')
# filename = 'data/trading_data.csv'
# trading_data.to_csv(filename, index=False)