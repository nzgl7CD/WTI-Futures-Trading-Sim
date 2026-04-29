import yfinance as yf
import pandas as pd
import os

os.makedirs('data', exist_ok=True)

tickers = ['CL=F']        # Will add more later: 'NG=F', 'GC=F'

data = yf.download(tickers, 
                   start='2020-01-01', 
                   end='2025-12-31',
                   progress=False)


if isinstance(data.columns, pd.MultiIndex):
    data.columns = [col[0] if isinstance(col, tuple) else col for col in data.columns]

if len(tickers) == 1:
    data = data.reset_index()
else:
    data = data.stack(level=0).reset_index()  

filename = 'data/CL_futures_raw.csv'
data.to_csv(filename, index=False)

print('done')