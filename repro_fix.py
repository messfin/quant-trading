import pandas as pd
import yfinance as yf
import numpy as np

def test_flattening():
    # Simulate yfinance MultiIndex output
    ticker = 'AAPL'
    df = yf.download(ticker, period='5d', auto_adjust=True)
    print(f"Original columns: {df.columns}")
    
    # Apply my flattening logic
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    
    print(f"Flattened columns: {df.columns}")
    print(f"Type of df['Close']: {type(df['Close'])}")
    
    new = df.copy().reset_index()
    print(f"Type of new['Close']: {type(new['Close'])}")
    
    try:
        val = new['Close'][1]
        print(f"new['Close'][1] access successful: {val}")
    except Exception as e:
        print(f"new['Close'][1] failed: {e}")

if __name__ == '__main__':
    test_flattening()
