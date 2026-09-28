import pandas as pd
import pandas_ta as ta
import os
import glob
from sklearn.preprocessing import MinMaxScaler

def process_and_normalize(filepath):
    print(f"Processing {filepath}...")
    
    # Load the raw CSV, skipping the broken yfinance ticker row
    df = pd.read_csv(filepath, skiprows=[1])
    
    # We need datetime index for some indicators
    if 'Datetime' in df.columns:
        df['Datetime'] = pd.to_datetime(df['Datetime'], utc=True)
        df.set_index('Datetime', inplace=True)
    elif 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'], utc=True)
        df.set_index('Date', inplace=True)
        
    # --- FEATURE ENGINEERING ---
    # Calculate RSI (14 period)
    df.ta.rsi(length=14, append=True)
    
    # Calculate MACD
    df.ta.macd(fast=12, slow=26, signal=9, append=True)
    
    # Calculate ATR (Volatility)
    df.ta.atr(length=14, append=True)
    
    # Calculate EMAs
    df.ta.ema(length=20, append=True)
    df.ta.ema(length=50, append=True)
    df.ta.ema(length=200, append=True)
    
    # Drop rows that have NaNs (because EMAs/RSI need a warmup period)
    df.dropna(inplace=True)
    
    if df.empty:
        print(f"Skipping {filepath} - not enough data after indicator warmup.")
        return
        
    # --- NORMALIZATION ---
    # The neural network prefers numbers between 0 and 1.
    scaler = MinMaxScaler()
    
    # Keep the columns we want to train on
    # Depending on what Yahoo Finance called them, find MACD/RSI columns
    feature_cols = [col for col in df.columns if col not in ['Open', 'High', 'Low', 'Adj Close']]
    
    # We will normalize Volume, Close, and all our indicators
    df_normalized = df.copy()
    df_normalized[feature_cols] = scaler.fit_transform(df[feature_cols])
    
    # Save the processed data
    out_name = filepath.replace('.csv', '_PROCESSED.csv')
    df_normalized.to_csv(out_name)
    print(f"Saved processed data to {out_name}")

if __name__ == "__main__":
    # Find all the raw CSV files we downloaded
    csv_files = glob.glob("*.csv")
    
    # Filter out ones we already processed
    raw_files = [f for f in csv_files if "_PROCESSED" not in f]
    
    print(f"Found {len(raw_files)} raw data files. Adding indicators and normalizing...")
    
    for file in raw_files:
        try:
            process_and_normalize(file)
        except Exception as e:
            print(f"Error processing {file}: {e}")
            
    print("Data processing complete! Ready for PyTorch Transformer.")
