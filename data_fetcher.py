import yfinance as yf
import pandas as pd
import os

def fetch_historical_data(ticker, period="2y", interval="1h"):
    """
    Fetches historical data directly from the internet (Yahoo Finance).
    No API keys required.
    
    Args:
        ticker (str): The symbol to fetch (e.g., 'BTC-USD', 'EURUSD=X')
        period (str): How far back to go (e.g., '2y' for 2 years, '60d' for 60 days)
        interval (str): Timeframe (e.g., '1h' for 1 hour, '5m' for 5 minute)
    """
    print(f"Downloading {ticker} data from the internet...")
    
    # Download data using yfinance
    df = yf.download(tickers=ticker, period=period, interval=interval)
    
    if df.empty:
        print("Failed to fetch data. Check your ticker or internet connection.")
        return None
        
    # Clean up the dataframe
    df = df.dropna()
    df.reset_index(inplace=True)
    
    # Save it to a CSV file automatically
    filename = f"{ticker.replace('-', '_').replace('=', '')}_{interval}.csv"
    df.to_csv(filename, index=False)
    
    print(f"Successfully downloaded {len(df)} candles!")
    print(f"Saved automatically as: {filename}")
    
    return df

if __name__ == "__main__":
    # A massive list of diverse assets to train the Oracle
    assets = [
        # Crypto
        "BTC-USD",   # Bitcoin
        "ETH-USD",   # Ethereum
        "SOL-USD",   # Solana
        
        # Forex
        "EURUSD=X",  # Euro / US Dollar
        "AUDUSD=X",  # Aussie / US Dollar
        "GBPUSD=X",  # British Pound / US Dollar
        "USDJPY=X",  # US Dollar / Japanese Yen
        
        # Metals (Commodities)
        "GC=F",      # Gold Futures
        "SI=F",      # Silver Futures
        
        # Indices
        "^GSPC",     # S&P 500
        "^IXIC",     # Nasdaq Composite
        "^DJI"       # Dow Jones
    ]
    
    print("Initializing massive data pull for Phase 3 Oracle...")
    
    for ticker in assets:
        # Fetch 2 years of 1-Hour data for every asset
        fetch_historical_data(ticker, period="2y", interval="1h")
