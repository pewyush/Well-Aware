
import yfinance as yf
import pandas as pd
import os

os.makedirs('./data/raw', exist_ok=True)

tickers = [
    'RELIANCE.NS',  # Reliance Industries
    'INFY.NS',      # Infosys
    'TCS.NS',       # Tata Consultancy Services
    'HDFCBANK.NS',  # HDFC Bank
    'ICICIBANK.NS', # ICICI Bank
    'SBIN.NS',      # State Bank of India
    'BHARTIARTL.NS', # Airtel
    'ITC.NS',       # ITC Limited
    'HINDUNILVR.NS', # Hindustan Unilever
    'AXISBANK.NS'   # Axis Bank
]

print("Downloading data for 10 major Indian stocks...")

for ticker in tickers:
    print(f"Downloading: {ticker}")
    try:
       
        data = yf.download(ticker, period="6mo", progress=False)
        
        
        filename = f'./data/raw/{ticker}_prices.csv'
        data.to_csv(filename)
        print(f"Saved: {filename}")
        
    except Exception as e:
        print(f"Failed to download {ticker}: {e}")

print("All downloads completed! Check the '../data/raw/' folder.")