import yfinance as yf

data = yf.download("TCS.NS", start="2025-01-01", end="2025-09-01")
data.to_csv("TCS.NS.csv")