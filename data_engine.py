import yfinance as yf
import pandas as pd
import os

def setup_market():
    if not os.path.exists('data'): os.makedirs('data')
    tickers = ["NVDA", "TSM", "AMD", "INTC", "ASML"]
    print("--- Project Tyche: Downloading Price & Volume Data ---")
    
    raw_data = yf.download(tickers, period="5d", interval="1m")
    
    # Get both Close and Volume
    prices = raw_data['Close'].dropna()
    volume = raw_data['Volume'].dropna()
    
    # Merge them: NVDA_price, NVDA_volume, etc.
    combined = pd.DataFrame()
    for t in tickers:
        combined[f"{t}_price"] = prices[t]
        combined[f"{t}_vol"] = volume[t]
    
    combined.to_csv("data/market_prices.csv")
    print("✅ Data Engine: Now including Volume features.")

if __name__ == "__main__":
    setup_market()