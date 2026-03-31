"""
Loads 1-second Binance parquet data for all assets.
Falls back to synthetic GBM data if parquet not found.
"""
import os
import numpy as np
import pandas as pd

DATA_DIR = "data"
ASSETS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]

def load_asset(symbol: str) -> pd.DataFrame:
    pq = os.path.join(DATA_DIR, symbol, f"{symbol}_1s.parquet")
    if os.path.exists(pq):
        df = pd.read_parquet(pq)
        df = df[["open","high","low","close","volume"]].astype(float).dropna()
        print(f"  [DATA] {symbol}: {len(df):,} 1-second bars loaded")
        return df
    print(f"  [SYN]  {symbol}: generating 2,000,000 synthetic 1-second bars")
    n = 2_000_000
    np.random.seed(abs(hash(symbol)) % (2**31))
    prices = {"BTCUSDT":45000,"ETHUSDT":2500,"BNBUSDT":300,"SOLUSDT":100,"ADAUSDT":0.5}
    p0 = prices.get(symbol, 100.0)
    dt = 1.0 / 86400.0
    log_r = np.random.normal(0.0002*dt, 0.02*np.sqrt(dt), n)
    close = p0 * np.exp(np.cumsum(log_r))
    high  = close * (1.0 + np.abs(np.random.normal(0, 0.0005, n)))
    low   = close * (1.0 - np.abs(np.random.normal(0, 0.0005, n)))
    op    = np.roll(close, 1); op[0] = p0
    vol   = np.abs(np.random.normal(1e6, 3e5, n))
    idx   = pd.date_range("2024-01-01", periods=n, freq="1s")
    return pd.DataFrame({"open":op,"high":high,"low":low,"close":close,"volume":vol}, index=idx)

def load_all() -> dict:
    return {sym: load_asset(sym) for sym in ASSETS}

def align_data(dfs: dict):
    common = None
    for df in dfs.values():
        common = df.index if common is None else common.intersection(df.index)
    aligned = {sym: df.loc[common] for sym, df in dfs.items()}
    print(f"  [ALIGN] {len(common):,} common bars across {len(ASSETS)} assets")
    return aligned, common
