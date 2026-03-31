import os
import io
import zipfile
import requests
import pandas as pd
from datetime import datetime, timedelta
import time

ASSETS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
DATA_DIR = "data"
BASE_URL = "https://data.binance.vision/data/spot/monthly/klines"

def get_months(n=12):
    months = []
    d = datetime.utcnow().replace(day=1)
    for _ in range(n):
        d -= timedelta(days=1)
        d = d.replace(day=1)
        months.append((d.year, d.month))
    return list(reversed(months))

def download_month(symbol, year, month, out_dir):
    fname = f"{symbol}-1s-{year}-{month:02d}.zip"
    url = f"{BASE_URL}/{symbol}/1s/{fname}"
    csv_name = fname.replace(".zip", ".csv")
    out_path = os.path.join(out_dir, csv_name)
    if os.path.exists(out_path):
        print(f"  [SKIP] {symbol} {year}-{month:02d}")
        return True
    print(f"  [DOWN] {symbol} {year}-{month:02d} ...", end=" ", flush=True)
    try:
        r = requests.get(url, timeout=120, stream=True)
        if r.status_code != 200:
            print(f"MISS (HTTP {r.status_code})")
            return False
        z = zipfile.ZipFile(io.BytesIO(r.content))
        z.extractall(out_dir)
        print("OK")
        return True
    except Exception as e:
        print(f"ERR: {e}")
        return False

def merge_symbol(symbol, out_dir):
    files = sorted([f for f in os.listdir(out_dir) if f.startswith(symbol) and f.endswith(".csv")])
    if not files:
        return
    cols = ["open_time","open","high","low","close","volume",
            "close_time","qav","num_trades","tbbav","tbqav","ignore"]
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(os.path.join(out_dir, f), header=None, names=cols)
            dfs.append(df)
        except:
            pass
    if not dfs:
        return
    merged = pd.concat(dfs, ignore_index=True)

    # Detect microseconds vs milliseconds automatically
    sample = merged["open_time"].iloc[0]
    if sample > 1e15:
        unit = "us"   # microseconds
    elif sample > 1e12:
        unit = "ms"   # milliseconds
    else:
        unit = "s"    # seconds

    print(f"  [INFO] {symbol} timestamps detected as: {unit}")
    merged["open_time"] = pd.to_datetime(merged["open_time"], unit=unit)
    merged = merged.set_index("open_time").sort_index()
    merged = merged[["open","high","low","close","volume"]].astype(float)
    pq_path = os.path.join(out_dir, f"{symbol}_1s.parquet")
    merged.to_parquet(pq_path)
    print(f"  [MERGE] {symbol}: {len(merged):,} rows saved")

def main():
    os.makedirs(DATA_DIR, exist_ok=True)
    months = get_months(12)
    print(f"Downloading 1-second data | {len(ASSETS)} assets | {len(months)} months each")
    print(f"Estimated size: ~2-4 GB total")
    print("=" * 60)
    for symbol in ASSETS:
        sym_dir = os.path.join(DATA_DIR, symbol)
        os.makedirs(sym_dir, exist_ok=True)
        print(f"\n[{symbol}]")
        for year, month in months:
            download_month(symbol, year, month, sym_dir)
            time.sleep(0.1)
        print(f"  Merging {symbol}...")
        merge_symbol(symbol, sym_dir)
    print("\n" + "=" * 60)
    print("DOWNLOAD COMPLETE")

if __name__ == "__main__":
    main()
