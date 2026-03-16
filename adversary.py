import pandas as pd
import numpy as np
import os

def create_stress_test():
    print("--- Tyche Adversary: Starting Stress Test Generation ---")
    
    # 1. Check if the clean data exists
    if not os.path.exists("data/market_prices.csv"):
        print("❌ Error: 'data/market_prices.csv' not found. Run data_engine.py first!")
        return

    # 2. Load the original clean data
    df = pd.read_csv("data/market_prices.csv", index_col=0)
    print(f"✅ Loaded {len(df)} rows of market data.")
    
    # 3. Inject a 'Flash Crash' (Pillar 2)
    # We drop the price of the first stock (NVDA) by 7% at the halfway point
    crash_point = len(df) // 2
    df.iloc[crash_point : crash_point + 15, 0] *= 0.93 
    print(f"⚠️  Injected Flash Crash at index {crash_point}.")
    
    # 4. Add High-Frequency Noise (Jitter)
    noise = np.random.normal(0, 0.002, df.shape)
    df_stressed = df + noise
    print("⚠️  Added Gaussian noise to simulate network jitter.")
    
    # 5. Save the stressed dataset
    df_stressed.to_csv("data/stressed_market.csv")
    print("✅ Success: 'data/stressed_market.csv' is ready for training.")

# THIS IS THE PART THAT CALLS THE FUNCTION (Don't forget this!)
if __name__ == "__main__":
    try:
        create_stress_test()
    except Exception as e:
        print(f"❌ An error occurred: {e}")