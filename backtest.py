import pandas as pd
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from tyche_core.env import TycheTradeEnv
import os

def run_backtest():
    print("--- Project Tyche: Initializing Backtest Engine ---")
    
    # 1. Verification
    if not os.path.exists("models/tyche_v1.zip"):
        print("❌ Error: Model 'tyche_v1' not found. Run train.py first!")
        return

    # 2. Load Environment and Model
    # Note: Make sure env.py is loading the data correctly!
    try:
        env = TycheTradeEnv()
        model = PPO.load("models/tyche_v1")
        print("✅ Model and Environment Loaded Successfully.")
    except Exception as e:
        print(f"❌ Initialization Failed: {e}")
        return
    
    # 3. Execution Loop
    obs, _ = env.reset()
    portfolio_history = []
    cumulative_reward = 0
    
    print("--- Tyche is now analyzing the market ---")
    
    for i in range(300): # Test over 300 minutes
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        cumulative_reward += reward
        portfolio_history.append(cumulative_reward)
        
        # Log every 50 steps so you know it's working
        if i % 50 == 0:
            print(f"Step {i}: Total P/L = {cumulative_reward:.4f}")
        
        if done:
            break
            
    # 4. Visualization
    print("--- Generating Performance Analytics ---")
    plt.figure(figsize=(12, 6))
    plt.plot(portfolio_history, label='Tyche Equity Curve', color='#008080', linewidth=2)
    plt.axhline(0, color='red', linestyle='--', alpha=0.5)
    plt.title("Tyche Performance: Adversarial Market Backtest", fontsize=14)
    plt.xlabel("Ticks (Minutes)", fontsize=12)
    plt.ylabel("Cumulative Profit / Loss", fontsize=12)
    plt.legend()
    plt.grid(True, alpha=0.2)
    plt.pause(0.1)
    # This forces the window to pop up on Windows
    plt.show() 
    
    print(f"✅ Backtest Complete. Final P/L: {cumulative_reward:.4f}")

# CRITICAL: This line runs the code!
if __name__ == "__main__":
    run_backtest()