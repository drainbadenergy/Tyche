import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd

class TycheTradeEnv(gym.Env):
    def __init__(self):
        super(TycheTradeEnv, self).__init__()
        # 1. Load data and handle missing values immediately
        raw_df = pd.read_csv("data/market_prices.csv", index_col=0).ffill().bfill()
        
        # 2. Normalize and CLIP percentages (Prevents huge spikes)
        self.df = raw_df.pct_change().fillna(0).clip(-0.05, 0.05)
        
        self.action_space = spaces.Discrete(3)
        # 10 features (5 Price % + 5 Volume %)
        self.observation_space = spaces.Box(low=-1, high=1, shape=(10,), dtype=np.float32)
        self.current_step = 0

    def step(self, action):
        latency = 2
        exec_idx = self.current_step + latency
        
        if exec_idx >= len(self.df):
            return np.zeros(10, dtype=np.float32), 0.0, True, False, {}

        # 3. Stable Reward Calculation
        # We use a 10x multiplier (down from 100x) to avoid 'NaN'
        future_return = self.df.iloc[exec_idx].values[0] * 10.0
        
        reward = 0.0
        if action == 2:   # Buy
            reward = float(future_return)
        elif action == 0: # Sell
            reward = float(-future_return)
        else:             # Hold
            reward = -0.001 # Small penalty
            
        # 4. Final Reward Clipping (The 'Anti-NaN' Guard)
        reward = np.clip(reward, -1.0, 1.0)

        self.current_step += 1
        done = self.current_step >= len(self.df) - 5
        
        obs = self.df.iloc[self.current_step].values.astype(np.float32)
        # Final safety check: replace any NaN in observation with 0
        obs = np.nan_to_num(obs)
        
        return obs, reward, done, False, {}

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        obs = self.df.iloc[self.current_step].values.astype(np.float32)
        return np.nan_to_num(obs), {}