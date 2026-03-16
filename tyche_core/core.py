import gymnasium as gym
from gymnasium import spaces
import numpy as np

class TycheEnv(gym.Env):
    """
    Project Tyche: Latency-Aware HFT Environment
    """
    def __init__(self, df, window_size=10):
        super(TycheEnv, self).__init__()
        self.df = df
        self.window_size = window_size
        
        # Actions: 0=Sell, 1=Hold, 2=Buy
        self.action_space = spaces.Discrete(3)
        
        # Observations: OHLCV data + Latency status
        # (Assuming 5 columns of OHLCV + 1 for latency)
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=(window_size, 6), dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = self.window_size
        # Get the first window of data
        obs = self._get_observation()
        return obs, {}

    def step(self, action):
        # 1. Simulate Latency (The Tyche Twist)
        # We simulate a delay by picking a future price instead of the current one
        delay = np.random.randint(1, 5) # 1-5 tick delay
        execution_idx = self.current_step + delay
        
        if execution_idx >= len(self.df):
            return self._get_observation(), 0, True, False, {}

        # 2. Logic for Profit/Loss
        price_now = self.df.iloc[self.current_step]['Close']
        price_exec = self.df.iloc[execution_idx]['Close']
        
        # Simple Reward: Change in price modified by action
        reward = (price_exec - price_now) if action == 2 else 0
        if action == 0: reward = (price_now - price_exec)

        self.current_step += 1
        done = self.current_step >= len(self.df) - 5
        
        return self._get_observation(), reward, done, False, {"delay": delay}

    def _get_observation(self):
        # Slice the dataframe and add a dummy latency column
        obs = self.df.iloc[self.current_step - self.window_size : self.current_step].values
        # Add a column for 'System Jitter' (simulated)
        jitter = np.random.normal(0, 0.1, (self.window_size, 1))
        return np.append(obs, jitter, axis=1).astype(np.float32)