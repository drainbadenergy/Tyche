"""
Tyche v7 — Adversarial Stress Engine
5 modes:
  0 = normal
  1 = flash crash   (-7% on BTC in 15 ticks)
  2 = liquidity dry  (volumes drop 90%)
  3 = pump & dump   (+10% then -12% on random asset)
  4 = corr breakdown (assets move independently)
"""
import numpy as np

MODES = {0:"NORMAL", 1:"FLASH CRASH", 2:"LIQUIDITY CRISIS", 3:"PUMP & DUMP", 4:"DECORRELATION"}

class AdversarialEngine:
    def __init__(self, p_event=0.003):
        self.p_event  = p_event  # prob of starting a stress event per step
        self.mode     = 0
        self.ticks_left = 0
        self.target_asset = 0

    def step(self, closes: np.ndarray, volumes: np.ndarray):
        """Potentially modify prices/volumes to stress the environment."""
        if self.ticks_left <= 0:
            if np.random.random() < self.p_event:
                self.mode = np.random.choice([1,2,3,4])
                self.ticks_left = np.random.randint(10, 60)
                self.target_asset = np.random.randint(0, len(closes))
            else:
                self.mode = 0
        else:
            self.ticks_left -= 1

        c = closes.copy()
        v = volumes.copy()

        if self.mode == 1:  # flash crash
            shock = np.random.uniform(-0.005, -0.002)
            c[self.target_asset] *= (1 + shock)
            v[self.target_asset] *= np.random.uniform(3.0, 8.0)  # volume spike

        elif self.mode == 2:  # liquidity dry
            v *= np.random.uniform(0.05, 0.15)

        elif self.mode == 3:  # pump & dump
            phase = self.ticks_left
            if phase > 30:
                c[self.target_asset] *= (1 + np.random.uniform(0.002, 0.006))
            else:
                c[self.target_asset] *= (1 - np.random.uniform(0.004, 0.008))

        elif self.mode == 4:  # decorrelation
            for i in range(len(c)):
                c[i] *= (1 + np.random.normal(0, 0.003))

        return np.clip(c, 0.001, None), np.clip(v, 1.0, None)

    @property
    def mode_name(self):
        return MODES.get(self.mode, "UNKNOWN")

    @property
    def is_stressed(self):
        return self.mode != 0
