"""
Tyche v7 — HFT Environment
NO pre-computation. Everything on-the-fly. Starts instantly.
OBS_DIM = 35, N_ACTIONS = 243
"""
import numpy as np
import gymnasium as gym
from gymnasium import spaces
from .config import INITIAL_CASH, MAX_POSITION_PCT, TRADE_FEE_PCT, N_ASSETS, ASSETS

OBS_DIM      = 35
N_ACTIONS    = 243
COOLDOWN     = 30   # minimum seconds between trades

class HFTEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, data: dict):
        super().__init__()
        self.asset_names = list(data.keys())
        self.closes  = np.stack([data[s]["close"].values  for s in self.asset_names], axis=1).astype(np.float32)
        self.volumes = np.stack([data[s]["volume"].values for s in self.asset_names], axis=1).astype(np.float32)
        self.n_steps = len(self.closes)

        self.action_space      = spaces.Discrete(N_ACTIONS)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(OBS_DIM,), dtype=np.float32)

        self.pct  = np.clip(np.diff(self.closes,  axis=0) / (self.closes[:-1]  + 1e-8), -0.05, 0.05).astype(np.float32)
        self.vpct = np.clip(np.diff(self.volumes, axis=0) / (self.volumes[:-1] + 1e-8), -1.0,  1.0 ).astype(np.float32)

        self.episode_end = self.n_steps
        self._reset_portfolio()
        print(f"  [ENV] Ready — {self.n_steps:,} steps | {N_ASSETS} assets | OBS_DIM={OBS_DIM}")

    def _reset_portfolio(self):
        self.cash            = float(INITIAL_CASH)
        self.positions       = np.zeros(N_ASSETS, dtype=np.float32)
        self.entry_prices    = np.zeros(N_ASSETS, dtype=np.float32)
        self.peak_value      = float(INITIAL_CASH)
        self.max_drawdown    = 0.0
        self.n_trades        = 0
        self.n_wins          = 0
        self.step_idx        = 0
        self.last_trade_step = 0

    def _pv(self):
        return float(self.cash + np.dot(self.positions, self.closes[min(self.step_idx, self.n_steps - 1)]))

    def _decode(self, action):
        acts = []
        for _ in range(N_ASSETS):
            acts.append(action % 3)
            action //= 3
        return acts

    def _obs(self):
        s      = min(self.step_idx, self.n_steps - 1)
        sp     = min(s, len(self.pct) - 1)
        prices = self.closes[s]
        pv     = self._pv()

        pc = self.pct[sp]
        vc = self.vpct[sp]

        ref = self.closes[max(0, s - 3600)]
        pn  = np.clip(prices / (ref + 1e-8) - 1.0, -0.5, 0.5).astype(np.float32)

        if sp >= 14:
            w      = self.pct[sp-14:sp]
            gains  = np.maximum(w, 0).mean(axis=0)
            losses = np.maximum(-w, 0).mean(axis=0)
            rsi    = (1.0 - 1.0 / (1.0 + gains / (losses + 1e-8))).astype(np.float32)
        else:
            rsi = np.full(N_ASSETS, 0.5, dtype=np.float32)

        if s >= 20:
            w   = self.closes[s-20:s]
            mu  = w.mean(axis=0)
            sig = w.std(axis=0) + 1e-8
            bb  = np.clip((prices - mu) / (2 * sig), -1.0, 1.0).astype(np.float32)
        else:
            bb = np.zeros(N_ASSETS, dtype=np.float32)

        pr = (self.positions * prices / (pv + 1e-8)).astype(np.float32)

        cash_r    = np.float32(self.cash / (pv + 1e-8))
        pnl_pct   = np.float32((pv - INITIAL_CASH) / INITIAL_CASH)
        step_norm = np.float32(self.step_idx / self.n_steps)
        dd_norm   = np.float32(self.max_drawdown / (INITIAL_CASH + 1e-8))
        tr_freq   = np.float32(self.n_trades / max(1, self.step_idx + 1))

        obs = np.concatenate([pc, vc, pn, rsi, bb, pr,
                               [cash_r, pnl_pct, step_norm, dd_norm, tr_freq]]).astype(np.float32)
        return np.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self._reset_portfolio()
        self.step_idx    = int(np.random.randint(20, max(21, self.n_steps // 20)))
        self.episode_end = self.step_idx + 50000
        return self._obs(), {}

    def step(self, action):
        lat      = max(1, int(round(np.random.normal(2, 1))))
        exec_idx = min(self.step_idx + lat, self.n_steps - 1)
        prices   = self.closes[exec_idx]
        pv_before = self._pv()
        acts      = self._decode(action)
        trade_happened = False
        trade_log = []

        cooldown_ok = (self.step_idx - self.last_trade_step) >= COOLDOWN

        for i, act in enumerate(acts):
            p = float(prices[i])
            if p <= 0:
                continue

            if act == 2 and cooldown_ok:
                budget = min(self._pv() * MAX_POSITION_PCT, self.cash * 0.25)
                if budget > 1.0 and self.cash >= budget * (1 + TRADE_FEE_PCT):
                    qty = budget / p
                    self.cash -= budget * (1 + TRADE_FEE_PCT)
                    self.positions[i] += qty
                    self.entry_prices[i] = p
                    self.n_trades += 1
                    trade_happened = True
                    trade_log.append({"asset_idx": i, "side": "BUY", "price": p, "qty": qty, "pnl": 0.0})

            elif act == 0 and cooldown_ok:
                qty = float(self.positions[i])
                if qty > 1e-8:
                    proceeds  = qty * p * (1 - TRADE_FEE_PCT)
                    pnl_trade = proceeds - qty * float(self.entry_prices[i])
                    if pnl_trade > 0:
                        self.n_wins += 1
                    self.cash += proceeds
                    self.positions[i]    = 0.0
                    self.entry_prices[i] = 0.0
                    self.n_trades += 1
                    trade_happened = True
                    trade_log.append({"asset_idx": i, "side": "SELL", "price": p, "qty": qty, "pnl": pnl_trade})

        if trade_happened:
            self.last_trade_step = self.step_idx

        self.step_idx = exec_idx + 1
        pv_after = self._pv()

        if pv_after > self.peak_value:
            self.peak_value = pv_after
        dd = self.peak_value - pv_after
        if dd > self.max_drawdown:
            self.max_drawdown = dd

        reward = float(np.clip((pv_after - pv_before) / INITIAL_CASH * 100.0, -1.0, 1.0))

        done = (self.step_idx >= self.n_steps - 2) or (self.step_idx >= self.episode_end)

        info = {
            "portfolio_value": pv_after,
            "pnl_usd":         pv_after - INITIAL_CASH,
            "n_trades":        self.n_trades,
            "win_rate":        self.n_wins / max(1, self.n_trades),
            "trade_occurred":  trade_happened,
            "trade_log":       trade_log,
            "prices":          prices.tolist(),
            "acts":            acts,
        }
        return self._obs(), reward, done, False, info