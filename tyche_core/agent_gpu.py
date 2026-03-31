"""
Tyche v7 — PyTorch CUDA Agent
"""
import os, json
import numpy as np
import torch
import torch.nn as nn
from torch.distributions import Categorical

#DEVICE       = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Change this:
# DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# To this:
DEVICE = torch.device("cpu")
WEIGHTS_PATH = "memory/agent_weights.pt"
BEST_PATH    = "memory/best_weights.pt"
LOG_PATH     = "memory/training_log.json"
OBS_DIM      = 35
N_ACTIONS    = 243
LR           = 3e-4
GAMMA        = 0.995
CLIP_EPS     = 0.2
ENTROPY_C = 0.001  # Force it to stop exploring randomly and focus on the P&L
VALUE_C      = 0.5
BATCH_SIZE   = 2048
N_EPOCHS     = 8

class ActorCritic(nn.Module):
    def __init__(self):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(OBS_DIM, 256), nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 256),     nn.LayerNorm(256), nn.ReLU(),
            nn.Linear(256, 128),     nn.ReLU(),
        )
        self.actor  = nn.Linear(128, N_ACTIONS)
        self.critic = nn.Linear(128, 1)

    def forward(self, x):
        h = self.shared(x)
        return self.actor(h), self.critic(h)

    def get_action(self, obs_np):
        x = torch.FloatTensor(obs_np).unsqueeze(0).to(DEVICE)
        with torch.no_grad():
            logits, val = self(x)
        dist   = Categorical(logits=logits)
        action = dist.sample()
        return int(action.item()), float(dist.log_prob(action).item()), float(val.item())

    def evaluate(self, obs, actions):
        logits, vals = self(obs)
        dist = Categorical(logits=logits)
        return dist.log_prob(actions), vals.squeeze(-1), dist.entropy()


class TycheAgent:
    def __init__(self):
        self.net      = ActorCritic().to(DEVICE)
        self.opt      = torch.optim.Adam(self.net.parameters(), lr=LR)
        self.best_pnl = -1e9
        self.episode  = 0
        self.total_steps = 0
        self._load()
        print(f"  [AGENT] Device: {DEVICE}  Params: {sum(p.numel() for p in self.net.parameters()):,}")
        print(f"  [AGENT] Resuming from episode {self.episode}, total_steps {self.total_steps:,}")

    def new_rollout(self):
        self._obs, self._acts, self._lps = [], [], []
        self._vals, self._rews, self._dones = [], [], []

    def record(self, obs, act, lp, val, rew, done):
        self._obs.append(obs)
        self._acts.append(act)
        self._lps.append(lp)
        self._vals.append(val)
        self._rews.append(rew)
        self._dones.append(done)
        self.total_steps += 1

    def get_action(self, obs):
        return self.net.get_action(obs)

    def update(self):
        n = len(self._rews)
        if n < 32:
            return {}
        # compute returns
        returns = []
        R = 0.0
        for r, d in zip(reversed(self._rews), reversed(self._dones)):
            R = r + GAMMA * R * (1.0 - float(d))
            returns.insert(0, R)
        returns  = np.array(returns, dtype=np.float32)
        vals_np  = np.array(self._vals, dtype=np.float32)
        adv_np   = returns - vals_np
        adv_np   = (adv_np - adv_np.mean()) / (adv_np.std() + 1e-8)

        obs_t  = torch.FloatTensor(np.array(self._obs)).to(DEVICE)
        acts_t = torch.LongTensor(np.array(self._acts)).to(DEVICE)
        old_lp = torch.FloatTensor(np.array(self._lps)).to(DEVICE)
        ret_t  = torch.FloatTensor(returns).to(DEVICE)
        adv_t  = torch.FloatTensor(adv_np).to(DEVICE)

        for _ in range(N_EPOCHS):
            idx = torch.randperm(n)
            for start in range(0, n, BATCH_SIZE):
                b = idx[start:start+BATCH_SIZE]
                new_lp, new_val, ent = self.net.evaluate(obs_t[b], acts_t[b])
                ratio = torch.exp(new_lp - old_lp[b])
                surr  = torch.min(ratio * adv_t[b],
                                  torch.clamp(ratio, 1-CLIP_EPS, 1+CLIP_EPS) * adv_t[b])
                loss  = (-surr.mean()
                         + VALUE_C  * nn.functional.mse_loss(new_val, ret_t[b])
                         - ENTROPY_C * ent.mean())
                self.opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.net.parameters(), 0.5)
                self.opt.step()
        return {}

    def save(self, pnl_usd):
        os.makedirs("memory", exist_ok=True)
        torch.save({
            "net": self.net.state_dict(),
            "opt": self.opt.state_dict(),
            "episode": self.episode,
            "total_steps": self.total_steps,
            "best_pnl": self.best_pnl,
        }, WEIGHTS_PATH)
        if pnl_usd > self.best_pnl:
            self.best_pnl = pnl_usd
            torch.save(self.net.state_dict(), BEST_PATH)
        log = []
        if os.path.exists(LOG_PATH):
            try:
                with open(LOG_PATH) as f:
                    log = json.load(f)
            except: pass
        log.append({"ep": self.episode, "pnl": round(pnl_usd, 2), "steps": self.total_steps})
        with open(LOG_PATH, "w") as f:
            json.dump(log[-500:], f)

    def _load(self):
        if not os.path.exists(WEIGHTS_PATH):
            print("  [AGENT] No weights found! Injecting Sniper Discipline directly...")
            # ⚡ FOOLPROOF SNIPER OVERRIDE ⚡
            with torch.no_grad():
                # 1. Silence all random noise in the action layer
                torch.nn.init.zeros_(self.net.actor.weight)
                torch.nn.init.zeros_(self.net.actor.bias)
                
                # 2. Make "Hold Everything" (Action 121) the only voice in the room
                self.net.actor.bias[121] = 12.0 
            return

        try:
            # We use your file's native 'DEVICE' variable here
            ck = torch.load(WEIGHTS_PATH, map_location=DEVICE, weights_only=False)
            self.net.load_state_dict(ck["net"])
            self.opt.load_state_dict(ck["opt"])
            self.episode      = ck.get("episode", 0)
            self.total_steps  = ck.get("total_steps", 0)
            self.best_pnl     = ck.get("best_pnl", -1e9)
            print(f"  [AGENT] Loaded weights from {WEIGHTS_PATH}")
        except Exception as e:
            print(f"  [AGENT] Could not load weights ({e}) — starting fresh")
