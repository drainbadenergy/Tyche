"""
Generates fresh starting weights. Run once after setup.
"""
import os, json, torch, torch.nn as nn

DEVICE    = torch.device("cuda" if torch.cuda.is_available() else "cpu")
OBS_DIM   = 35
N_ACTIONS = 243
os.makedirs("memory", exist_ok=True)

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

# Always build on CPU first, then move — avoids CUDA init issues
net = ActorCritic().cpu()
# Always build on CPU first, then move — avoids CUDA init issues
net = ActorCritic().cpu()

with torch.no_grad():
    # 1. Silence all random noise in the action layer
    torch.nn.init.zeros_(net.actor.weight)
    torch.nn.init.zeros_(net.actor.bias)
    # 2. Make "Hold Everything" (Action 121) the only voice in the room
    # e^15 in a softmax gives it a 99.999% mathematical probability of holding
    net.actor.bias[121] = 15.0 
opt = torch.optim.Adam(net.parameters(), lr=3e-4)

torch.save({
    "net": net.state_dict(),
    "opt": opt.state_dict(),
    "episode": 0,
    "total_steps": 0,
    "best_pnl": -1e9,
}, "memory/agent_weights.pt")


torch.save(net.state_dict(), "memory/best_weights.pt")

with open("memory/training_log.json", "w") as f:
    json.dump([], f)

import json as _json
status = {
    "running": False, "episode": 0, "step": 0,
    "portfolio_value": 10000.0, "pnl_usd": 0.0, "pnl_pct": 0.0,
    "n_trades": 0, "win_rate": 0.0, "mode": "NORMAL",
    "prices": {"BTCUSDT":45000.0,"ETHUSDT":2500.0,"BNBUSDT":300.0,"SOLUSDT":100.0,"ADAUSDT":0.5},
    "best_pnl": 0.0, "total_steps": 0, "last_update": "",
}
with open("memory/trainer_status.json", "w") as f:
    _json.dump(status, f)
with open("memory/recent_trades.json", "w") as f:
    _json.dump([], f)

params = sum(p.numel() for p in net.parameters())
print(f"[BOOTSTRAP] Done — {params:,} params | device available: {DEVICE}")
