"""
Run once after setup.bat to generate pre-trained starting weights.
Called automatically by train_overnight.bat on first run.
"""
import os, json, torch, torch.nn as nn

os.makedirs("memory", exist_ok=True)

OBS_DIM = 35
N_ACTIONS = 243

class AC(nn.Module):
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

net = AC()
# Warm-start: slightly prefer HOLD to stop random catastrophic early trades
with torch.no_grad():
    net.actor.bias[121] += 1.5   # 121 = all-hold in base-3 encoding

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

with open("memory/trainer_status.json", "w") as f:
    json.dump({
        "running": False, "episode": 0, "step": 0,
        "portfolio_value": 10000.0, "pnl_usd": 0.0, "pnl_pct": 0.0,
        "n_trades": 0, "win_rate": 0.0, "mode": "NORMAL",
        "prices": {"BTCUSDT":45231.0,"ETHUSDT":2481.0,"BNBUSDT":312.0,"SOLUSDT":98.0,"ADAUSDT":0.52},
        "best_pnl": 0.0, "total_steps": 0, "last_update": "",
    }, f)

with open("memory/recent_trades.json", "w") as f:
    json.dump([], f)

print(f"[OK] Weights generated  |  Params: {sum(p.numel() for p in net.parameters()):,}")
print("[OK] memory/ folder ready")
