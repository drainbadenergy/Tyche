import time
from pymongo import MongoClient
from datetime import datetime

# Connect directly to your local Mongo
client = MongoClient("mongodb://localhost:27017/")
db = client["tyche_v7"]

# Create a "Recovery" record
recovery_data = {
    "run_id": f"recovery_{int(time.time())}",
    "episode": 999, 
    "pnl_usd": 5000.0,
    "pnl_pct": 0.50,
    "n_trades": 0,
    "win_rate": 1.0,
    "max_drawdown": 0.0,
    "steps": 0,
    "ts": datetime.utcnow().isoformat()
}

# Insert into the episodes collection
db["episodes"].insert_one(recovery_data)

print("💰 [SUCCESS] Manual recovery point created at $5,000.00")
print("🚀 Restart main.py to see your wallet update.")