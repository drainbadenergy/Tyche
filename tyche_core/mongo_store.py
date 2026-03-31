import time, os, threading
from datetime import datetime


def mongo_stats():
    # ⚡ FORCE Flask to ask MongoDB for the latest numbers
    return jsonify(get_stats())

def trades():
    # ⚡ FORCE Flask to pull the latest trades from the DB
    return jsonify(get_recent_trades(n=50))

def episodes():
    # ⚡ FORCE Flask to pull the history from the DB
    return jsonify(get_episode_history(n=100))

# --- CONFIG & INITIALIZATION ---
MONGO_OK = False
_db      = None
_lock    = threading.Lock()

try:
    from pymongo import MongoClient
    # Connect with a 2-second timeout so the bot doesn't hang if Mongo is off
    _client = MongoClient("mongodb://localhost:27017/", serverSelectionTimeoutMS=2000)
    _client.server_info() # Trigger connection check
    _db = _client["tyche_v7"]
    MONGO_OK = True
    print("  [MONGO] Persistence Active — Linked to local tyche_v7")
except Exception:
    print("  [MONGO] Persistence Offline — Using Memory Fallback")

# Unique ID for this specific trading session
_run_id          = f"run_{int(time.time())}"
_memory_trades   = []
_memory_episodes = []

# --- CORE LOGGING FUNCTIONS ---

def log_trade(episode, step, asset, side, qty, price, pnl_usd, portfolio_value):
    """Saves a single trade to MongoDB and local memory."""
    doc = {
        "run_id": _run_id, 
        "episode": episode, 
        "step": step,
        "asset": asset, 
        "side": side.upper(),
        "qty": round(float(qty), 6), 
        "exec_price": round(float(price), 4),
        "pnl_usd": round(float(pnl_usd), 4) if pnl_usd is not None else None,
        "portfolio_value": round(float(portfolio_value), 2),
        "ts": datetime.utcnow().isoformat(),
    }
    
    if MONGO_OK:
        try:
            _db["trades"].insert_one(doc)
            doc.pop("_id", None) # Remove Mongo ID for memory storage
        except: pass
        
    with _lock:
        _memory_trades.append(doc)
        if len(_memory_trades) > 500: _memory_trades.pop(0)

def log_episode(episode, pnl_usd, pnl_pct, n_trades, win_rate, max_dd, steps):
    """Saves an entire episode summary for the history table."""
    doc = {
        "run_id": _run_id, 
        "episode": episode,
        "pnl_usd": round(float(pnl_usd), 2),
        "pnl_pct": round(float(pnl_pct), 4),
        "n_trades": int(n_trades),
        "win_rate": round(float(win_rate), 4),
        "max_drawdown": round(float(max_dd), 2),
        "steps": int(steps),
        "ts": datetime.utcnow().isoformat(),
    }
    
    if MONGO_OK:
        try:
            _db["episodes"].insert_one(doc)
            doc.pop("_id", None)
        except: pass
        
    with _lock:
        _memory_episodes.append(doc)

# --- DASHBOARD DATA PROVIDERS ---

def get_stats(live_pnl=0.0): # ⚡ Add live_pnl as a parameter
    if MONGO_OK:
        try:
            total_history_trades = _db["trades"].count_documents({})
            total_eps = _db["episodes"].count_documents({})
            
            # Find the best COMPLETED episode in the DB
            best_doc = _db["episodes"].find_one({}, sort=[("pnl_usd", -1)])
            db_best = best_doc["pnl_usd"] if best_doc else 0.0
            
            # ⚡ The Magic Logic: Use the higher of the two
            # This ensures the box updates the second your live profit hits a new high
            absolute_best = max(db_best, live_pnl)
            
            return {
                "n_trades": _db["trades"].count_documents({"run_id": _run_id}),
                "total_history_trades": total_history_trades,
                "n_episodes": total_eps,
                "best_pnl": absolute_best, 
                "source": "mongodb"
            }
        except Exception as e:
            print(f" [STATS ERROR] {e}")
            
    return {
        "n_trades": len(_memory_trades),
        "total_history_trades": len(_memory_trades),
        "n_episodes": len(_memory_episodes),
        "best_pnl": live_pnl, # Fallback to live profit if Mongo is down
        "source": "memory",
    }

def get_recent_trades(n=50):
    if MONGO_OK:
        try:
            # ⚡ FIX: Remove the run_id filter to see all recent activity
            docs = list(_db["trades"].find({}, {"_id":0}).sort("ts", -1).limit(n))
            return list(reversed(docs))
        except: pass
    return _memory_trades[-n:]

def get_episode_history(n=200):
    if MONGO_OK:
        try:
            # ⚡ FIX: Remove the run_id filter
            docs = list(_db["episodes"].find({}, {"_id":0}).sort("episode", -1).limit(n))
            return list(reversed(docs))
        except: pass
    return _memory_episodes[-n:]

def get_latest_lifetime_pnl():
    if MONGO_OK:
        try:
            # Sort by timestamp (ts) descending to get the ABSOLUTE last state
            last_ep = _db["episodes"].find_one(sort=[("ts", -1)])
            if last_ep:
                return float(last_ep.get("pnl_usd", 0.0))
        except Exception as e:
            print(f" [DB RECOVERY ERROR] {e}")
            pass
    return 0.0