"""
Tyche v7 — Persistent Trainer
"""
import json, os, time, threading
import numpy as np
from datetime import datetime
from .config      import INITIAL_CASH, ASSETS
from .data_loader import load_all, align_data
from .environment import HFTEnv
from .agent_gpu   import TycheAgent
from .mongo_store import log_trade, log_episode
from .adversarial import AdversarialEngine
from .mongo_store import _db as db, _run_id
from datetime import datetime

STATUS_PATH = "memory/trainer_status.json"
TRADE_PATH  = "memory/recent_trades.json"

LIFETIME_PNL    = 0.0
LIFETIME_TRADES = 0
PNL_HISTORY     = []   # [{steps, pnl}]
_RETURNS_BUF    = []   # for Sharpe
PNL_BUFFER = []    # ⚡ FIX: This MUST be here

_lock   = threading.Lock()
_status = {
    "running": False, "episode": 0, "step": 0,
    "total_steps": 0,
    "portfolio_value": INITIAL_CASH,
    "pnl_usd": 0.0, "pnl_pct": 0.0,
    "n_trades": 0, "win_rate": 0.0,
    "sharpe": 0.0, "drawdown": 0.0,
    "mode": "NORMAL",
    "prices": {a: 0.0 for a in ASSETS},
    "best_pnl": 0.0, "last_update": "",
}
_recent_trades = []


def get_status():
    with _lock:
        return dict(_status)

def get_pnl_history():
    with _lock:
        return list(PNL_HISTORY)

def get_recent_trades(n=50):
    with _lock:
        return list(_recent_trades[-n:])


def _write_status():
    os.makedirs("memory", exist_ok=True)
    try:
        with _lock:
            s = dict(_status)
            t = list(_recent_trades[-200:])
        with open(STATUS_PATH, "w") as f: json.dump(s, f)
        with open(TRADE_PATH,  "w") as f: json.dump(t, f)
    except:
        pass


def _compute_sharpe():
    """Rolling Sharpe on PNL_HISTORY deltas."""
    if len(PNL_HISTORY) < 10:
        return 0.0
    vals = [p["pnl"] for p in PNL_HISTORY[-200:]]
    diffs = np.diff(vals)
    std = np.std(diffs)
    if std < 1e-8:
        return 0.0
    return float(np.clip((np.mean(diffs) / std) * np.sqrt(31_536_000), -99, 99))


def run_training(n_episodes=999_999):
    # ⚡ 1. DECLARE ALL GLOBALS AT THE TOP (Fixes UnboundLocalError)
    global LIFETIME_PNL, LIFETIME_TRADES, PNL_HISTORY, PNL_BUFFER

    try:
        from .mongo_store import get_latest_lifetime_pnl
        # This pulls the total P&L from your last MongoDB entry
        LIFETIME_PNL = get_latest_lifetime_pnl()
        print(f" [RECOVERY] Resuming Wallet: ${LIFETIME_PNL:+,.2f}")
    except Exception as e:
        print(f" [RECOVERY ERROR] Could not load wallet: {e}")
        LIFETIME_PNL = 0.0

    raw       = load_all()
    data, _   = align_data(raw)
    env       = HFTEnv(data)
    agent     = TycheAgent()
    # adv     = AdversarialEngine(p_event=0.003) # Uncomment if using

    with _lock:
        _status["running"] = True

    for ep in range(agent.episode, agent.episode + n_episodes):
        obs, _ = env.reset()
        agent.new_rollout()
        done, step, t0 = False, 0, time.time()

        while not done:
            action, log_p, val = agent.get_action(obs)
            next_obs, reward, done, _, info = env.step(action)
            agent.record(obs, action, log_p, val, reward, done)

            pv     = info["portfolio_value"]
            ep_pnl = pv - INITIAL_CASH

            # ── TRADE LOGGING ──────────────────────────────
            if info["trade_occurred"]:
                for td in info["trade_log"]:
                    asset = ASSETS[td["asset_idx"]]
                    side  = td["side"]
                    pnl_v = float(td["pnl"]) # Keep full precision

                    log_trade(ep, env.step_idx, asset, side,
                              td["qty"], td["price"], pnl_v, pv)

                    doc = {
                        "ts":      datetime.utcnow().isoformat(),
                        "episode": ep,
                        "step":    env.step_idx,
                        "asset":   asset,
                        "action":  side,
                        "price":   round(td["price"], 5),
                        "qty":     round(td["qty"], 8),
                        # ⚡ FIXED PRECISION: Use 6 decimals so sub-cent profits show up
                        "pnl_usd": None if side == "BUY" else round(pnl_v, 6),
                        "portfolio_value": round(LIFETIME_PNL + INITIAL_CASH + ep_pnl, 2),
                    }
                    with _lock:
                        _recent_trades.append(doc)
                        if len(_recent_trades) > 500: del _recent_trades[:100]

            # ── STATUS UPDATE every 100 steps ──────────────
            if step % 100 == 0:
                try:
                    pnl_usd = pv - INITIAL_CASH
                    display_pnl = LIFETIME_PNL + pnl_usd
                    display_value = INITIAL_CASH + display_pnl
                    
                    # ⚡ THE FIX: Get prices from 'info' instead of 'env'
                    current_prices = info["prices"] 
                    
                    # Update global buffers
                    PNL_BUFFER.append(display_pnl)
                    PNL_HISTORY.append({"steps": agent.total_steps + step, "pnl": round(display_pnl, 2)})
                    
                    if len(PNL_BUFFER) > 1000: PNL_BUFFER.pop(0)
                    if len(PNL_HISTORY) > 1000: PNL_HISTORY.pop(0)

                    # Sharpe Math
                    live_sharpe = 0.0
                    if len(PNL_BUFFER) > 20:
                        diffs = np.diff(PNL_BUFFER[-100:])
                        std = np.std(diffs)
                        if std > 1e-6:
                            live_sharpe = (np.mean(diffs) / std) * np.sqrt(31536000)

                    # Build the price dictionary for the UI
                    price_dict = {ASSETS[i]: float(current_prices[i]) for i in range(len(ASSETS))}

                    
                    with _lock:
                        # print(f"DEBUG: Price Dict -> {price_dict}") # ⚡ Temporary check
                        _status.update({
                            "episode": ep,
                            "portfolio_value": round(display_value, 2),
                            "pnl_usd": round(display_pnl, 2),
                            "n_trades": LIFETIME_TRADES + info["n_trades"],
                            "win_rate": round(info["win_rate"] * 100, 1),
                            "sharpe": round(live_sharpe, 2),
                            "drawdown": round(env.max_drawdown * 100, 2),
                            "prices": price_dict, # Now it has the real numbers
                            "total_steps": agent.total_steps + step,
                        })
                    _write_status()
                except Exception as e:
                    print(f" [STATS ERROR] {e}")

            # Inside trainer.py, in the "while not done" loop:

            if step % 500 == 0:
                # Save a temporary "Live State" to a special collection
                db["live_checkpoint"].update_one(
                    {"run_id": _run_id},
                    {"$set": {
                        "pnl_usd": display_pnl,
                        "ts": datetime.utcnow()
                    }},
                    upsert=True
                )

            if step > 0 and step % 4096 == 0:
                agent.update()
                agent.new_rollout()

            obs, step = next_obs, step + 1

        # ── EPISODE END ──
        pnl_final = env._pv() - INITIAL_CASH
        LIFETIME_PNL    += pnl_final
        LIFETIME_TRADES += env.n_trades

        # ── EPISODE END ────────────────────────────────────
        pnl_final = env._pv() - INITIAL_CASH
        LIFETIME_PNL    += pnl_final
        LIFETIME_TRADES += env.n_trades

        agent.update()
        agent.episode = ep + 1
        agent.save(pnl_final)

        log_episode(ep, pnl_final, pnl_final / INITIAL_CASH,
                    env.n_trades, env.n_wins / max(1, env.n_trades),
                    env.max_drawdown, step)

        elapsed = time.time() - t0
        print(f"[EP {ep:5d}]  P&L ${pnl_final:+9.2f}  "
              f"trades {env.n_trades:4d}  "
              f"win {env.n_wins/max(1,env.n_trades)*100:.1f}%  "
              f"steps {step:6,d}  {elapsed:.1f}s  "
              f"best ${_status['best_pnl']:+.2f}  "
              f"LIFETIME ${LIFETIME_PNL:+,.2f}")

    with _lock:
        _status["running"] = False