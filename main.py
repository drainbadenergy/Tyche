"""
Tyche v7 — Main entry point
"""
import threading, time, argparse
#from tyche.shared import _status

import threading, time, argparse, sys

# 🛠️ THE EMERGENCY OVERRIDE
try:
    from tyche.shared import _status
except ImportError:
    print("⚠️ [PATCH] Forcing shared state injection...")
    import tyche.shared as shared
    _status = {
        "pnl_usd": 0.0,
        "steps": 0,
        "trades": 0,
        "win_rate": 0.0
    }
    shared._status = _status





def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--episodes", type=int, default=999999)
    parser.add_argument("--no-server", action="store_true")
    args = parser.parse_args()

    print("=" * 60)
    print("  TYCHE v7 — Crypto HFT System")
    print("  1-second bars | PyTorch PPO | CUDA")
    print("=" * 60)

    if not args.no_server:
        def _run_flask():
            from server import app
            from tyche.mongo_store import get_stats
            s = get_stats()
            print(f"[SERVER] Starting on :5000 | DB: {s['source'].upper()}")
            app.run(host="0.0.0.0", port=5000, debug=False, use_reloader=False)
        t = threading.Thread(target=_run_flask, daemon=True)
        t.start()
        time.sleep(2)
        print("[MAIN] Flask server running on http://localhost:5000")

    from tyche.trainer import run_training
    run_training(n_episodes=args.episodes)

if __name__ == "__main__":
    main()
