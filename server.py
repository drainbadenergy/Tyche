from flask import Flask, jsonify
from flask_cors import CORS
from tyche.mongo_store import get_stats, get_recent_trades, get_episode_history
from tyche.trainer import get_status, get_recent_trades as get_live_trades
from tyche.shared import _status

app = Flask(__name__)
CORS(app)

@app.route("/status")
def status():
    return jsonify(get_status())

@app.route("/state")
def state():
    return jsonify(get_status())

@app.route("/portfolio")
def portfolio():
    s = get_status()
    return jsonify({
        "portfolio_value": s.get("portfolio_value", 10000),
        "pnl_usd":  s.get("pnl_usd",  0),
        "pnl_pct":  s.get("pnl_pct",  0),
        "n_trades": s.get("n_trades",  0),
        "win_rate": s.get("win_rate",  0),
    })

@app.route("/prices")
def prices():
    return jsonify(get_status().get("prices", {}))

@app.route("/stress")
def stress():
    return jsonify({"mode": get_status().get("mode", "NORMAL")})

@app.route("/agent")
def agent():
    s = get_status()
    return jsonify({
        "episode":     s.get("episode", 0),
        "total_steps": s.get("total_steps", 0),
        "best_pnl":    s.get("best_pnl", 0),
    })

@app.route('/pnl_history')
def route_get_pnl_history(): # ⚡ Renamed to avoid clashing with imports
    try:
        from tyche.trainer import get_pnl_history
        data = get_pnl_history()
        return jsonify(data)
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/trades")
def trades():
    t = get_live_trades(50)
    if not t:
        t = get_recent_trades(50)
    return jsonify(t)


@app.route("/episodes")
def episodes():
    return jsonify(get_episode_history(200))

# server.py

# Ensure this matches where you actually defined the _status dict
from main import _status 

@app.route('/mongo_stats')
def mongo_stats():
    # Now _status is defined and accessible
    current_profit = _status.get("pnl_usd", 0.0)
    return jsonify(get_stats(live_pnl=current_profit))

@app.route("/health")
def health():
    return jsonify({"ok": True})

if __name__ == "__main__":
    print("[SERVER] Starting on :5000")
    app.run(host="0.0.0.0", port=5000, debug=False)
