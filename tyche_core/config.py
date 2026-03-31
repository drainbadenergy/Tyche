ASSETS = ["BTCUSDT", "ETHUSDT", "BNBUSDT", "SOLUSDT", "ADAUSDT"]
N_ASSETS = 5
INITIAL_CASH   = 10_000.0
MAX_POSITION_PCT = 0.20
LATENCY_MS_MEAN  = 2.0
LATENCY_MS_STD   = 0.5
TRADE_FEE_PCT    = 0.0001
MONGO_URI = "mongodb://localhost:27017"
MONGO_DB  = "tyche_v7"
FLASK_PORT = 5000
STREAMLIT_PORT = 8501
# tyche/config.py

# ── DATABASE ──
MONGO_URI = "mongodb://localhost:27017/" # Or your MongoDB Atlas string
DB_NAME   = "project_tyche"

# ── EXCHANGE ──
BINANCE_API_KEY = "REPLACE_WITH_YOUR_KEY"
BINANCE_SECRET  = "REPLACE_WITH_YOUR_SECRET"
USE_TESTNET     = True  # Set to False for real money