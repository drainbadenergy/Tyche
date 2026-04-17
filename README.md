# Tyche v7 — Crypto HFT Research System

**1-second bars · PyTorch PPO · MongoDB · Streamlit**


## What the dashboard shows

| Panel | Refresh |
|-------|---------|
| P&L (`$10,000 → $X`) | every 1 second |
| Live trade feed (BUY/SELL + exact $ P&L) | every 1 second |
| Live prices (5 assets) | every 1 second |
| MongoDB trade count | every 5 seconds |
| Episode P&L chart | every 10 seconds |
| Adversary mode (FLASH CRASH etc.) | every 1 second |

## Architecture

```
Binance S3 (1-second OHLCV)
        ↓
  data_loader.py  →  tyche/environment.py  (HFTEnv)
                              ↓
                   tyche/agent_gpu.py  (PPO, RTX 4070)
                              ↓
                   tyche/trainer.py  (episode loop, persistent weights)
                              ↓
          ┌────────────────────────────────────┐
          │  memory/trainer_status.json        │
          │  memory/recent_trades.json         │
          │  memory/agent_weights.pt           │
          │  MongoDB (trades, episodes)        │
          └────────────────────────────────────┘
                              ↓
                         server.py  (Flask :5000)
                              ↓
                      dashboard_app.py  (Streamlit :8501)
```

## File map

```
tyche_v7/
├── setup.bat               one-click install
├── train_overnight.bat     overnight training
├── start.bat               launch dashboard
├── download_data.py        Binance S3 1-second downloader
├── generate_weights.py     bootstrap agent weights
├── main.py                 training entry point
├── server.py               Flask REST API (:5000)
├── dashboard_app.py        Streamlit UI (:8501)
├── requirements.txt
├── memory/
│   ├── agent_weights.pt    latest weights (auto-saved)
│   ├── best_weights.pt     best episode weights
│   ├── training_log.json   episode history
│   ├── trainer_status.json live status (read by dashboard)
│   └── recent_trades.json  last 200 trades
└── tyche/
    ├── config.py
    ├── data_loader.py      loads parquet or synthetic GBM
    ├── environment.py      HFTEnv (OBS_DIM=35, N_ACTIONS=243)
    ├── agent_gpu.py        PyTorch actor-critic PPO
    ├── trainer.py          episode loop
    ├── adversarial.py      5-mode stress engine
    └── mongo_store.py      MongoDB + memory fallback
```

## Key numbers

| Parameter | Value |
|-----------|-------|
| Starting capital | $10,000 |
| Assets | BTC ETH BNB SOL ADA |
| Data granularity | 1 second |
| Observation dim | 35 |
| Actions | 243 (3^5 per-asset) |
| PPO clip ε | 0.2 |
| Learning rate | 3×10⁻⁴ |
| Trade fee | 0.1% (Binance taker) |
| Execution latency | ~2s mean |
