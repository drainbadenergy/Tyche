# ⚡ Tyche

> An AI-powered Reinforcement Learning framework for high-frequency trading research and market simulation.

Tyche is a research-focused quantitative trading system that trains a PPO (Proximal Policy Optimization) reinforcement learning agent on high-frequency cryptocurrency market data. The project includes a complete training pipeline, backtesting engine, real-time monitoring dashboard, persistent model storage, and market stress testing for evaluating trading strategies under different conditions.

---

# Features

- 📈 Reinforcement Learning based trading agent (PPO)
- ⚡ High-frequency trading environment using 1-second OHLCV data
- 📊 Interactive real-time dashboard built with Streamlit
- 🧠 Automatic model checkpointing and persistent training
- 💾 MongoDB integration for trade and episode storage
- 📉 Historical backtesting engine
- 🌪️ Adversarial market simulation (flash crashes, volatility spikes, etc.)
- 📦 Modular architecture for experimentation
- 🚀 GPU acceleration with PyTorch (CUDA supported)

---

# Architecture

```
                 Historical Market Data
                          │
                          ▼
                Data Download & Loader
                          │
                          ▼
             Custom Trading Environment
                          │
                          ▼
               PPO Reinforcement Agent
                          │
                          ▼
              Training / Learning Loop
                          │
         ┌────────────────┴────────────────┐
         │                                 │
         ▼                                 ▼
 Model Checkpoints                  Trade History
 Training Logs                      MongoDB Storage
         │                                 │
         └────────────────┬────────────────┘
                          ▼
                Flask REST API Server
                          │
                          ▼
               Streamlit Live Dashboard
```

---

# Project Structure

```
Tyche/
│
├── dashboard_app.py          # Live monitoring dashboard
├── server.py                 # Flask API
├── main.py                   # Training entry point
├── backtest.py               # Performance evaluation
├── download_data.py          # Dataset downloader
├── data_engine.py            # Market data processing
├── generate_weights.py       # Initial model generation
├── bootstrap_weights.py      # Weight initialization
├── restore.py                # Restore checkpoints
├── inject.py                 # Utility functions
├── requirements.txt
│
├── memory/
│   ├── agent_weights.pt
│   ├── best_weights.pt
│   ├── training_log.json
│   ├── trainer_status.json
│   └── recent_trades.json
│
└── tyche/
    ├── trainer.py
    ├── environment.py
    ├── agent_gpu.py
    ├── adversarial.py
    ├── mongo_store.py
    ├── data_loader.py
    └── config.py
```

---

# Technologies Used

- Python
- PyTorch
- Stable-Baselines3
- Streamlit
- Flask
- MongoDB
- Pandas
- NumPy
- Matplotlib
- yFinance
- Binance Historical Data

---

# Trading Environment

The environment simulates high-frequency cryptocurrency trading using:

- 1-second OHLCV candles
- Portfolio management
- Transaction fees
- Position sizing
- Multiple tradable assets
- Reward shaping
- Observation vectors
- Action space optimized for RL

---

# Dashboard

The Streamlit dashboard provides live visualization of:

- Portfolio value
- Current profit/loss
- Agent status
- Trade history
- Asset prices
- Episode statistics
- Database connectivity
- Recent trading activity

---

# Reinforcement Learning

Tyche uses Proximal Policy Optimization (PPO) to train an autonomous trading agent.

Training pipeline includes:

- Environment interaction
- Reward optimization
- Policy updates
- Automatic checkpoint saving
- Continuous learning
- GPU acceleration (CUDA)

---

# Backtesting

The backtesting module allows trained models to be evaluated on historical market data.

Performance metrics include:

- Equity curve
- Cumulative profit
- Episode reward
- Portfolio growth
- Historical trade replay

---

# Data Pipeline

Market data can be collected from:

- Binance historical datasets
- Yahoo Finance
- Custom CSV datasets

Data is processed into feature-rich observations before being fed into the reinforcement learning environment.

---

# Installation

Clone the repository

```bash
git clone https://github.com/yourusername/Tyche.git
cd Tyche
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

# Running the Project

Start training

```bash
python main.py
```

Launch the dashboard

```bash
streamlit run dashboard_app.py
```

Run the Flask API

```bash
python server.py
```

Run a backtest

```bash
python backtest.py
```

---

# Persistent Training

Tyche automatically saves:

- Latest model weights
- Best-performing weights
- Recent trades
- Episode logs
- Training statistics

This allows interrupted training sessions to resume without losing progress.

---

# Research Goals

This project is intended for experimentation in:

- Reinforcement Learning
- Quantitative Finance
- High-Frequency Trading
- Portfolio Optimization
- AI Decision Making
- Market Simulation
- Financial Machine Learning

---

# Disclaimer

This project is intended for educational and research purposes only.

It is **not** financial advice and should not be used for live trading without extensive testing, validation, and risk management.

---

# Future Improvements

- Live paper trading
- Multi-agent reinforcement learning
- Transformer-based market prediction
- Hyperparameter optimization
- Risk-adjusted reward functions
- WebSocket market streaming
- Multi-exchange support
- Docker deployment
- Distributed training

---

# Author

**Ayush Rai**

Computer Science Student • AI & Machine Learning • Quantitative Finance • Reinforcement Learning

---

## If you found this project interesting, consider giving it a ⭐.
