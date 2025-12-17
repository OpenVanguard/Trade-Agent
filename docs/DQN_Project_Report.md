# Deep Q-Learning (DQN) Stock Trading Project — Detailed Report ✅

**Generated:** 2026-01-06

---

## 1. Executive Summary 💡
This project implements a Deep Q-Learning (DQN) system to trade equities using historical daily CSV data. The pipeline loads CSV price data, computes extensive technical indicators (TA-Lib), constructs an environment that simulates trading with transaction costs and position limits, and trains a Deep Q-Network agent to learn a buy/hold/sell policy via experience replay and target network updates.

Key facts:
- Training has been done on daily data only (not intraday).
- Observations include price + technical indicators (54 features) plus 5 normalized portfolio features → total state dimension = 59.
- Action space: discrete {0: HOLD, 1: BUY, 2: SELL}.

---

## 2. Data & Features 🔧
- Source files: `data/SYMBOL_daily.csv` (e.g. `AMZN_daily.csv`, `AAPL_daily.csv`).
- Data cleaning: removes rows with non-positive or NaN prices and volume; metadata logs removed rows.
- Indicators: wide set computed using TA-Lib and custom logic (SMA, EMA, RSI, MACD, STOCH, Bollinger bands, ATR, NATR, OBV, VWAP approx, candle patterns, multi-horizon returns, etc.).
- Feature count: 54 market features (see `TechnicalIndicators.get_feature_columns()`), plus normalized portfolio fields (balance, shares, net_worth, position_value, portfolio_return) → observation vector length 59.
- Normalization: `MinMaxScaler` is used; note: currently the code sometimes fits scaler on test data during validation (`fit_scaler=True`) — this risks leaking test information (see Recommendations).

---

## 3. Environment & Reward Design 🎯
File: `src/trading_environment.py`

- Portfolio simulation: initial balance (default $10,000), transaction cost (default 0.1%), position sizing limit (max 10% of initial balance per buy), max_shares parameter.
- Actions: Hold/Buy/Sell. Buy executes affordable shares capped by position limits; Sell liquidates all shares held.
- Reward shaping (per step):
  - Small buy bonus: +0.01 when a buy occurs
  - Sell reward: +0.02 if sell results in profit (based on average buy price), -0.01 when selling at loss
  - Scaled portfolio change: reward += portfolio_return * 10 (captures realized + unrealized PnL)
  - Trading penalty: -0.01 for any trade (discourage excessive trading)
  - New high net worth bonus: +0.05 when net worth exceeds prior max

- Performance metrics tracked: total_return, win_rate, max_drawdown, Sharpe ratio (daily), volatility, total_trades.

---

## 4. DQN Agent Architecture & Algorithms 🧠
File: `src/dqn_model.py`

- Network architecture (default):
  - Input: state_dim (59)
  - Dense 1024 → LayerNorm → ReLU → Dropout(0.2)
  - Dense 512 → LayerNorm → ReLU → Dropout(0.15)
  - Dense 256 → LayerNorm → ReLU → Dropout(0.1)
  - Dense 128 → LayerNorm → ReLU → Dropout(0.05)
  - Dense 64 → ReLU
  - Output: action_dim (3 Q-values)

- Algorithmic choices:
  - Epsilon-greedy exploration (ε start=1.0, decay applied per replay; trainer config uses ε decay 0.999 and ε_min 0.01).
  - Double-DQN style target estimation: next-actions selected by online network, target values from target network.
  - Target network updated periodically (`agent.update_target_network()`), default every 10 episodes in training.
  - Experience replay buffer with thread-safe access; batch-based training on GPU when available.
  - Optimization: Adam with weight decay, gradient clipping (max_norm=1.0), MSE loss on Q-values.

---

## 5. Training Loop & Hyperparameters ⚙️
File: `src/training.py`

- High-level process:
  - Prepare features (indicators) → normalize (MinMaxScaler) → create `StockTradingEnvironment` → instantiate `DQNAgent`
  - Episodes (default 1000), each episode runs through dataset once as an episode.
  - Per step: agent acts, environment steps, experience stored, and learning occurs repeatedly (the code replays multiple mini-batches per step to accelerate convergence).
  - Save best models and periodic checkpoints (every `save_freq` episodes).
- Notable hyperparameters:
  - Learning rate: 1e-4
  - Gamma (discount): 0.99 (passed from trainer)
  - Epsilon decay: 0.999, min 0.01
  - Batch size: 512 (if GPU) else 128
  - Replay memory: 40k (GPU) else 20k
  - Replay frequency: replay called per step and up to 3 replays per step when memory sufficient
  - Target network update: every 10 episodes

---

## 6. Validation & Backtesting 🔍
File: `src/validation.py`

- Backtest procedure:
  - Load best or specified model → re-run environment on last 20% (default) of prepared data
  - Normalize test portion (currently often fit on test data) — caution: this may leak information, see Recommendations.
  - Record actions, portfolio evolution; compute metrics (final balance, total return, Sharpe, max drawdown, win rate, trade counts)
  - Save a JSON backtest summary to `validation_logs/backtest_SYMBOL_TIMESTAMP.json`

- Additional utilities: walk-forward analysis, model comparison (best vs final), automated report creation.

---

## 7. Results & Observations (from repository artifacts) 📈
- The repo contains many saved models and training histories (e.g., `models/AMZN_*`, `logs/*_training_history_*.json`) indicating multiple training runs and saved checkpoints.
- README suggests example backtest numbers (illustrative): AMZN daily: total return ~907% (final balance ~$100k), Sharpe ~1.25 — these figures appear to be demonstrative benchmarks in README and may not be reproducible without the exact model and dataset splits.

Observations and likely behaviors:
- Reward shaping mixes realized + unrealized returns with small action bonuses/penalties; this can encourage both capturing positive drift and avoiding churn.
- Large network capacity and aggressive replay (multi-replay per step) can lead to fast convergence, but also risks overfitting on limited daily data.
- Training on daily data only reduces sample frequency; model may underperform on intraday regimes if ported directly.

---

## 8. Strengths & Current Limitations ⚖️
Strengths:
- Comprehensive feature set (wide technical indicators).
- Realistic environment with transaction costs and position limits.
- GPU-aware training and careful implementation details (Double DQN, LayerNorm, gradient clipping).
- Solid logging, model checkpointing and validation/backtest utilities.

Limitations:
- Training only on daily data (no intraday models trained yet).
- Potential data leakage: scaler fitting on test in some validation paths (should use training-fitted scaler).
- Sell action liquidates entire position (no partial sizing or continuous position size decisions).
- Reward uses portfolio_return scaling (fast feedback) but may conflate realized vs unrealized PnL.
- No explicit modeling of slippage or realistic commission beyond a fixed transaction_cost.

---

## 9. Recommendations & Future Enhancements 🚀
Immediate fixes:
1. Persist and load the scaler used in training; **do not** fit scaler on test data. Use the training-fitted scaler for validation and backtests.
2. Split normalization/feature pipeline so that `train` saves scaler to `models/` and `validate` loads it.
3. Log realized vs unrealized PnL separately and report both in backtests.

Algorithmic / Architecture improvements:
- Try Dueling DQN and Prioritized Experience Replay to improve sample efficiency.
- Consider distributional RL or multi-step returns (n-step targets) for stability in financial signals.
- Explore actor-critic (PPO/A2C) or parameterized action spaces for position sizing instead of discrete all-in/all-out trades.
- Add action that allows partial sells/buys or continuous fraction-sizing (or discretize sizes e.g., buy 10%, 20% etc.).

Risk & robustness:
- Add slippage modeling and variable commission per trade (simulate more conservative returns).
- Implement early stopping criteria or evaluation holdout and cross-validation via walk-forward to reduce overfitting.
- Add seed control and deterministic training logging for reproducibility.

Monitoring & evaluation:
- Persist training histories, model hyperparameters, and randomness seeds in a reproducible artifact (e.g., `models/{symbol}_metadata.json`).
- Add more evaluation metrics: realized PnL, profit factor, tail-risk measures, drawdown duration, trade duration distributions.

Operational:
- Train and validate intraday models with appropriately downsampled features (or use intraday features) if intraday trading is desired.
- Add a simple CLI parameter to run validations with more realistic cost assumptions (slippage bps, fixed commission).

---

## 10. Reproducibility & How to Run 🧭
- Train (daily):
  - python src/main.py train --symbol AMZN --episodes 1000 --freq daily
- Validate (use best model):
  - python src/main.py validate --symbol AMZN
- Validate with explicit model:
  - python src/main.py validate --symbol AMZN --model-path models/AMZN_best_episode_912.pth
- Plot training progress:
  - Training automatically saves `logs/{SYMBOL}_training_history_*.json` and the trainer can plot them.

---

## 11. Where to look in the codebase 🔎
- Model: `src/dqn_model.py`
- Environment: `src/trading_environment.py`
- Training loop: `src/training.py`
- Indicators & features: `src/technical_indicators.py`
- Validation/backtesting: `src/validation.py`
- Entrypoint CLI: `src/main.py`

---

## 12. Short Summary ✅
This repo implements a production-grade DQN trading pipeline for daily equities with an extensive technical feature set and careful environment modeling. The system is flexible and GPU-aware, but improvements are advised around data pipeline reproducibility (scaler persistence), reward clarity (realized vs unrealized), and enhanced RL algorithms or action representations to allow partial position sizing and better generalization.

---

If you'd like, I can:
- Generate a short results-focused executive report only, or
- Implement the recommended scaler save/load fix and run a validation (requires your consent to run training/validation locally).

