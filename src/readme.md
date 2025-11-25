
# Trade-Agent: Running Scripts Guide

This guide explains how to run the main scripts for training and validating trading models in this repository. It covers setup, commands, expected outputs, troubleshooting, and tips for best results.

---

## 1. Setup

1. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
2. **Prepare data:**
   - Place your CSVs in the `data/daily/` and `data/intraday/` folders as appropriate.
   - Ensure columns are clean, numeric, and aligned (see troubleshooting below).

---

## 2. Running Validation

### Daily Models
Run for each symbol:
```sh
python src/main.py validate --symbol AMZN --freq daily
python src/main.py validate --symbol MSFT --freq daily
python src/main.py validate --symbol TSLA --freq daily
```

### Intraday (5m) Models
```sh
python src/main.py validate --symbol AMZN --freq intraday
python src/main.py validate --symbol MSFT --freq intraday
python src/main.py validate --symbol TSLA --freq intraday
```

### Custom Model Path
If you want to validate a specific model checkpoint:
```sh
python src/main.py validate --symbol AMZN --model-path models/AMZN_final_episode_1500.pth
python src/main.py validate --symbol GOOGL --model-path models/GOOGL_final_episode_1500.pth
python src/main.py validate --symbol MSFT --model-path models/MSFT_final_episode_1500.pth
python src/main.py validate --symbol TSLA --model-path models/TSLA_final_episode_1500.pth
```

---

## 3. Expected Outputs

- **Backtest logs:**
  - Saved as `validation_logs/backtest_{SYMBOL}_{timestamp}.json`
- **Console summary:**
  - Total Return (%)
  - Final Balance ($)
  - Sharpe Ratio
  - Max Drawdown
  - Volatility
  - Total Trades and Win Rate
  - Actions distribution and total steps

---

## 4. Quality Gates

**Daily models:**
- Sharpe ≥ 0.8 is solid; ≥ 1.0 is strong
- Drawdown should be plausible for each ticker (TSLA tends to be higher)
- Trades: low/moderate count is fine if returns are strong

**Intraday models:**
- More trades, lower Sharpe expected; ≥ 0.6 is decent, ≥ 0.9 is strong
- If you haven’t modeled costs, be cautious with very high returns

---

## 5. Troubleshooting

- **Scaler NotFittedError:**
  - Best: implement scaler save/load with the model
  - Quick fix: call `normalize_data(..., fit_scaler=True)` in validation
- **Frequent “Invalid price” warnings:**
  - Check CSV: ensure Close/Price columns are floats, >0, and aligned with Date
  - Consistently use the same field (Price vs Close) in your environment
- **Model not loading:**
  - Ensure `models/{SYMBOL}_best_episode_*.pth` exists
  - State dimension at validation must match training features (e.g., 59 for daily)

---

## 6. Interpreting Results

- **High return, 0% win rate:**
  - Can occur if positions aren’t closed (open PnL)
  - Consider reporting both realized and unrealized PnL
  - Optionally enforce periodic flattening during validation
- **TSLA: high drawdown + high Sharpe:**
  - Consider volatility-aware position sizing
- **Intraday: unrealistically high returns:**
  - Add conservative cost assumptions (e.g., 1–3bps per trade + slippage)

---

## 7. Optional Enhancements

- Save a per-run summary CSV (aggregate JSONs for quick comparison)
- Plot equity curves for each run and save as PNG in `validation_logs/`
- Add realized vs unrealized PnL split to validation summary

---

## 8. Support

If you share the three validation JSONs for AMZN/MSFT/TSLA (daily and/or intraday), a compact comparison table and key takeaways can be produced.

---

**For more details, see the main project [README.md](../README.md) and [PROJECT_DOCUMENTATION.md](../PROJECT_DOCUMENTATION.md).**

