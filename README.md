# Trade-Agent

Trade-Agent is a project that applies Deep Q-Learning (DQN), a reinforcement learning technique, to predict and analyze stock prices using various sources of stock market data.

## Features

- **Deep Q-Learning Model:** Utilizes DQN for stock price prediction and trading strategy optimization.
- **Multiple Data Sources:** Supports data acquisition from Yahoo Finance, Alpha Vantage, Quandl, and more.
- **Technical Indicators:** Integrates TA-Lib, pandas-ta, and other libraries for feature engineering.
- **Visualization:** Provides data visualization using Matplotlib, Seaborn, and Plotly.
- **Modular Design:** Easily extendable for new data sources and RL algorithms.

### Model Architure
```
==========================================================================================
Layer (type:depth-idx)                   Output Shape              Param #
==========================================================================================
Sequential                               [1, 3]                    --
├─Linear: 1-1                            [1, 1024]                 61,440
├─LayerNorm: 1-2                         [1, 1024]                 2,048
├─ReLU: 1-3                              [1, 1024]                 --
├─Dropout: 1-4                           [1, 1024]                 --
├─Linear: 1-5                            [1, 512]                  524,800
├─LayerNorm: 1-6                         [1, 512]                  1,024
├─ReLU: 1-7                              [1, 512]                  --
├─Dropout: 1-8                           [1, 512]                  --
├─Linear: 1-9                            [1, 256]                  131,328
├─LayerNorm: 1-10                        [1, 256]                  512
├─ReLU: 1-11                             [1, 256]                  --
├─Dropout: 1-12                          [1, 256]                  --
├─Linear: 1-13                           [1, 128]                  32,896
├─LayerNorm: 1-14                        [1, 128]                  256
├─ReLU: 1-15                             [1, 128]                  --
├─Dropout: 1-16                          [1, 128]                  --
├─Linear: 1-17                           [1, 64]                   8,256
├─ReLU: 1-18                             [1, 64]                   --
├─Linear: 1-19                           [1, 3]                    195
==========================================================================================
Total params: 762,755
Trainable params: 762,755
Non-trainable params: 0
Total mult-adds (M): 0.76
==========================================================================================
Input size (MB): 0.00
Forward/backward pass size (MB): 0.03
Params size (MB): 3.05
Estimated Total Size (MB): 3.08
==========================================================================================
```
## Installation

1. Clone the repository:
    ```sh
    git clone https://github.com/OpenVanguard/Trade-Agent.git
    cd Trade-Agent
    ```
2. Set up the Python environment (recommended: use `stock_rl_env`):
    ```sh
    python -m venv stock_rl_env
    source stock_rl_env/Scripts/activate  # On Windows
    # Or
    source stock_rl_env/bin/activate      # On Linux/Mac
    ```
3. Install dependencies:
    ```sh
    pip install -r requirements.txt
    ```

## Usage
- Refer [src/README.md](src/README.md).
OR
- Run the main script:
    ```sh
    python src/main.py
    ```
- Explore data analysis and experiments in [notebooks/dataAnalysis.ipynb](notebooks/dataAnalysis.ipynb).

## Project Structure

- `src/`: Main source code for RL environment and training.
- `data/`: Storage for raw and processed stock data.
- `notebooks/`: Jupyter notebooks for exploratory analysis.
- `stock_rl_env/`: Python virtual environment.

## Requirements

- Python 3.10+
- PyTorch
- Stable Baselines3
- Gym
- yfinance, pandas_datareader, alpha_vantage, quandl, investpy
- TA-Lib, pandas-ta, scikit-learn, matplotlib, seaborn, plotly

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Author
Virat Srivastava