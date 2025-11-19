import os
import json
import requests
import numpy as np

# Directory containing configurations
CONFIG_DIR = "agent_configs"

# Alpha Vantage API
ALPHA_VANTAGE_API_KEY = "----------------your_alpha_vantage_api_key_here----------------"
BASE_URL = "https://www.alphavantage.co/query"

# Function to fetch stock data from Alpha Vantage
def fetch_stock_data(stock_name, lookback_window):
    params = {
        "function": "TIME_SERIES_DAILY_ADJUSTED",
        "symbol": stock_name,
        "apikey": ALPHA_VANTAGE_API_KEY,
        "outputsize": "compact"
    }
    response = requests.get(BASE_URL, params=params)
    data = response.json()
    
    if "Time Series (Daily)" not in data:
        raise ValueError(f"Error fetching data for {stock_name}: {data}")
    
    time_series = data["Time Series (Daily)"]
    closing_prices = [
        float(value["5. adjusted close"])
        for date, value in time_series.items()
    ][:lookback_window]
    
    return np.array(closing_prices[::-1])  # Reverse to chronological order

# Function to simulate agent behavior
def run_agent(config):
    stock_name = config["stock_name"]
    lookback_window = config["lookback_window"]
    learning_rate = config["learning_rate"]
    batch_size = config["batch_size"]
    
    print(f"Running agent for {stock_name} with config: {config}")
    
    # Fetch stock data
    stock_data = fetch_stock_data(stock_name, lookback_window)
    print(f"Fetched {len(stock_data)} data points for {stock_name}")
    
    # Example: Placeholder for model training (implement your logic)
    print(f"Simulating model training with learning_rate={learning_rate}, batch_size={batch_size}...")
    print(f"Training complete for agent with config: {config}")

# Main function to load configurations and run agents
def main():
    config_files = [f for f in os.listdir(CONFIG_DIR) if f.endswith(".json")]
    
    if not config_files:
        print(f"No configuration files found in {CONFIG_DIR}")
        return
    
    for config_file in config_files:
        config_path = os.path.join(CONFIG_DIR, config_file)
        with open(config_path, 'r') as file:
            config = json.load(file)
        try:
            run_agent(config)
        except Exception as e:
            print(f"Error running agent for {config_file}: {e}")

if __name__ == "__main__":
    main()
