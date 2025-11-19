import json
import os
import random

# Parameters for the configuration generator
STOCK_NAME = "AAPL"  # Example stock
NUM_AGENTS = 10      # Number of agents to generate
CONFIG_DIR = "agent_configs"  # Directory to store the configurations

# Hyperparameter ranges
HYPERPARAMETER_RANGES = {
    "learning_rate": (0.001, 0.01),
    "batch_size": [16, 32, 64],
    "num_layers": [2, 3, 4],
    "units_per_layer": [32, 64, 128],
    "dropout_rate": (0.1, 0.5),
    "lookback_window": [30, 60, 90],  # Days of historical data to use
}

# Function to generate a random hyperparameter configuration
def generate_random_config(stock_name):
    config = {
        "stock_name": stock_name,
        "learning_rate": round(random.uniform(*HYPERPARAMETER_RANGES["learning_rate"]), 6),
        "batch_size": random.choice(HYPERPARAMETER_RANGES["batch_size"]),
        "num_layers": random.choice(HYPERPARAMETER_RANGES["num_layers"]),
        "units_per_layer": random.choice(HYPERPARAMETER_RANGES["units_per_layer"]),
        "dropout_rate": round(random.uniform(*HYPERPARAMETER_RANGES["dropout_rate"]), 2),
        "lookback_window": random.choice(HYPERPARAMETER_RANGES["lookback_window"]),
    }
    return config

# Main function to generate configurations
def main():
    os.makedirs(CONFIG_DIR, exist_ok=True)
    
    for i in range(1, NUM_AGENTS + 1):
        config = generate_random_config(STOCK_NAME)
        config_path = os.path.join(CONFIG_DIR, f"agent_{i}.json")
        
        with open(config_path, 'w') as config_file:
            json.dump(config, config_file, indent=4)
        
        print(f"Generated configuration for Agent {i}: {config_path}")

if __name__ == "__main__":
    main()
