import os
import pandas as pd
import numpy as np

# Constants
RISK_FREE_RATE = 0.02  # Assume a constant risk-free rate (e.g., 2% annual)

# Function to calculate Sharpe Ratio
def calculate_sharpe_ratio(returns, risk_free_rate=RISK_FREE_RATE):
    excess_returns = returns - risk_free_rate
    mean_excess_return = np.mean(excess_returns)
    std_dev = np.std(returns)
    if std_dev == 0:  # Handle division by zero
        return np.nan
    return mean_excess_return / std_dev

# Load data
def load_data(stock_file, agent_dir):
    # Load stock returns
    stock_data = pd.read_csv(stock_file)
    stock_returns = stock_data['returns']

    # Load agent-predicted returns
    agents = {}
    for file in os.listdir(agent_dir):
        if file.endswith(".csv"):
            agent_name = os.path.splitext(file)[0]
            agent_data = pd.read_csv(os.path.join(agent_dir, file))
            agents[agent_name] = agent_data['predicted_returns']
    
    return stock_returns, agents

# Main function
def main():
    stock_file = "stock_returns.csv"  # CSV file with stock returns (e.g., daily returns)
    agent_dir = "agents_data"        # Directory with agent-predicted returns CSVs

    # Load stock and agent data
    stock_returns, agents = load_data(stock_file, agent_dir)
    
    # Calculate Sharpe Ratio for the stock
    stock_sharpe_ratio = calculate_sharpe_ratio(stock_returns)
    print(f"Stock Sharpe Ratio: {stock_sharpe_ratio:.2f}")

    # Calculate Sharpe Ratios for all agents
    agent_sharpe_ratios = {}
    for agent_name, predicted_returns in agents.items():
        agent_sharpe_ratios[agent_name] = calculate_sharpe_ratio(predicted_returns)

    # Log results
    for agent, sharpe_ratio in agent_sharpe_ratios.items():
        print(f"Agent: {agent}, Sharpe Ratio: {sharpe_ratio:.2f}")

    # Optionally save to a CSV
    results = pd.DataFrame({
        'Agent': agent_sharpe_ratios.keys(),
        'Sharpe Ratio': agent_sharpe_ratios.values()
    })
    results.to_csv("agent_sharpe_ratios.csv", index=False)
    print("Results saved to agent_sharpe_ratios.csv")

# Run the script
if __name__ == "__main__":
    main()
