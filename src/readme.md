# Usage Instructions

1. Generate Configurations
Run the configuration generator script to create JSON files:

bash
Copy code
python generate_configs.py
This will create a directory agent_configs/ containing files like agent_1.json, agent_2.json, etc., each with randomized hyperparameters for a specific stock.

2. Run Agents
Run the agent script to process all generated configurations:

```bash
python run_agent.py
```

This will run each agent with its corresponding configuration and save the results in the results/ directory.
Example JSON Configuration (agent_1.json):
JSON Configuration

```json
{
    "stock_name": "AAPL",
    "learning_rate": 0.008214,
    "batch_size": 32,
    "num_layers": 3,
    "units_per_layer": 64,
    "dropout_rate": 0.25,
    "lookback_window": 60
}
```

Output Logs Example:

``` plaintext

Generated configuration for Agent 1: agent_configs/agent_1.json
Generated configuration for Agent 2: agent_configs/agent_2.json
...

Running agent for AAPL with config: {config}
Fetched 60 data points for AAPL
Simulating model training with learning_rate=0.008214, batch_size=32...
Training complete for agent with config: {config}

```
