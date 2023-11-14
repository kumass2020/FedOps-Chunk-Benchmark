from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics
import wandb


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


# Define strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=50,
    min_evaluate_clients=50,
    min_available_clients=50,
)

# start a new wandb run to track this script
wandb.init(
    entity="hoho",
    # set the wandb project where this run will be logged
    project="fedops-baselines-fedavg",
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=20),
    strategy=strategy,
)

wandb.finish()
