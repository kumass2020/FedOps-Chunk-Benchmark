from typing import List, Tuple

import flwr as fl
from flwr.common import Metrics


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    return {"accuracy": sum(accuracies) / sum(examples)}


client_manager = fl.server.SimpleClientManager(is_random=1)
client_manager2 = fl.server.SimpleClientManager(is_random=1)
# Define strategy
strategy = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
)
server = fl.server.Server(
    client_manager=client_manager,
    strategy=strategy,
)

strategy2 = fl.server.strategy.FedAvg(
    evaluate_metrics_aggregation_fn=weighted_average,
    min_fit_clients=3,
    min_evaluate_clients=3,
    min_available_clients=3,
)


# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy,
    client_manager=client_manager,
    server=server
)

drop_cid_list: [str] = []


def select_client():
    client_list_by_time = server.get_client_list_by_time()
    # for list in client_list_by_time:
    drop_cid_list.append((client_list_by_time[2])[0])
    return drop_cid_list


server2 = fl.server.Server(
    client_manager=client_manager2,
    drop_cid_list=select_client(),
    strategy=strategy2
)

# Start Flower server
fl.server.start_server(
    server_address="0.0.0.0:8080",
    config=fl.server.ServerConfig(num_rounds=1),
    strategy=strategy2,
    client_manager=client_manager2,
    server=server2,
)
