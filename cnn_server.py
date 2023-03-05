from typing import Any, Callable, Dict, List, Optional, Tuple

import flwr as fl
from flwr.common import Metrics

import utils
import argparse
from collections import OrderedDict

import wandb

import torch
from torch.utils.data import DataLoader, SubsetRandomSampler

from typing import Dict, Optional, Tuple

torch.set_num_threads(4)

def fit_config(server_round: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 16,
        "local_epochs": 1 if server_round < 2 else 2,
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    val_steps = 5 if server_round < 4 else 10
    return {"val_steps": val_steps}


def get_evaluate_fn(model: torch.nn.Module, toy: bool):
    """Return an evaluation function for server-side evaluation."""

    # Load data and model here to avoid the overhead of doing it in `evaluate` itself
    trainset, _, _ = utils.load_data()

    n_train = len(trainset)
    if toy:
        # use only 10 samples as validation set
        valset = torch.utils.data.Subset(trainset, range(n_train - 10, n_train))
    else:
        # Use the last 5k training examples as a validation set
        valset = torch.utils.data.Subset(trainset, range(n_train - 5000, n_train))

    valLoader = DataLoader(valset, batch_size=16)

    # The `evaluate` function will be called after every round
    def evaluate(
        server_round: int,
        parameters: fl.common.NDArrays,
        config: Dict[str, fl.common.Scalar],
    ) -> Optional[Tuple[float, Dict[str, fl.common.Scalar]]]:
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = utils.test(model, valLoader)
        return loss, {"accuracy": accuracy}

    return evaluate


# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    accuracy = sum(accuracies) / sum(examples)
    wandb.log({"accuracy": accuracy})
    return {"accuracy": accuracy}


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """

    # Parse command line argument `partition`
    parser = argparse.ArgumentParser(description="Flower")
    parser.add_argument(
        "--toy",
        type=bool,
        default=False,
        required=False,
        help="Set to true to use only 10 datasamples for validation. \
            Useful for testing purposes. Default: False",
    )

    args = parser.parse_args()

    model = utils.load_efficientnet(classes=10)

    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    client_manager = fl.server.SimpleClientManager(is_random=1)

    # Define strategy
    strategy = fl.server.strategy.FedAvg(
        fraction_fit=0.2,
        fraction_evaluate=0.2,
        evaluate_metrics_aggregation_fn=weighted_average,
        evaluate_fn=get_evaluate_fn(model, args.toy),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        min_fit_clients=3,
        min_evaluate_clients=3,
        min_available_clients=3,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
    )

    # server = fl.server.Server(
    #     client_manager=client_manager,
    #     strategy=strategy,
    # )

    # start a new wandb run to track this script
    wandb.init(
        entity="hoho",
        # set the wandb project where this run will be logged
        project="fedops-server",

        # track hyperparameters and run metadata
        config={
            "learning_rate": 0.001,
            "architecture": "CNN",
            "dataset": "CIFAR-10",
            "epochs": 5,
        }
    )

    # Start Flower server
    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=4),
        strategy=strategy,
        client_manager=client_manager,
        # server=server,
    )

if __name__ == "__main__":
    main()