from typing import Dict, Optional, Tuple, List
from collections import OrderedDict
import argparse
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Metrics

import torch

import utils
import warnings
import wandb

import time

start_time: float

warnings.filterwarnings("ignore")
torch.set_num_threads(8)

_server_round = 0

# Define metric aggregation function
def weighted_average(metrics: List[Tuple[int, Metrics]]) -> Metrics:
    # Multiply accuracy of each client by number of examples used
    accuracies = [num_examples * m["accuracy"] for num_examples, m in metrics]
    examples = [num_examples for num_examples, _ in metrics]

    # Aggregate and return custom metric (weighted average)
    accuracy = sum(accuracies) / sum(examples)
    wandb.log({"distributed_accuracy": accuracy, "server_round": _server_round})
    # fl.server.History().losses_distributed
    return {"accuracy": accuracy}


def fit_config(server_round: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 64,
        # "local_epochs": 1 if server_round < 2 else 5,
        "local_epochs": 5,
        "server_round": server_round
    }
    return config


def evaluate_config(server_round: int):
    """Return evaluation configuration dict for each round.
    Perform five local evaluation steps on each client (i.e., use five
    batches) during rounds one to three, then increase to ten local
    evaluation steps.
    """
    # val_steps = 5 if server_round < 4 else 10
    val_steps = 5
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
        global _server_round
        _server_round = server_round
        # Update model with the latest parameters
        params_dict = zip(model.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        model.load_state_dict(state_dict, strict=True)

        loss, accuracy = utils.test(model, valLoader)
        global start_time
        end_time = time.time()
        elapsed_time = (end_time - start_time) / 60.0
        elapsed_minutes = round(elapsed_time, 3)
        wandb.log({"centralized_loss": loss,
                   "centralized_accuracy": accuracy, "server_round": server_round, "Time (Minutes)": elapsed_minutes})
        return loss, {"accuracy": accuracy}

    return evaluate


def main():
    """Load model for
    1. server-side parameter initialization
    2. server-side parameter evaluation
    """
    global start_time
    start_time = time.time()

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

    model = utils.Net()

    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]

    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        # fraction_fit=0.2,
        # fraction_evaluate=0.2,

        min_fit_clients=50,
        min_evaluate_clients=50,
        min_available_clients=50,
        # min_fit_clients=3,
        # min_evaluate_clients=3,
        # min_available_clients=3,

        evaluate_fn=get_evaluate_fn(model, args.toy),
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
        evaluate_metrics_aggregation_fn=weighted_average,
    )

    # start a new wandb run to track this script
    wandb.init(
        entity="hoho",
        # set the wandb project where this run will be logged
        project="fedops-server",

        # track hyperparameters and run metadata
        config={
            "architecture": "CNN",
            "dataset": "CIFAR-10",

            "server_version": "v20",
            "min_clients": 50,
            "rounds": 1000,
            "client_selection": "off",
            "threshold": 3,

            "client_version": "v18",
            "epochs": 5,
            "batch_size": 64,
            "learning_rate": 0.001,
            "momentum": 0.9,
            # "test": "True",
        },

        # (str, optional) A longer description of the run
        # notes='''
        # [server]
        # server_version = v8
        # min_clients = 50
        # rounds = 300
        # client_selection = off
        #
        # [client]
        # client_version = v8
        # local_epochs = 5
        # batch_size = 32
        # learning_rate = 0.01
        # momentum = 0.9
        # '''

        # notes='''
        #     def __init__(self):
        #         super(Net, self).__init__()
        #         self.conv1 = nn.Conv2d(3, 32, 3, padding=1)
        #         self.bn1 = nn.BatchNorm2d(32)
        #         self.conv2 = nn.Conv2d(32, 64, 3, padding=1)
        #         self.bn2 = nn.BatchNorm2d(64)
        #         self.conv3 = nn.Conv2d(64, 128, 3, padding=1)
        #         self.bn3 = nn.BatchNorm2d(128)
        #         self.fc1 = nn.Linear(128 * 4 * 4, 512)
        #         self.fc2 = nn.Linear(512, 10)
        #
        #     def forward(self, x):
        #         x = F.relu(self.bn1(self.conv1(x)))
        #         x = F.relu(self.bn2(self.conv2(x)))
        #         x = F.relu(self.bn3(self.conv3(x)))
        #         x = F.max_pool2d(x, 2)
        #         x = x.view(-1, 128 * 4 * 4)
        #         x = F.relu(self.fc1(x))
        #         x = F.dropout(x, training=self.training)
        #         x = self.fc2(x)
        #         return x
        # '''


    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",

        config=fl.server.ServerConfig(num_rounds=1000),
        # config=fl.server.ServerConfig(num_rounds=3),

        strategy=strategy,
    )


if __name__ == "__main__":
    main()