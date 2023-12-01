from typing import Dict, Optional, Tuple, List
from collections import OrderedDict
import argparse
from torch.utils.data import DataLoader

import flwr as fl
from flwr.common import Metrics

import torch

import sys
sys.path.append('../../baselines')
sys.path.append('./baselines')
from fedavg_mnist import utils
import warnings
import wandb

import time

from dataset import load_datasets
import model as _model
import torch

start_time: float

warnings.filterwarnings("ignore")
torch.set_num_threads(8)


def fit_config(server_round: int):
    """Return training configuration dict for each round.
    Keep batch size fixed at 32, perform two rounds of training with one
    local epoch, increase to two local epochs afterwards.
    """
    config = {
        "batch_size": 10,
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

    model = _model.Net()

    iid = False
    balance = True
    num_clients = 25
    batch_size = 10

    trainloaders, valloaders, testloader = load_datasets(
        iid=iid, balance=balance, num_clients=num_clients, batch_size=batch_size
    )
    

    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    evaluate_fn = utils.gen_evaluate_fn(testloader, DEVICE)

    model_parameters = [val.cpu().numpy() for _, val in model.state_dict().items()]


    # Create strategy
    strategy = fl.server.strategy.FedAvg(
        # fraction_fit=0.2,
        # fraction_evaluate=0.2,

        min_fit_clients=25,
        min_evaluate_clients=25,
        min_available_clients=25,

        evaluate_fn=evaluate_fn,
        on_fit_config_fn=fit_config,
        on_evaluate_config_fn=evaluate_config,
        # initial_parameters=fl.common.ndarrays_to_parameters(model_parameters),
        evaluate_metrics_aggregation_fn=utils.weighted_average,
    )

    # start a new wandb run to track this script
    wandb.init(
        entity="hoho",
        # set the wandb project where this run will be logged
        project="fedops-baselines-fedavg",

        # track hyperparameters and run metadata
        config={
            "architecture": "CNN",
            "dataset": "MNIST",

            "server_version": "v2",
            "min_clients": 25,
            "rounds": 1000,
            "client_selection": "on",
            "threshold": 3,

            "client_version": "v2",
            "epochs": 5,
            "batch_size": 10,
            "learning_rate": 0.1,
            "momentum": 0.0,
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