"""Flower Server."""


from collections import OrderedDict
from typing import Callable, Dict, Optional, Tuple

import torch
from flwr.common.typing import NDArrays, Scalar
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from fedprox.models import test

import hydra
from fedprox.dataset import load_datasets
import flwr as fl
import wandb
from wandb import AlertLevel

torch.set_num_threads(8)

alerted = False


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    model: DictConfig,
) -> Callable[
    [int, NDArrays, Dict[str, Scalar]], Optional[Tuple[float, Dict[str, Scalar]]]
]:
    """Generate the function for centralized evaluation.

    Parameters
    ----------
    testloader : DataLoader
        The dataloader to test the model with.
    device : torch.device
        The device to test the model on.

    Returns
    -------
    Callable[ [int, NDArrays, Dict[str, Scalar]],
                Optional[Tuple[float, Dict[str, Scalar]]] ]
        The centralized evaluation function.
    """

    global alerted  # Declare 'alerted' as global

    def evaluate(
        server_round: int, parameters_ndarrays: NDArrays, config: Dict[str, Scalar]
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        # pylint: disable=unused-argument
        global alerted  # Declare 'alerted' as global inside this function as well

        """Use the entire CIFAR-10 test set for evaluation."""
        net = instantiate(model)
        params_dict = zip(net.state_dict().keys(), parameters_ndarrays)
        state_dict = OrderedDict({k: torch.Tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)
        net.to(device)

        # We could compile the model here but we are not going to do it because
        # running test() is so lightweight that the overhead of compiling the model
        # negate any potential speedup. Please note this is specific to the model and
        # dataset used in this baseline. In general, compiling the model is worth it

        loss, accuracy = test(net, testloader, device=device)
        wandb.log({"centralized_loss": loss, "centralized_accuracy": accuracy, "server_round": server_round})
        if accuracy > 0.8 and not alerted:
            wandb.alert(
                title="Centralized Accuracy Alert",
                text=f"Centralized Accuracy is above 0.8 at round {server_round}",
                level=AlertLevel.INFO,
            )
            alerted = True
        if server_round == 500:
            wandb.alert(
                title="Round Alert",
                text=f"Round is at 500",
                level=AlertLevel.INFO,
            )
        # return statistics
        return loss, {"accuracy": accuracy}

    return evaluate


@hydra.main(config_path="conf", config_name="config", version_base=None)
def main(cfg: DictConfig):
    """Run CNN federated learning on MNIST.

    Parameters
    ----------
    cfg : DictConfig
        An omegaconf object that stores the hydra config.
    """
    # print config structured as YAML
    print(OmegaConf.to_yaml(cfg))

    # partition dataset and get dataloaders
    trainloaders, valloaders, testloader = load_datasets(
        config=cfg.dataset_config,
        num_clients=cfg.num_clients,
        batch_size=cfg.batch_size,
    )

    # get function that will executed by the strategy's evaluate() method
    # Set server's device
    device = cfg.server_device
    evaluate_fn = gen_evaluate_fn(testloader, device=device, model=cfg.model)

    # get a function that will be used to construct the config that the client's
    # fit() method will received
    def get_on_fit_config():
        def fit_config_fn(server_round: int):
            # resolve and convert to python dict
            fit_config: FitConfig = OmegaConf.to_container(  # type: ignore
                cfg.fit_config, resolve=True
            )
            fit_config["curr_round"] = server_round  # add round info
            return fit_config

        return fit_config_fn

    # instantiate strategy according to config. Here we pass other arguments
    # that are only defined at run time.
    strategy = instantiate(
        cfg.strategy,
        evaluate_fn=evaluate_fn,
        on_fit_config_fn=get_on_fit_config(),
    )

    wandb.init(
        entity="hoho",
        # set the wandb project where this run will be logged
        project="fedops-baselines-fedavg-mnist",

        # track hyperparameters and run metadata
        config={
            "architecture": "CNN",
            "dataset": "MNIST",

            "server_version": "v1",
            "min_clients": cfg.clients_per_round,
            "rounds": cfg.num_rounds,
            "client_selection": cfg.client_selection,
            "threshold": cfg.threshold,

            "client_version": "v1",
            "epochs": cfg.num_epochs,
            "batch_size": cfg.batch_size,
            "initial_learning_rate": cfg.initial_learning_rate,
            "learning_rate": cfg.learning_rate,
            "mu": 1.0,
            # "test": "True",
        },
    )

    # Start Flower server for four rounds of federated learning
    fl.server.start_server(
        server_address="0.0.0.0:8080",

        config=fl.server.ServerConfig(num_rounds=cfg.num_rounds),
        # config=fl.server.ServerConfig(num_rounds=3),

        strategy=strategy,
    )


if __name__=="__main__":
    main()