import torch
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm

import warnings

warnings.filterwarnings("ignore")

# DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)

    num_examples = {"trainset": len(trainset), "testset": len(testset)}
    return trainset, testset, num_examples


def load_partition(idx: int):
    """Load 1/10th of the training and test data to simulate a partition."""
    assert idx in range(50)
    trainset, testset, num_examples = load_data()
    n_train = int(num_examples["trainset"] / 50)
    n_test = int(num_examples["testset"] / 50)

    train_partition = torch.utils.data.Subset(
        trainset, range(idx * n_train, (idx + 1) * n_train)
    )
    test_partition = torch.utils.data.Subset(
        testset, range(idx * n_test, (idx + 1) * n_test)
    )
    return (train_partition, test_partition)


# def train(net, trainloader, valloader, epochs, device: str = "cpu"):
#     """Train the network on the training set."""
#     print("Starting training...")
#     net.to(device)  # move model to GPU if available
#     criterion = torch.nn.CrossEntropyLoss().to(device)
#     optimizer = torch.optim.SGD(
#         net.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4
#     )
#     net.train()
#     for _ in range(epochs):
#         for images, labels in trainloader:
#             images, labels = images.to(device), labels.to(device)
#             optimizer.zero_grad()
#             loss = criterion(net(images), labels)
#             loss.backward()
#             optimizer.step()
#
#     net.to("cpu")  # move model back to CPU
#
#     train_loss, train_acc = test(net, trainloader)
#     val_loss, val_acc = test(net, valloader)
#
#     results = {
#         "train_loss": train_loss,
#         "train_accuracy": train_acc,
#         "val_loss": val_loss,
#         "val_accuracy": val_acc,
#     }
#     return results
#
#
# def test(net, testloader, steps: int = None, device: str = "cpu"):
#     """Validate the network on the entire test set."""
#     print("Starting evalutation...")
#     net.to(device)  # move model to GPU if available
#     criterion = torch.nn.CrossEntropyLoss()
#     correct, loss = 0, 0.0
#     net.eval()
#     with torch.no_grad():
#         for batch_idx, (images, labels) in enumerate(testloader):
#             images, labels = images.to(device), labels.to(device)
#             outputs = net(images)
#             loss += criterion(outputs, labels).item()
#             _, predicted = torch.max(outputs.data, 1)
#             correct += (predicted == labels).sum().item()
#             if steps is not None and batch_idx == steps:
#                 break
#     accuracy = correct / len(testloader.dataset)
#     net.to("cpu")  # move model back to CPU
#     return loss, accuracy


def replace_classifying_layer(efficientnet_model, num_classes: int = 10):
    """Replaces the final layer of the classifier."""
    num_features = efficientnet_model.classifier.fc.in_features
    efficientnet_model.classifier.fc = torch.nn.Linear(num_features, num_classes)


# def load_efficientnet(entrypoint: str = "nvidia_efficientnet_b0", classes: int = None):
#     """Loads pretrained efficientnet model from torch hub. Replaces final
#     classifying layer if classes is specified.
#     Args:
#         entrypoint: EfficientNet model to download.
#                     For supported entrypoints, please refer
#                     https://pytorch.org/hub/nvidia_deeplearningexamples_efficientnet/
#         classes: Number of classes in final classifying layer. Leave as None to get the downloaded
#                  model untouched.
#     Returns:
#         EfficientNet Model
#     Note: One alternative implementation can be found at https://github.com/lukemelas/EfficientNet-PyTorch
#     """
#     efficientnet = torch.hub.load(
#         "NVIDIA/DeepLearningExamples:torchhub", entrypoint, pretrained=True
#     )
#
#     if classes is not None:
#         replace_classifying_layer(efficientnet, classes)
#     return efficientnet


def get_model_params(model):
    """Returns a model's parameters."""
    return [val.cpu().numpy() for _, val in model.state_dict().items()]


class Net(nn.Module):
    """Model (simple CNN adapted from 'PyTorch: A 60 Minute Blitz')"""

    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


def train(net, trainloader, valloader, epochs, device: str = "cpu"):
    """Train the model on the training set."""
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(device)), labels.to(device)).backward()
            optimizer.step()

    train_loss, train_acc = test(net, trainloader)
    val_loss, val_acc = test(net, valloader)

    results = {
        "train_loss": train_loss,
        "train_accuracy": train_acc,
        "val_loss": val_loss,
        "val_accuracy": val_acc,
    }
    return results


def test(net, testloader, device: str = "cpu"):
    """Validate the model on the test set."""
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(device))
            labels = labels.to(device)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy


