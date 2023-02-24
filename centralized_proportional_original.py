import torch
import torchvision
import torchvision.transforms as transforms
import time
import torch.nn as nn
import torch.nn.functional as F
import torchsummary


# Define the neural network architecture
class Net(torch.nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

        # super(Net, self).__init__()
        # self.conv1 = nn.Conv2d(3, 3, 5)
        # self.pool = nn.MaxPool2d(2, 2)
        # self.conv2 = nn.Conv2d(3, 8, 5)
        # self.fc1 = nn.Linear(8 * 5 * 5, 60)
        # self.fc2 = nn.Linear(60, 42)
        # self.fc3 = nn.Linear(42, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

        # x = self.pool(F.relu(self.conv1(x)))
        # x = self.pool(F.relu(self.conv2(x)))
        # x = x.view(-1, 8 * 5 * 5)
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # return self.fc3(x)


if __name__ == "__main__":
    # Define the data transformations
    transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(32, padding=4),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # Load the CIFAR-10 dataset
    trainset = torchvision.datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=128, shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root='./data', train=False, download=True, transform=transform)
    testloader = torch.utils.data.DataLoader(testset, batch_size=128, shuffle=False, num_workers=2)

    # Define the loss function and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    net = Net()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    torchsummary.summary(net, input_size=(3, 32, 32))

    # Train the model with benchmarking
    num_epochs = 1
    for epoch in range(num_epochs):
        start_time = time.time()
        running_loss = 0.0
        for i, data in enumerate(trainloader, 0):
            # start_time_4 = time.time()
            inputs, labels = data
            optimizer.zero_grad()
            # print('Gradients Time: %.2f seconds' % (time.time() - start_time_4))
            # start_time_2 = time.time()
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            # print('back-propagation Time: %.2f seconds' % (time.time() - start_time_2))
            # start_time_3 = time.time()
            optimizer.step()
            # print('parameter update Time: %.2f seconds' % (time.time() - start_time_3))

            running_loss += loss.item()
            if i % 100 == 99:    # Print every 100 mini-batches
                print('[Epoch %d, Batch %5d] loss: %.3f' %
                    (epoch + 1, i + 1, running_loss / 100))
                running_loss = 0.0
        
        # Evaluate the model on the test set
        correct = 0
        total = 0
        with torch.no_grad():
            for data in testloader:
                images, labels = data
                outputs = net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

        end_time = time.time()
        epoch_time = end_time - start_time
        print('Epoch %d, Test accuracy: %.2f%%, Time taken: %.2f seconds' % (epoch + 1, 100 * correct / total, epoch_time))

