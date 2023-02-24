import torch
import time
import argparse

# Define the command line arguments
parser = argparse.ArgumentParser(description='PyTorch Benchmark')
parser.add_argument('--batch-size', type=int, default=32, metavar='N', help='input batch size for training (default: 32)')
parser.add_argument('--num-workers', type=int, default=0, metavar='N', help='number of workers to use for data loading (default: 0)')
args = parser.parse_args()

# Set the device to GPU if available
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Define the model
model = torch.nn.Sequential(
    torch.nn.Linear(784, 128),
    torch.nn.ReLU(),
    torch.nn.Linear(128, 10)
).to(device)

# Define the loss function and optimizer
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

# Generate random data for benchmarking
data = torch.randn(args.batch_size, 784).to(device)
labels = torch.randint(10, size=(args.batch_size,)).to(device)

# Measure the time for a single forward and backward pass
model.train()
start_time = time.time()
output = model(data)
loss = criterion(output, labels)
loss.backward()
optimizer.step()
print(f"Elapsed time for a single forward and backward pass: {time.time() - start_time:.6f} seconds")

# Define the data loader
trainset = torch.utils.data.TensorDataset(torch.randn(10000, 784), torch.randint(10, size=(10000,)))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

# Measure the time for a full epoch of training
model.train()
start_time = time.time()
for i, (data, labels) in enumerate(trainloader, 0):
    data, labels = data.to(device), labels.to(device)
    optimizer.zero_grad()
    output = model(data)
    loss = criterion(output, labels)
    loss.backward()
    optimizer.step()
print(f"Elapsed time for a full epoch of training: {time.time() - start_time:.6f} seconds")
