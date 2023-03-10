import csv
import matplotlib.pyplot as plt

# Open the CSV file in read mode
with open('ML_ALL_benchmarks.csv', mode='r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)

    # Initialize an empty list to hold the data
    data = []

    # Loop through each row in the CSV file
    for row in csv_reader:
        # Extract the value of the second column and append it to the list
        data.append(row[5])

# Print the list of data
print(data[1:], "\n", len(data[1:]))

# Convert the list of strings to a list of integers
data = [int(x) for x in data[1:]]

# data = data[1:]
mean = sum(data) / len(data)

new_mean = 1400

# Scale each value in the list to have a mean of 1400
scaled_numbers = [(value - mean) * (new_mean / mean) + new_mean for value in data]

# Print the scaled list of numbers
print(scaled_numbers)
print(sum(scaled_numbers) / len(scaled_numbers))

import numpy as np
from sklearn.decomposition import PCA
from sklearn.random_projection import GaussianRandomProjection

# Generate a sample list of 188 elements
X = scaled_numbers
# X = [np.random.normal(loc=0, scale=1) for i in range(188)]

# Convert the list to an ndarray
X = np.array(X)

# Calculate the histogram of the original data
hist, bins = np.histogram(X, bins=100)

# Compute the probability of each bin being chosen
prob = hist / len(X)

# Sample 50 elements with replacement based on the probability of each bin
X_sampled = np.random.choice(bins[:-1], size=50, replace=True, p=prob)

# Check the length, mean and variance of the original and sampled data
print(f"Original data length: {len(X)}")
print(f"Sampled data length: {len(X_sampled)}")

print(f"Original data mean: {round(np.mean(X), 6)}")
print(f"Sampled data mean: {round(np.mean(X_sampled), 6)}")

print(f"Original data standard deviation: {round(np.std(X), 6)}")
print(f"Sampled data standard deviation: {round(np.std(X_sampled), 6)}")