import csv
import numpy as np
def get_cpu_distribution():
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
    # print(data[1:], "\n", len(data[1:]))

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

    # pod_cpu_limits = random.sample(scaled_numbers, 50)
    pod_cpu_limits = [int(x) for x in X_sampled]
    print(pod_cpu_limits)

    X_sampled = np.random.choice(bins[:-1], size=20, replace=True, p=prob)

    pod_cpu_limits = [int(x) for x in X_sampled]
    print(pod_cpu_limits)

    core_limited_list = [x if x <= 1000 else 1000 for x in pod_cpu_limits]
    print(core_limited_list)
    print(sum(core_limited_list))

    return pod_cpu_limits

if __name__=="__main__":
    dist = get_cpu_distribution()