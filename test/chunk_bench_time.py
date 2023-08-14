import csv
import matplotlib.pyplot as plt

# Open the CSV file in read mode
with open('wandb_export_2023-03-16T16_12_48.841+09_00.csv', mode='r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)

    # Initialize an empty list to hold the data
    normal_time = []
    chunk_time = []

    # Loop through each row in the CSV file
    for row in csv_reader:
        # Extract the value of the second column and append it to the list
        normal_time.append(row[1])
        chunk_time.append(row[2])

normal_time = [float(i) for i in normal_time[1:]]
chunk_time = [float(i) for i in chunk_time[1:]]

import statistics
normal_mean = statistics.mean(normal_time)
normal_std = statistics.stdev(normal_time)

chunk_mean = statistics.mean(chunk_time)
chunk_std = statistics.stdev(chunk_time)

print(normal_mean, normal_std)
print(chunk_mean, chunk_std)
print(normal_mean / chunk_mean, normal_std / chunk_std)