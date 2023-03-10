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

a = [i for i in range(len(data))]
plt.scatter(a, data, s=12)
# plt.title('Score Distribution of ML Benchmark Test')
plt.xlabel('Device Number', fontsize=16)
plt.ylabel('CPU Score', fontsize=16)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.ylim(0, max(data)+100)
# plt.savefig('moon3.png', dpi=600)
plt.show()

[1580, 548, 892, 2230, 586, 854, 1504, 471, 1542, 739, 892, 739, 2727, 586, 1542, 816, 968, 3530, 816, 1618, 1695, 1427, 1580, 1542, 816, 1121, 892, 816, 624, 3530, 1580, 586, 1504, 2115, 930, 1313, 2765, 892, 2765, 930, 1427, 1274, 1504, 1313, 892, 1542, 777, 663, 586, 1542]
