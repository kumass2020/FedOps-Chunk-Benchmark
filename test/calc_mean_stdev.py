import statistics

original_mean_list = [1400, 1400, 1400, 1400, 1400]
sample_mean_list = [1400.230215, 1431.582437, 1311.526369, 1416.288671, 1447.640892]
original_stdev_list = [821.594907, 821.594907, 821.594907, 821.594907, 821.594907]
sample_stdev_list = [891.8390596963, 964.940028, 711.733848, 846.536236, 770.904456]

# Calculate the mean of the list
sample_mean_mean = statistics.mean(sample_mean_list)
sample_stdev_mean = statistics.mean(sample_stdev_list)

# Calculate the standard deviation of the list
sample_mean_stdev = statistics.stdev(sample_mean_list)
sample_stdev_stdev = statistics.stdev(sample_stdev_list)

print(sample_mean_mean, "+-", sample_mean_stdev)
print(sample_stdev_mean, "+-", sample_stdev_stdev)
