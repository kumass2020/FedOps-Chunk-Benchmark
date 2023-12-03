import statistics

time_k0 = [541.02, 396.36, 742.74]
time_k3 = [219.518, 221.852, 185.713]
time_k5 = [192.215, 183.645, 314.1]
time_k7 = [280.948, 269.425, 338.304]

print(statistics.mean(time_k0), statistics.stdev(time_k0))
print(statistics.mean(time_k3), statistics.stdev(time_k3))
print(statistics.mean(time_k5), statistics.stdev(time_k5))
print(statistics.mean(time_k7), statistics.stdev(time_k7))

acc_k0 = [0.8257, 0.8559, 0.8666]
acc_k3 = [0.8441, 0.8252, 0.839]
acc_k5 = [0.8237, 0.8519, 0.8417]
acc_k7 = [0.8597, 0.8772, 0.8787]

time_to_acc_k0 = [i / j for i, j in zip(time_k0, acc_k0)]
time_to_acc_k3 = [i / j for i, j in zip(time_k3, acc_k3)]
time_to_acc_k5 = [i / j for i, j in zip(time_k5, acc_k5)]
time_to_acc_k7 = [i / j for i, j in zip(time_k7, acc_k7)]

print(statistics.mean(time_to_acc_k0), statistics.stdev(time_to_acc_k0))
print(statistics.mean(time_to_acc_k3), statistics.stdev(time_to_acc_k3))
print(statistics.mean(time_to_acc_k5), statistics.stdev(time_to_acc_k5))
print(statistics.mean(time_to_acc_k7), statistics.stdev(time_to_acc_k7))

mean_of_time_to_acc_list = [statistics.mean(time_to_acc_k0), statistics.mean(time_to_acc_k3), statistics.mean(time_to_acc_k5), statistics.mean(time_to_acc_k7)]
time_to_acc_scaler = [i / min(mean_of_time_to_acc_list) for i in mean_of_time_to_acc_list]
print(time_to_acc_scaler)

mean_of_time_to_acc_list = [131, 77.8, 83.4, 109]
time_to_acc_scaler = [i / min(mean_of_time_to_acc_list) for i in mean_of_time_to_acc_list]
print(time_to_acc_scaler)