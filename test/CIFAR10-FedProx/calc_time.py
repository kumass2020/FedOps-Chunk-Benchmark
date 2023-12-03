import statistics

time_k0 = [306, 305.4]
time_k3 = [145.07, 164.414]
time_k5 = [184.974, 196.29, 162.147]
time_k7 = [196.141, 187.117, 192.107]

print(statistics.mean(time_k0), statistics.stdev(time_k0))
print(statistics.mean(time_k3), statistics.stdev(time_k3))
print(statistics.mean(time_k5), statistics.stdev(time_k5))
print(statistics.mean(time_k7), statistics.stdev(time_k7))
print("-------------------------------------------------")

acc_k0 = [0.796, 0.783]
acc_k3 = [0.735, 0.737]
acc_k5 = [0.795, 0.801, 0.734]
acc_k7 = [0.752, 0.797, 0.797]

time_to_acc_k0 = [i / j for i, j in zip(time_k0, acc_k0)]
time_to_acc_k3 = [i / j for i, j in zip(time_k3, acc_k3)]
time_to_acc_k5 = [i / j for i, j in zip(time_k5, acc_k5)]
time_to_acc_k7 = [i / j for i, j in zip(time_k7, acc_k7)]

print(statistics.mean(time_to_acc_k0), statistics.stdev(time_to_acc_k0))
print(statistics.mean(time_to_acc_k3), statistics.stdev(time_to_acc_k3))
print(statistics.mean(time_to_acc_k5), statistics.stdev(time_to_acc_k5))
print(statistics.mean(time_to_acc_k7), statistics.stdev(time_to_acc_k7))
print("-------------------------------------------------")

mean_of_time_to_acc_list = [statistics.mean(time_to_acc_k0), statistics.mean(time_to_acc_k3), statistics.mean(time_to_acc_k5), statistics.mean(time_to_acc_k7)]
time_to_acc_scaler = [i / min(mean_of_time_to_acc_list) for i in mean_of_time_to_acc_list]
print(time_to_acc_scaler)

# mean_of_time_to_acc_list = [131, 77.8, 83.4, 109]
# time_to_acc_scaler = [i / min(mean_of_time_to_acc_list) for i in mean_of_time_to_acc_list]
# print(time_to_acc_scaler)