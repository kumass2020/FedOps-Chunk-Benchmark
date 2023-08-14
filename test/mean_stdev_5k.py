import statistics
import csv

loss_list = [186.43, 164.182, 182.398]
acc_list = [0.8366, 0.8556, 0.8394]
time_list = [115.487, 104.623, 112.297]
time_to_acc_list = [89.638, 72.354, 88.393]
transmit_bytes = [11299612926, 10767713381, 11748300818]
receive_bytes = [5787266481, 5501398771, 6005143362]
comm_cost = [8505253225+4356590057, 7364005272+3764054608, 9206732563+4706520154]
dropped_proportion = [5/50, 7/50, 3/50]

loss_list = [i/50.0 for i in loss_list]

loss_mean = statistics.mean(loss_list)
loss_stdev = statistics.stdev(loss_list)

acc_mean = statistics.mean(acc_list)
acc_stdev = statistics.stdev(acc_list)

print(loss_mean, loss_stdev)
print(acc_mean, acc_stdev)


time_mean = statistics.mean(time_list)
time_stdev = statistics.stdev(time_list)
print(time_mean, time_stdev)


time_to_acc_mean = statistics.mean(time_to_acc_list)
time_to_acc_stdev = statistics.stdev(time_to_acc_list)
print(time_to_acc_mean, time_to_acc_stdev)

transmit_bytes = [i / 1024**3 for i in transmit_bytes]
transmit_bytes_mean = statistics.mean(transmit_bytes)
transmit_bytes_stdev = statistics.stdev(transmit_bytes)
print(transmit_bytes_mean, transmit_bytes_stdev)

receive_bytes = [i / 1024**3 for i in receive_bytes]
receive_bytes_mean = statistics.mean(receive_bytes)
receive_bytes_stdev = statistics.stdev(receive_bytes)
print(receive_bytes_mean, receive_bytes_stdev)

comm_cost = [i / 1024**3 for i in comm_cost]
comm_cost_bytes_mean = statistics.mean(comm_cost)
comm_cost_bytes_stdev = statistics.stdev(comm_cost)
print(comm_cost_bytes_mean, comm_cost_bytes_stdev)

dropped_proportion_mean = statistics.mean(dropped_proportion)
dropped_proportion_stdev = statistics.stdev(dropped_proportion)
print(dropped_proportion_mean, dropped_proportion_stdev)