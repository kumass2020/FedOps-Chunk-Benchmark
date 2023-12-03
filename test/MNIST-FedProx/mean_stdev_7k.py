import statistics
import csv

loss_list = [178.884, 169.164, 176.312]
acc_list = [0.8394, 0.8416, 0.844]
time_list = [121.196, 166.757, 135.883]
time_to_acc_list = [93.445, 130.195, 104.9]
transmit_bytes = [11763256367, 12051826762, 12010097594]
receive_bytes = [6010336990, 6164272894, 6139505339]
comm_cost = [8824681798+4509725689, 9267641349+4740587119, 9024970927+4613774641]
dropped_proportion = [3/50, 2/50, 2/50]

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