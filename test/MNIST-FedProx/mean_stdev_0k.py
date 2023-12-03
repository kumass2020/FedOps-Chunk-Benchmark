import statistics
import csv

loss_list = [168.752, 197.833, 172.154]
acc_list = [0.841, 0.8168, 0.839]
time_list = [166.182, 141.209, 140.782]
time_to_acc_list = [128.691, 128.815, 108.396]
transmit_bytes = [12553302335, 12547441776, 12540402314]
receive_bytes = [6417768381, 6419032634, 5403098204]
comm_cost = [9628372585+4922933517, 11339674044+5802067562, 9642080317+4935700228]
dropped_proportion = [11/50, 11/50, 14/50]

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