import statistics
import csv

loss_list = [297.225/50.0, 281.201/50.0, 148.197/50.0]
acc_list = [0.8802, 0.821, 0.8066]

loss_mean = statistics.mean(loss_list)
loss_stdev = statistics.stdev(loss_list)

acc_mean = statistics.mean(acc_list)
acc_stdev = statistics.stdev(acc_list)

print(loss_mean, loss_stdev)
print(acc_mean, acc_stdev)

time_list = [115.056, 112.495, 107.392]
time_mean = statistics.mean(time_list)
time_stdev = statistics.stdev(time_list)
print(time_mean, time_stdev)

time_to_acc_list = [72.749, 60.516, 63.315]
time_to_acc_mean = statistics.mean(time_to_acc_list)
time_to_acc_stdev = statistics.stdev(time_to_acc_list)
print(time_to_acc_mean, time_to_acc_stdev)

transmit_bytes = [11044960932 / 1024**3, 9703912295 / 1024**3, 9446531688 / 1024**3]
transmit_bytes_mean = statistics.mean(transmit_bytes)
transmit_bytes_stdev = statistics.stdev(transmit_bytes)
print(transmit_bytes_mean, transmit_bytes_stdev)

receive_bytes = [5679754961 / 1024**3, 4981986075 / 1024**3, 4850162176 / 1024**3]
receive_bytes_mean = statistics.mean(receive_bytes)
receive_bytes_stdev = statistics.stdev(receive_bytes)
print(receive_bytes_mean, receive_bytes_stdev)

comm_cost = [7800984271+4005740763, 6200032785+3187426716, 7312122384+3757454324]
comm_cost = [i / 1024**3 for i in comm_cost]
comm_cost_bytes_mean = statistics.mean(comm_cost)
comm_cost_bytes_stdev = statistics.stdev(comm_cost)
print(comm_cost_bytes_mean, comm_cost_bytes_stdev)

dropped_proportion = [11/50, 11/50, 14/50]
dropped_proportion_mean = statistics.mean(dropped_proportion)
dropped_proportion_stdev = statistics.stdev(dropped_proportion)
print(dropped_proportion_mean, dropped_proportion_stdev)

# Open the CSV file in read mode
with open('wandb_export_2023-03-08T16_46_45.252+09_00.csv', mode='r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)

    # Initialize an empty list to hold the data
    comm_time1 = []
    comm_time2 = []
    comm_time3 = []

    # Loop through each row in the CSV file
    for row in csv_reader:
        # Extract the value of the second column and append it to the list
        comm_time1.append(row[3])
        comm_time2.append(row[2])
        comm_time3.append(row[1])

overhead_time1 = []
overhead_time2 = []
overhead_time3 = []

comm_time1 = [float(i) for i in comm_time1[1:]]
comm_time2 = [float(i) for i in comm_time2[1:]]
comm_time3 = [float(i) for i in comm_time3[1:]]

for i in comm_time1:
    if i > 0.1:
        overhead_time1.append(i)

for i in comm_time2:
    if i > 0.1:
        overhead_time2.append(i)

for i in comm_time3:
    if i > 0.1:
        overhead_time3.append(i)

overhead_time = [statistics.mean(overhead_time1), statistics.mean(overhead_time2), statistics.mean(overhead_time3)]
overhead_time_mean = statistics.mean(overhead_time)
overhead_time_stdev = statistics.stdev(overhead_time)
# print(overhead_time)
print(overhead_time_mean, overhead_time_stdev)

overhead_clients = [len(overhead_time1), len(overhead_time2), len(overhead_time3)]
overhead_clients_mean = statistics.mean(overhead_clients)
overhead_clients_stdev = statistics.stdev(overhead_clients)
print(overhead_clients_mean, overhead_clients_stdev)