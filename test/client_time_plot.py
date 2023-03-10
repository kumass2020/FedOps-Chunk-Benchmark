import csv
import matplotlib.pyplot as plt

# Open the CSV file in read mode
with open('wandb_export_2023-03-07T16_32_50.767+09_00.csv', mode='r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)

    # Initialize an empty list to hold the data
    submit_list = []
    receive_list = []
    train_list = []
    total_list = []

    # Loop through each row in the CSV file
    for row in csv_reader:
        # Extract the value of the second column and append it to the list
        submit_list.append(row[2])
        receive_list.append(row[3])
        train_list.append(row[4])
        total_list.append(row[5])


print(submit_list[1:])
print(receive_list[1:])
print(train_list[1:])
print(total_list[1:])

submit_list = [float(i) for i in submit_list[1:]]
receive_list = [float(i) for i in receive_list[1:]]
train_list = [float(i) for i in train_list[1:]]
total_list = [float(i) for i in total_list[1:]]

client_number_list = [i for i in range(1, 51)]

# Create a figure with 2 rows and 2 columns
fig, axs = plt.subplots(2, 2, figsize=(16, 8))

# Plot the first scatter plot in the top left corner
axs[0, 0].bar(client_number_list, submit_list, color='r')
axs[0, 0].set_title('Communication Time (Before Training)', fontsize=14)
axs[0, 0].set_ylim(0, max(submit_list)+0.5)

# Plot the second scatter plot in the top right corner
axs[1, 0].bar(client_number_list, train_list, color='g')
axs[1, 0].set_title('Training Time', fontsize=14)
axs[1, 0].set_ylim(0, max(train_list)+2.5)

# Plot the third scatter plot in the bottom left corner
axs[0, 1].bar(client_number_list, receive_list, color='b')
axs[0, 1].set_title('Communication Time (After Training)', fontsize=14)
axs[0, 1].set_ylim(0, max(receive_list)+0.01)

# Plot the fourth scatter plot in the bottom right corner
axs[1, 1].bar(client_number_list, total_list, color='m')
axs[1, 1].set_title('Total Elapsed Time', fontsize=14)
axs[1, 1].set_ylim(0, max(total_list)+2.5)

# Add a common x-axis label and a common y-axis label to the figure
fig.text(0.5, 0.04, 'Client Number', ha='center', fontsize=12)
fig.text(0.07, 0.5, 'Time (seconds)', va='center', rotation='vertical', fontsize=12)

plt.savefig('moon4.png', dpi=600)

# Show the plot
plt.show()