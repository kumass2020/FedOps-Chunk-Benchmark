import csv
import matplotlib.pyplot as plt

# Open the CSV file in read mode
with open('wandb_export_2023-03-23T20_48_33.549+09_00.csv', mode='r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)

    # Initialize an empty list to hold the data
    t0 = []
    t3 = []
    t5 = []
    t7 = []

    # Loop through each row in the CSV file
    for row in csv_reader:
        # Extract the value of the second column and append it to the list
        t0.append(row[2])
        t3.append(row[3])
        t5.append(row[4])
        t7.append(row[1])


print(t0)
print(t3)
print(t5)
print(t7)

t0 = t0[1:]
t3 = t3[1:]
t5 = t5[1:]
t7 = t7[1:]

# Put the string lists in a list
str_lists = [t0, t3, t5, t7]

# Convert the string lists to float lists
float_lists = [[float(x) for x in str_list] for str_list in str_lists]

# Unpack the float lists
t0, t3, t5, t7 = float_lists

# Generate the corresponding x-values
x = list(range(len(t0)))

# Create the line plot
plt.plot(x, t0, label='Baseline FedAvg')
plt.plot(x, t3, label='FedAvg + CAT(k=3)')
plt.plot(x, t5, label='FedAvg + CAT(k=5)')
plt.plot(x, t7, label='FedAvg + CAT(k=7)')

plt.ylim(0.5, 1.0)

# Add labels, title, and legend
plt.xlabel('Communications Rounds', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
# plt.title('Line Plot of Four 1D Lists')
plt.legend(loc='lower right', fontsize=11)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

plt.savefig('moon6.png', dpi=600)

# Show the plot
plt.show()