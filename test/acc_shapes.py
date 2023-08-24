import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset

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

# # Adjust the figure size
# plt.figure(figsize=(10, 6))

# Plot with markers, reduced marker size, and transparency
marker_size = 7
opacity = 0.7
marker_spacing = 50

plt.plot(x, t0, label='Baseline FedAvg', marker='o', markevery=marker_spacing, markersize=marker_size, alpha=opacity)
plt.plot(x, t3, label='FedAvg + CAT(k=3)', marker='s', markevery=marker_spacing, markersize=marker_size, alpha=opacity)
plt.plot(x, t5, label='FedAvg + CAT(k=5)', marker='^', markevery=marker_spacing, markersize=marker_size, alpha=opacity)
plt.plot(x, t7, label='FedAvg + CAT(k=7)', marker='D', markevery=marker_spacing, markersize=marker_size, alpha=opacity)

# plt.plot(x, t0, label='Baseline FedAvg', linestyle='-')   # solid line
# plt.plot(x, t3, label='FedAvg + CAT(k=3)', linestyle='--') # dashed line
# plt.plot(x, t5, label='FedAvg + CAT(k=5)', linestyle='-.') # dash-dot line
# plt.plot(x, t7, label='FedAvg + CAT(k=7)', linestyle=':')  # dotted line


fig, ax = plt.subplots()

# Main plot
ax.plot(x, t0, label='Baseline FedAvg')
ax.plot(x, t3, label='FedAvg + CAT(k=3)')
ax.plot(x, t5, label='FedAvg + CAT(k=5)')
ax.plot(x, t7, label='FedAvg + CAT(k=7)')

# Define the inset plot (location and zoom level)
axins = zoomed_inset_axes(ax, zoom=2, loc='upper right')

# Plot the same data on both the main and inset axes
axins.plot(x, t0)
axins.plot(x, t3)
axins.plot(x, t5)
axins.plot(x, t7)

# Define the limits of the inset plot
x1, x2, y1, y2 = 20, 40, 0.6, 0.9 # specify these based on your data and desired zoom
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# Add rectangle in the main plot to indicate where we're zooming from
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")


plt.ylim(0.6, 1.0)

# Add labels, title, and legend
plt.xlabel('Communications Rounds', fontsize=16)
plt.ylabel('Accuracy', fontsize=16)
plt.legend(loc='lower right', fontsize=11)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)

# plt.savefig('moon6.png', dpi=600)

# Show the plot
plt.show()
