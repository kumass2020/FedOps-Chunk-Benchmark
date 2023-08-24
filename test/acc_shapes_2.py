import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

# Open the CSV file in read mode
with open('wandb_export_2023-03-23T20_48_33.549+09_00.csv', mode='r') as csv_file:
    # Create a CSV reader object
    csv_reader = csv.reader(csv_file)

    # Initialize an empty list to hold the data
    t0, t3, t5, t7 = [], [], [], []

    # Loop through each row in the CSV file
    for row in csv_reader:
        t0.append(row[2])
        t3.append(row[3])
        t5.append(row[4])
        t7.append(row[1])

# Remove header
t0, t3, t5, t7 = t0[1:], t3[1:], t5[1:], t7[1:]

# Convert the string lists to float lists
t0, t3, t5, t7 = [list(map(float, lst)) for lst in [t0, t3, t5, t7]]

# Generate the corresponding x-values
x = list(range(len(t0)))

fig, ax = plt.subplots()

# Main plot
ax.plot(x, t0, label='Baseline FedAvg')
ax.plot(x, t3, label='FedAvg + CAT(k=3)')
ax.plot(x, t5, label='FedAvg + CAT(k=5)')
ax.plot(x, t7, label='FedAvg + CAT(k=7)')

# Create inset of width 30% and height 30% of the parent axes' bounding box
# and set its position manually
axins = zoomed_inset_axes(ax, zoom=4)
ip = InsetPosition(ax, [0.1, 0.65, 0.4, 0.4])  # position: x, y, width, height
axins.set_axes_locator(ip)

# # Plot the same data on both the main and inset axes with markers
# # Plot with markers, reduced marker size, and transparency
# marker_size = 5
# opacity = 0.7
# marker_spacing = 10
# axins.plot(x, t0, marker='o', markevery=marker_spacing, markersize=marker_size, alpha=opacity)
# axins.plot(x, t3, marker='s', markevery=marker_spacing, markersize=marker_size, alpha=opacity)
# axins.plot(x, t5, marker='^', markevery=marker_spacing, markersize=marker_size, alpha=opacity)
# axins.plot(x, t7, marker='D', markevery=marker_spacing, markersize=marker_size, alpha=opacity)

# Plot the same data on both the main and inset axes
axins.plot(x, t0)
axins.plot(x, t3)
axins.plot(x, t5)
axins.plot(x, t7)

# Define the limits of the inset plot (Modify these values according to your data's region of interest)
x1, x2, y1, y2 = 850, 950, 0.89, 0.94
axins.set_xlim(x1, x2)
axins.set_ylim(y1, y2)

# Add rectangle in the main plot to indicate where we're zooming from
mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

# Set main plot y-limit and other details
ax.set_ylim(0.6, 1.0)
ax.set_xlabel('Communications Rounds', fontsize=16)
ax.set_ylabel('Accuracy', fontsize=16)
ax.legend(loc='lower right', fontsize=11)
ax.tick_params(labelsize=12)

# Show the plot
plt.show()
