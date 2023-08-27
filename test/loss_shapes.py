import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

def plot_with_custom_inset(filepath, xlim_inset, ylim_inset, inset_position, axins_markers):
    # Load data from CSV
    with open(filepath, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        t0, t3, t5, t7 = [], [], [], []
        for row in csv_reader:
            t0.append(row[2])
            t3.append(row[3])
            t5.append(row[4])
            t7.append(row[1])
        t0, t3, t5, t7 = [list(map(float, lst)) for lst in [t0[1:], t3[1:], t5[1:], t7[1:]]]

    # Divide each list by 50
    t0 = [val/50 for val in t0]
    t3 = [val/50 for val in t3]
    t5 = [val/50 for val in t5]
    t7 = [val/50 for val in t7]

    x = list(range(len(t0)))
    fig, ax = plt.subplots()

    marker_styles = ['o', 's', '^', 'D']
    marker_size = 8

    # Main plot with markers at the end
    ax.plot(x, t0, label='Baseline FedAvg', marker=marker_styles[0], markevery=[-1], markersize=marker_size)
    ax.plot(x, t3, label='FedAvg + CAT(k=3)', marker=marker_styles[1], markevery=[-1], markersize=marker_size)
    ax.plot(x, t5, label='FedAvg + CAT(k=5)', marker=marker_styles[2], markevery=[-1], markersize=marker_size)
    ax.plot(x, t7, label='FedAvg + CAT(k=7)', marker=marker_styles[3], markevery=[-1], markersize=marker_size)

    # Adjust y-axis for visibility of markers with a buffer
    buffer_y = 0.5
    ax.set_ylim(0, 12 + buffer_y)

    # Create inset with the given position
    axins = zoomed_inset_axes(ax, zoom=1)
    ip = InsetPosition(ax, inset_position)
    axins.set_axes_locator(ip)
    axins.plot(x, t0, marker=marker_styles[0], markevery=[-1])
    axins.plot(x, t3, marker=marker_styles[1], markevery=[-1])
    axins.plot(x, t5, marker=marker_styles[2], markevery=[-1])
    axins.plot(x, t7, marker=marker_styles[3], markevery=[-1])

    # Set the desired x and y limits for the inset with a buffer
    x1, x2 = xlim_inset
    y1, y2 = ylim_inset
    buffer_x = 5
    buffer_y_inset = 0.03
    axins.set_xlim(x1 - buffer_x, x2 + buffer_x)
    axins.set_ylim(y1 - buffer_y_inset, y2 + buffer_y_inset)

    marker_size_in = 6

    # Mark specified rounds with markers for inset
    for x_val in axins_markers:
        axins.scatter(x_val, t0[x_val], marker=marker_styles[0], color=axins.lines[0].get_color(), s=marker_size_in ** 2)
        axins.scatter(x_val, t3[x_val], marker=marker_styles[1], color=axins.lines[1].get_color(), s=marker_size_in ** 2)
        axins.scatter(x_val, t5[x_val], marker=marker_styles[2], color=axins.lines[2].get_color(), s=marker_size_in ** 2)
        axins.scatter(x_val, t7[x_val], marker=marker_styles[3], color=axins.lines[3].get_color(), s=marker_size_in ** 2)

    # Add rectangle in the main plot to indicate the zoomed area
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax.set_xlabel('Communication Rounds', fontsize=16)
    ax.set_ylabel('Loss', fontsize=16)
    ax.legend(loc='lower left', fontsize=11)
    ax.tick_params(labelsize=12)

    plt.savefig('loss.png', dpi=600)

    plt.show()

# Example usage
filepath = 'wandb_export_2023-03-23T20_55_44.563+09_00.csv'
xlim_inset = [900, 950]
ylim_inset = [3.0, 3.13]
inset_position = [0.25, 0.65, 0.4, 0.3]
axins_markers = [912, 927, 941, 954]  # List of rounds you want to mark on axins
plot_with_custom_inset(filepath, xlim_inset, ylim_inset, inset_position, axins_markers)
