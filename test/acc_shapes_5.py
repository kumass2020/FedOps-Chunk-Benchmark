import csv
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset
from mpl_toolkits.axes_grid1.inset_locator import InsetPosition

def plot_with_custom_inset(filepath, xlim_inset, ylim_inset, inset_position):
    # Load data from CSV
    with open(filepath, mode='r') as csv_file:
        csv_reader = csv.reader(csv_file)
        t0, t3, t5, t7 = [], [], [], []
        for row in csv_reader:
            t0.append(row[2])
            t3.append(row[3])
            t5.append(row[4])
            t7.append(row[1])
        t0, t3, t5, t7 = t0[1:], t3[1:], t5[1:], t7[1:]
        t0, t3, t5, t7 = [list(map(float, lst)) for lst in [t0, t3, t5, t7]]

    x = list(range(len(t0)))
    fig, ax = plt.subplots()

    marker_styles = ['o', 's', '^', 'D']
    marker_size = 10

    # Main plot with zorder=1
    ax.plot(x, t0, label='Baseline FedAvg', zorder=1)
    ax.plot(x, t3, label='FedAvg + CAT(k=3)', zorder=1)
    ax.plot(x, t5, label='FedAvg + CAT(k=5)', zorder=1)
    ax.plot(x, t7, label='FedAvg + CAT(k=7)', zorder=1)

    # Calculate the first visible x-coordinate
    visible_xs = [i for i, y in enumerate(t0) if 0.6 <= y <= 1.0]
    first_visible_x = visible_xs[0] if visible_xs else x[0]

    # Markers with zorder=5 to make them appear above the graph
    ax.scatter(first_visible_x, 0.6, marker=marker_styles[0], color=ax.lines[0].get_color(), s=marker_size ** 2,
               zorder=5)
    ax.scatter(first_visible_x, 0.6, marker=marker_styles[1], color=ax.lines[1].get_color(), s=marker_size ** 2,
               zorder=5)
    ax.scatter(first_visible_x, 0.6, marker=marker_styles[2], color=ax.lines[2].get_color(), s=marker_size ** 2,
               zorder=5)
    ax.scatter(first_visible_x, 0.6, marker=marker_styles[3], color=ax.lines[3].get_color(), s=marker_size ** 2,
               zorder=5)

    ax.scatter(x[-1], t0[-1], marker=marker_styles[0], color=ax.lines[0].get_color(), s=marker_size ** 2, zorder=5)
    ax.scatter(x[-1], t3[-1], marker=marker_styles[1], color=ax.lines[1].get_color(), s=marker_size ** 2, zorder=5)
    ax.scatter(x[-1], t5[-1], marker=marker_styles[2], color=ax.lines[2].get_color(), s=marker_size ** 2, zorder=5)
    ax.scatter(x[-1], t7[-1], marker=marker_styles[3], color=ax.lines[3].get_color(), s=marker_size ** 2, zorder=5)

    # Adjust y-axis for visibility of markers with a buffer
    buffer_y = 0.01
    ax.set_ylim(0.6 - buffer_y, 1.0)

    # Create inset with the given position
    axins = zoomed_inset_axes(ax, zoom=1)
    ip = InsetPosition(ax, inset_position)
    axins.set_axes_locator(ip)
    axins.plot(x, t0)
    axins.plot(x, t3)
    axins.plot(x, t5)
    axins.plot(x, t7)

    # Set the desired x and y limits for the inset with a buffer
    x1, x2 = xlim_inset
    y1, y2 = ylim_inset
    buffer_x = 5
    buffer_y_inset = 0.002
    axins.set_xlim(x1 - buffer_x, x2 + buffer_x)
    axins.set_ylim(y1 - buffer_y_inset, y2 + buffer_y_inset)

    # Mark start and end with markers for inset
    axins.scatter([x1, x2], [t0[x1], t0[x2]], marker=marker_styles[0], color=axins.lines[0].get_color(), s=marker_size ** 2)
    axins.scatter([x1, x2], [t3[x1], t3[x2]], marker=marker_styles[1], color=axins.lines[1].get_color(), s=marker_size ** 2)
    axins.scatter([x1, x2], [t5[x1], t5[x2]], marker=marker_styles[2], color=axins.lines[2].get_color(), s=marker_size ** 2)
    axins.scatter([x1, x2], [t7[x1], t7[x2]], marker=marker_styles[3], color=axins.lines[3].get_color(), s=marker_size ** 2)

    # Add rectangle in the main plot to indicate the zoomed area
    mark_inset(ax, axins, loc1=2, loc2=4, fc="none", ec="0.5")

    ax.set_xlabel('Communications Rounds', fontsize=16)
    ax.set_ylabel('Accuracy', fontsize=16)
    ax.legend(loc='lower right', fontsize=11)
    ax.tick_params(labelsize=12)

    plt.show()

# Example usage
filepath = 'wandb_export_2023-03-23T20_48_33.549+09_00.csv'
xlim_inset = [850, 950]
ylim_inset = [0.9, 0.93]
inset_position = [0.1, 0.65, 0.35, 0.3]
plot_with_custom_inset(filepath, xlim_inset, ylim_inset, inset_position)
