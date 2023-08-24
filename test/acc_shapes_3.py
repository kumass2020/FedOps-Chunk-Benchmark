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

    # Main plot
    ax.plot(x, t0, label='Baseline FedAvg')
    ax.plot(x, t3, label='FedAvg + CAT(k=3)')
    ax.plot(x, t5, label='FedAvg + CAT(k=5)')
    ax.plot(x, t7, label='FedAvg + CAT(k=7)')

    # Create inset of specified dimensions and position
    axins = zoomed_inset_axes(ax, zoom=1)  # zoom=1 means no zoom
    ip = InsetPosition(ax, inset_position)  # position: x, y, width, height
    axins.set_axes_locator(ip)
    axins.plot(x, t0)
    axins.plot(x, t3)
    axins.plot(x, t5)
    axins.plot(x, t7)

    # Set the desired x and y limits for the inset
    x1, x2 = xlim_inset
    y1, y2 = ylim_inset
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

    # Display the plot
    plt.show()


# Example usage
filepath = 'wandb_export_2023-03-23T20_48_33.549+09_00.csv'
xlim_inset = [850, 950]
ylim_inset = [0.9, 0.93]
inset_position = [0.1, 0.65, 0.35, 0.3]
plot_with_custom_inset(filepath, xlim_inset, ylim_inset, inset_position)
