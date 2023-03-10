import matplotlib.pyplot as plt

# Sample data
data = [10, 15, 20, 25]

# Labels for the bars
labels = ['Bar 1']

# Colors for each bar
colors = ['red', 'blue', 'green', 'orange']
descriptions = ['Data 1', 'Data 2', 'Data 3', 'Data 4']
ranges = [(1, 10), (11, 15), (16, 20), (21, 25)]

# Create the bar chart
plt.barh(labels, data[0], color=colors[0], height=10)

# Loop through the remaining data values and add them to the chart
for i in range(1, len(data)):
    plt.barh(labels, data[i], left=sum(data[:i]), color=colors[i], height=10)

# Add a title
plt.title('Stacked Horizontal Bar Graph')

plt.ylim([-20, 20])

# Create the legend
handles = []
for i in range(len(colors)):
    handles.append(plt.Rectangle((0,0),1,1, color=colors[i]))
plt.legend(handles, descriptions)


# # Add the pointing ranges
# for i in range(len(colors)):
#     plt.annotate(descriptions[i], xy=(sum(data[:i+1]), 0), xytext=((ranges[i][0] + ranges[i][1])/2, -1.5),
#                  arrowprops=dict(arrowstyle='<->', lw=1.5, color=colors[i]), ha='center')

# Add the connecting lines
for i in range(len(colors)):
    plt.plot([sum(data[:i+1]), sum(data[:i+1])], [-20, 20], linestyle='--', color=colors[i], linewidth=1)
    plt.plot([sum(data[:i]), sum(data[:i+1])], [0, 0], linestyle='-', color=colors[i], linewidth=1)


# Show the chart
plt.show()
