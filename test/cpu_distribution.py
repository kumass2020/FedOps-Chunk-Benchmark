import numpy as np

# Original CPU core and percentage data
cpu_cores = np.array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 32, 36, 44, 48, 56, 64, 128])
percentages = np.array([0.15, 9.95, 0.39, 29.51, 0.02, 32.08, 0.02, 19.55, 0.01, 1.91, 0.0, 3.42, 0.0, 1.71, 0.0, 1.06, 0.02, 0.01, 0.0, 0.18, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])

# Scale down the distribution
scale_factor = 5.947 / 1.5
scaled_cpu_cores = cpu_cores / scale_factor
scaled_percentages = percentages / np.sum(percentages)

# Generate 50 samples from the scaled distribution
num_pods = 50
pod_cpu_limits = np.round(scaled_cpu_cores * 1000).astype(int)
pod_cpu_limits = np.random.choice(pod_cpu_limits, num_pods, p=scaled_percentages)

print(pod_cpu_limits[0])