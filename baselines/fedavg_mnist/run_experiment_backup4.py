import subprocess
import os
import time

# Function to create a cgroup with given CPU quota and period
def create_cgroup(group_name, cpu_quota, cpu_period):
    subprocess.run(['sudo', 'cgcreate', '-g', 'cpu:/' + group_name], check=True)
    subprocess.run(['sudo', 'cgset', '-r', f'cpu.cfs_quota_us={cpu_quota}', group_name], check=True)
    subprocess.run(['sudo', 'cgset', '-r', f'cpu.cfs_period_us={cpu_period}', group_name], check=True)

# Function to start a process within a cgroup
def start_process(command, group_name):
    # Start the process and return the Popen object without waiting
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a bit for the process to start up and register its PID
    time.sleep(5)  # Sleep to allow the process to start

    # Apply the cgroup classification to the process
    subprocess.run(['sudo', 'cgclassify', '-g', 'cpu:/' + group_name, str(process.pid+1)], check=True)
    
    return process

# Main logic
group_name = "fedops"
total_cores = os.cpu_count()
cpu_period = 100000  # default period of 100ms in microseconds

# Start a list to keep track of the process objects
processes = []

# Define your CPU core allocations (as a fraction of total cores)
core_allocations = [4.5, 0.5]  # For example, 1.5 cores, 0.5 core

# Create a cgroup for each CPU limit and start the processes
for i, cores in enumerate(core_allocations):
    cpu_quota = int((cores / total_cores) * cpu_period * total_cores)  # Calculate the quota as cores fraction of period
    group_name_with_cid = f"{group_name}_cid_{i}"
    create_cgroup(group_name_with_cid, cpu_quota, cpu_period)

    # Start the process within the cgroup
    cmd = f"python client.py --cid {i}"
    process = start_process(cmd, group_name_with_cid)
    processes.append(process)  # Keep track of the started processes

    print(f"Started process with command '{cmd}' within cgroup '{group_name_with_cid}' with quota {cpu_quota}us and period {cpu_period}us")

# You can now monitor the list of processes or perform other tasks
# If you want to wait for all of them to complete, you can do that here
for process in processes:
    process.communicate()  # This will wait for each process to complete
