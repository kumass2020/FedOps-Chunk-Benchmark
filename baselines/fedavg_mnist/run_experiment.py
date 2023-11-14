import subprocess
import os
import time
import psutil

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
    time.sleep(0.5)  # Sleep to allow the process to start

    # Use psutil to wait for child processes to be created
    parent = psutil.Process(process.pid)
    children = parent.children(recursive=True)
    while not children:
        time.sleep(0.1)
        children = parent.children(recursive=True)
    
    # Apply the cgroup classification to the child process
    for child in children:
        subprocess.run(['sudo', 'cgclassify', '-g', 'cpu:/' + group_name, str(child.pid)], check=True)
        print('command:', 'sudo', 'cgclassify', '-g', 'cpu:/' + group_name, str(child.pid))

    return process

# Function to terminate all running processes
def terminate_processes(processes):
    for process in processes:
        process.terminate()  # Send SIGTERM to the process
        process.wait()  # Wait for the process to terminate

# Main logic
try:
    group_name = "fedops"
    total_cores = os.cpu_count()
    cpu_period = 100000  # default period of 100ms in microseconds

    # Start a list to keep track of the process objects
    processes = []

    # Define your CPU core allocations (as a fraction of total cores)
    core_allocations = [548, 318, 471, 854, 1000, 1000, 854, 471, 1000, 1000, 471, 1000, 816, 471, 777, 1000, 816, 854, 1000, 854, 586, 1000, 1000, 624, 701, 1000, 586, 1000, 1000, 624, 1000, 586, 1000, 624, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 777, 1000, 1000, 624, 1000, 1000, 1000, 1000, 1000]

    # Create a cgroup for each CPU limit and start the processes
    for i, cores in enumerate(core_allocations):
        cpu_quota = int((cores * 0.5 / 1000 / total_cores) * cpu_period * total_cores)  # Calculate the quota
        group_name_with_cid = f"{group_name}_cid_{i}"
        create_cgroup(group_name_with_cid, cpu_quota, cpu_period)

        # Start the process within the cgroup
        cmd = f"python client.py --cid {i}"
        process = start_process(cmd, group_name_with_cid)
        processes.append(process)  # Keep track of the started processes

        print(f"Started process with command '{cmd}' within cgroup '{group_name_with_cid}' with quota {cpu_quota}us and period {cpu_period}us")

    # Monitor the processes
    for process in processes:
        process.communicate()  # This will wait for each process to complete

except Exception as e:
    print(f"An error occurred: {e}")
    terminate_processes(processes)
    raise  # Re-raise the exception to handle it as needed
