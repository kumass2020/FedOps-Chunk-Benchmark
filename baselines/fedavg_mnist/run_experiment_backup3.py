import subprocess
import os
import time

# Function to create a cgroup with given CPU shares
def create_cgroup(group_name, cpu_shares):
    subprocess.run(['sudo', 'cgcreate', '-g', f'cpu:/{group_name}'], check=True)
    subprocess.run(['sudo', 'cgset', '-r', f'cpu.shares={cpu_shares}', group_name], check=True)

import psutil

def start_process(command, group_name):
    # Start the process and return the Popen object without waiting
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a bit for the process to start up and register its PID
    time.sleep(10)  # Sleep to allow the process to start

    # Use psutil to wait for child processes to be created
    parent = psutil.Process(process.pid)
    children = parent.children(recursive=True)
    while not children:
        time.sleep(0.1)
        children = parent.children(recursive=True)
    
    # Apply the cgroup classification to the child process
    for child in children:
        subprocess.run(['sudo', 'cgclassify', '-g', f'cpu:/{group_name}', str(child.pid)], check=True)
        print('command:', 'sudo', 'cgclassify', '-g', f'cpu:/{group_name}', str(child.pid))

    # Return the process object
    return process


# Main logic
group_name = "fedops"
total_cores = os.cpu_count()
cores_list = [4.5, 0.1]  # The number of cores you wish to allocate

# Start a list to keep track of the process objects
processes = []

# Create a cgroup for each CPU limit and start the processes
for i, cores in enumerate(cores_list):
    cpu_shares = int((cores / total_cores) * 1024)
    group_name_with_cid = f"{group_name}_cid_{i}"
    create_cgroup(group_name_with_cid, cpu_shares)

    # Start the process within the cgroup
    cmd = f"python client.py --cid {i}"
    process = start_process(cmd, group_name_with_cid)
    processes.append(process)  # Keep track of the started processes

    print(f"Started process with command '{cmd}' within cgroup '{group_name_with_cid}'")

# You can now monitor the list of processes or perform other tasks
# If you want to wait for all of them to complete, you can do that here
for process in processes:
    process.wait()  # This will wait for each process to complete
