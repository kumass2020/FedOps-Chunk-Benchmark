import subprocess
import os
import time
import psutil
import datetime


def create_cgroup(group_name, cpu_quota, cpu_period):
    subprocess.run(['sudo', 'cgcreate', '-g', 'cpu:/' + group_name], check=True)
    subprocess.run(['sudo', 'cgset', '-r', f'cpu.cfs_quota_us={cpu_quota}', group_name], check=True)
    subprocess.run(['sudo', 'cgset', '-r', f'cpu.cfs_period_us={cpu_period}', group_name], check=True)

# Function to start a process within a cgroup
def start_process(command, group_name, log_file):
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

    # Return the original subprocess.Popen object, not the psutil.Process object
    return process, log_file


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
    # core_allocations = [548, 318, 471, 854, 1000, 1000, 854, 471, 1000, 1000, 471, 1000, 816, 471, 777, 1000, 816, 854, 1000, 854, 586, 1000, 1000, 624, 701, 1000, 586, 1000, 1000, 624, 1000, 586, 1000, 624, 1000, 1000, 1000, 1000, 1000, 1000, 1000, 777, 1000, 1000, 624, 1000, 1000, 1000, 1000, 1000]
    # core_allocations = [3721, 930, 2230, 2804, 3760, 1274, 1542, 586, 586, 624, 1427, 1580, 1618, 1657, 3492, 1657, 586, 586, 624, 1542, 510, 892, 2268, 624, 1618, 548, 2192, 892, 1198, 2230, 2230, 892, 1733, 624, 471, 624, 854, 816, 1121, 1465, 1007, 1274, 777, 1580, 548, 2115, 1618, 1580, 1771, 892]
    core_allocations = [1580, 357, 892, 2804, 854, 624, 854, 3454, 892, 1580, 1045, 1542, 1274, 510, 892, 1236, 1618, 1618, 1313, 1771, 586, 1618, 1657, 739, 1504]

    # Test
    core_allocations = [357, 892, 2804, 854, 624, 854, 3454, 892, 1580, 1045, 1542, 1274, 510, 892, 1236, 1618, 1618, 1313, 1771, 586, 1618, 1657, 739, 1504]

    # Create log directory outside the loop
    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    log_dir = f"./logs/{now}"
    os.makedirs(log_dir, exist_ok=True)

    # Create a cgroup for each CPU limit and start the processes
    for i, cores in enumerate(core_allocations):
        cpu_quota = int((cores * 0.7 / 1000 / total_cores) * cpu_period * total_cores)
        group_name_with_cid = f"{group_name}_cid_{i}"
        create_cgroup(group_name_with_cid, cpu_quota, cpu_period)

        cmd = f"python -m /home/hoho/github/FedOps-Chunk-Benchmark/baselines/fedprox/fedprox/client.py cid={i}"
        log_file = f"{log_dir}/client_{i}.log"
        process, log_file_path = start_process(cmd, group_name_with_cid, log_file)
        processes.append((process, log_file_path))

        # Print the details of the started process
        print(f"Started process with command '{cmd}' within cgroup '{group_name_with_cid}' with quota {cpu_quota}us and period {cpu_period}us")

    # Monitor the processes and write logs
    for process, log_file in processes:
        stdout, stderr = process.communicate()  # Wait for process to complete
        with open(log_file, 'w') as log:
            log.write(stdout.decode())
            if stderr:
                log.write("\n--- STDERR ---\n")
                log.write(stderr.decode())

except Exception as e:
    print(f"An error occurred: {e}")
    terminate_processes([p for p, _ in processes])
    raise