import subprocess
import os
import time
import psutil
import datetime
import threading


def create_cgroup(group_name, cpu_quota, cpu_period):
    subprocess.run(['sudo', 'cgcreate', '-g', 'cpu:/' + group_name], check=True)
    subprocess.run(['sudo', 'cgset', '-r', f'cpu.cfs_quota_us={cpu_quota}', group_name], check=True)
    subprocess.run(['sudo', 'cgset', '-r', f'cpu.cfs_period_us={cpu_period}', group_name], check=True)

# Function to start a process within a cgroup
def start_process(command, group_name, log_file):
    # Start the process and return the Popen object without waiting
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)

    time.sleep(0.2)

    # Use psutil to wait for child processes to be created
    parent = psutil.Process(process.pid)
    children = parent.children(recursive=True)
    count = 0
    while not children and count < 3:
        time.sleep(0.1)
        children = parent.children(recursive=True)
        count += 1
    
    # Apply the cgroup classification to the child process
    if children:
        for child in children:
            subprocess.run(['sudo', 'cgclassify', '-g', 'cpu:/' + group_name, str(child.pid)], check=True)
            print('command:', 'sudo', 'cgclassify', '-g', 'cpu:/' + group_name, str(child.pid))
    
    # Apply the cgroup classification to the process
    subprocess.run(['sudo', 'cgclassify', '-g', 'cpu:/' + group_name, str(process.pid)], check=True)
    print('command:', 'sudo', 'cgclassify', '-g', 'cpu:/' + group_name, str(process.pid))

    return process, log_file


# Function to terminate all running processes
def terminate_processes(processes):
    for process in processes:
        process.terminate()  # Send SIGTERM to the process
        process.wait()  # Wait for the process to terminate


def write_logs(process, log_file_path):
    def write_stream(stream, log_file):
        for line in iter(stream.readline, b''):  # Read until an empty byte is found
            log_file.write(line.decode())  # Decode from bytes to str before writing
            log_file.flush()

    try:
        with open(log_file_path, 'w') as log_file:
            stdout_thread = threading.Thread(target=write_stream, args=(process.stdout, log_file))
            stderr_thread = threading.Thread(target=write_stream, args=(process.stderr, log_file))

            stdout_thread.start()
            stderr_thread.start()

            stdout_thread.join()
            stderr_thread.join()
    except Exception as e:
        print(f"Error writing logs: {e}")



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
    # core_allocations = [1580, 357, 892, 2804, 854, 624, 854, 3454, 892, 1580, 1045, 1542, 1274, 510, 892, 1236, 1618, 1618, 1313, 1771, 586, 1618, 1657, 739, 1504]

    # 30 clients
    core_allocations = [930, 1542, 892, 1236, 968, 2727, 1580, 624, 1465, 1274, 1618, 1580, 3530, 777, 586, 1274, 892, 1045, 1542, 1504, 357, 1618, 624, 816, 1427, 2765, 2230, 968, 624, 854]

    # 20 clients
    # core_allocations = [1389, 1045, 1733, 1618, 1274, 739, 471, 854, 968, 624, 510, 357, 1121, 3492, 1121, 892, 1236, 1313, 1504, 854]

    # Test
    # core_allocations = [357, 892, 2804, 854, 624, 854, 3454, 892, 1580, 1045, 1542, 1274, 510, 892, 1236, 1618, 1618, 1313, 1771, 586, 1618, 1657, 739, 1504]

    # Create log directory outside the loop
    now = datetime.datetime.now().strftime("%y%m%d_%H%M%S")
    log_dir = f"./logs/{now}"
    os.makedirs(log_dir, exist_ok=True)

    # Create a cgroup for each CPU limit and start the processes
    for i, cores in enumerate(core_allocations):
        cpu_quota = int((cores / 1000 / total_cores) * cpu_period * total_cores)
        group_name_with_cid = f"{group_name}_cid_{i}"
        create_cgroup(group_name_with_cid, cpu_quota, cpu_period)

        cmd = f"python -m fedprox.client cid={i}"
        log_file = f"{log_dir}/client_{i}.log"
        process, log_file_path = start_process(cmd, group_name_with_cid, log_file)
        processes.append((process, log_file_path))

        # Print the details of the started process
        print(f"Started process with command '{cmd}' within cgroup '{group_name_with_cid}' with quota {cpu_quota}us and period {cpu_period}us")

    # Monitor the processes and write logs
    for process, log_file in processes:
        write_logs_thread = threading.Thread(target=write_logs, args=(process, log_file))
        write_logs_thread.start()
        write_logs_thread.join()

except Exception as e:
    print(f"An error occurred: {e}")
    terminate_processes([p for p, _ in processes])
    raise