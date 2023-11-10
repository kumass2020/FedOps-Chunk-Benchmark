import subprocess
import os
import time

def start_process(command, cpu_limit):
    # Start the process
    process = subprocess.Popen(command, shell=True)
    
    # Check if the process has started and get the pid
    while process.poll() is None:
        time.sleep(0.1)  # Sleep for 100 milliseconds to avoid tight loop
        try:
            pid = process.pid
            # Apply CPU limit to the process using subprocess.run to ensure it executes properly
            subprocess.run(['cpulimit', '-l', str(cpu_limit), '-p', str(pid+1)])
            print(f"cpulimit applied to PID {pid+1} with limit {cpu_limit}%")
            break  # If cpulimit was applied successfully, break the loop
        except Exception as e:
            print(f"Error applying cpulimit: {e}")
            continue  # If there was an error, try again

    return process.pid

# Define your commands and CPU limits based on your system's core count
total_cores = os.cpu_count()
cores_list = [4, 0.5]  # The number of cores you wish to allocate
commands = []

# Calculate CPU percentage based on the number of cores
for i, cores in enumerate(cores_list):
    cpu_percent = (cores / total_cores) * 100
    commands.append((f"python client.py --cid {i}", cpu_percent))

# Start all processes with their respective CPU limits
for cmd, cpu_percent in commands:
    pid = start_process(cmd, cpu_percent)
    print(f"Started process {pid+1} with command '{cmd}' limited to {cpu_percent}% of CPU usage")
