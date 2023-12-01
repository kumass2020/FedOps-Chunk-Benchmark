import subprocess
import os
import time

# Function to start a process with a specific CPU usage limit
def start_process(command, cpu_limit):
    # Start the process in a new shell
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Wait a bit for the process to start up and register its PID
    time.sleep(1)  # Sleep for 500 milliseconds

    # Apply CPU limit to the process
    cpulimit_command = ['cpulimit', '-l', str(cpu_limit), '-p', str(process.pid)]
    cpulimit_process = subprocess.Popen(cpulimit_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    
    # Return both Popen objects
    return process, cpulimit_process

# Define your commands and CPU limits based on your system's core count
total_cores = os.cpu_count()
cores_list = [10, 0.5]  # The number of cores you wish to allocate
commands = []

# Calculate CPU percentage based on the number of cores
for i, cores in enumerate(cores_list):
    cpu_percent = (cores / total_cores) * 100
    cmd = f"python client.py --cid {i}"
    commands.append((cmd, cpu_percent))

# Start all processes with their respective CPU limits
processes = []
for cmd, cpu_percent in commands:
    process, cpulimit_process = start_process(cmd, cpu_percent)
    processes.append((process, cpulimit_process))
    print(f"Started process {process.pid} with command '{cmd}' limited to {cpu_percent}% of CPU usage")

# Optionally, wait for all processes to complete
for process, cpulimit_process in processes:
    process.wait()
    cpulimit_process.terminate()  # Terminate cpulimit process if the main process is done
