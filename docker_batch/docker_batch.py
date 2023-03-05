import subprocess
import time

server_container_name = 'fedops-server-mjh'  # Define the container name
container_name = 'fedops-client-mjh-'

network_name = 'fedops-mjh'

# Check if the network exists
network_list = subprocess.run(['docker', 'network', 'ls', '--format', '{{.Name}}'], capture_output=True, text=True).stdout.splitlines()
if network_name not in network_list:
    # Create the network if it doesn't exist
    subprocess.run(['docker', 'network', 'create', network_name], check=True)

server_process = subprocess.Popen(
    ['docker', 'run',
     '--name', server_container_name,
     # '--cpus', '1',
     # '--memory', '1G',
     # '--cpu-shares', '2',
     # '--memory-reservation', '250m',
     '--network', 'fedops-mjh',
     'kumass2020/fedops-server:tf'])

# server_process.wait()
time.sleep(5)

for i in range(10):
    client_process = subprocess.Popen(
        ['docker', 'run',
         '--name', container_name + str(i),
         '--cpus', '1',
         '--memory', '1G',
         '--network', 'fedops-mjh',
         'kumass2020/fedops-client:tf-docker'])
    time.sleep(0.5)
    # client_process.wait()
