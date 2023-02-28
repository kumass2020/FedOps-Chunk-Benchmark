import subprocess

server_container_name = 'fedops-server-mjh'  # Define the container name
container_name = 'fedops-client-mjh-'

subprocess.Popen(
    ['docker', 'run',
     '--name', server_container_name,
     '--cpus', '1',
     '--memory', '1G',
     '--cpu-shares', '2',
     '--memory-reservation', '250m',
     '--network', 'fedops-mjh',
     'kumass2020/fedops-server:10-5-client'])

for i in range(10):
    subprocess.Popen(
        ['docker', 'run',
         '--name', container_name + str(i),
         '--cpus', '0.5',
         '--memory', '1G',
         '--network', 'fedops-mjh',
         'kumass2020/fedops-client:docker'])
