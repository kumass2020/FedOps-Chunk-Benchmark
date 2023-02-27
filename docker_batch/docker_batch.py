import subprocess

container_name = 'fedops-server-mjh'  # Define the container name
subprocess.Popen(['docker', 'run', '--name', container_name, '--cpus', '1', '--memory', '1G', '--cpu-shares', '2', '--memory-reservation', '250m', 'my-image'])
