import subprocess

# Stop all running containers created from the 'kumass2020/fedops-server' image
subprocess.run('docker stop $(docker ps --filter "ancestor=kumass2020/fedops-server" -q)', shell=True)
subprocess.run('docker stop $(docker ps --filter "ancestor=kumass2020/fedops-client" -q)', shell=True)

# Remove all containers created from the 'kumass2020/fedops-server' image
subprocess.run('docker rm $(docker ps --filter "ancestor=kumass2020/fedops-server" -aq)', shell=True)
subprocess.run('docker rm $(docker ps --filter "ancestor=kumass2020/fedops-client" -aq)', shell=True)
