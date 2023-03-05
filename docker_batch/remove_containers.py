import docker

client = docker.from_env()

# Define the name patterns for the containers you want to remove
server_name = "fedops-server-mjh"
client_name_pattern = "fedops-client-mjh-*"

# Define the image name for the client image
client_image = "kumass2020/fedops-client"

# Stop and remove containers created from the server image
for container in client.containers.list(all=True, filters={'name': server_name}):
    print(f"Stopping container {container.name}...")
    container.stop()
    print(f"Removing container {container.name}...")
    container.remove()

# Stop and remove containers created from the client image
for container in client.containers.list(all=True, filters={'name': client_name_pattern}):
    print(f"Stopping container {container.name}...")
    container.stop()
    print(f"Removing container {container.name}...")
    container.remove()