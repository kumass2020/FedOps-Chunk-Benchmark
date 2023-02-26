from kubernetes import client, config

if __name__ == '__main__':
    config.load_kube_config('config_ssh.txt')

    v1 = client.CoreV1Api()
    print(v1.list_node())
