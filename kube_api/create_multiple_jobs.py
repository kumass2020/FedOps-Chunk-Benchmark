from kubernetes import client, config
import time
import numpy as np
import random

JOB_NAME = "fedops-client-mjh"
SERVER_JOB_NAME = "fedops-server-mjh"
service_account_name = "fedops-svc-mjh"

jobs_num = 50

client_image = "kumass2020/fedops-client:v23"
server_image = "kumass2020/fedops-server:client50-v24"

# client_image_pull_policy = "IfNotPresent"
# server_image_pull_policy = "IfNotPresent"
client_image_pull_policy = "Always"
server_image_pull_policy = "Always"

# delay_after_server = 540
delay_after_server = 10
delay_per_client = 0.5

cpu_limits_list: list[int]


def get_cpu_distribution():
    import csv

    # Open the CSV file in read mode
    with open('ML_ALL_benchmarks.csv', mode='r') as csv_file:
        # Create a CSV reader object
        csv_reader = csv.reader(csv_file)

        # Initialize an empty list to hold the data
        data = []

        # Loop through each row in the CSV file
        for row in csv_reader:
            # Extract the value of the second column and append it to the list
            data.append(row[5])

    # Print the list of data
    # print(data[1:], "\n", len(data[1:]))

    # Convert the list of strings to a list of integers
    data = [int(x) for x in data[1:]]

    # data = data[1:]
    mean = sum(data) / len(data)

    new_mean = 1400

    # Scale each value in the list to have a mean of 1400
    scaled_numbers = [(value - mean) * (new_mean / mean) + new_mean for value in data]

    # Print the scaled list of numbers
    print(scaled_numbers)
    print(sum(scaled_numbers) / len(scaled_numbers))

    X = scaled_numbers
    # X = [np.random.normal(loc=0, scale=1) for i in range(188)]

    # Convert the list to an ndarray
    X = np.array(X)

    # Calculate the histogram of the original data
    hist, bins = np.histogram(X, bins=100)

    # Compute the probability of each bin being chosen
    prob = hist / len(X)

    # Sample 50 elements with replacement based on the probability of each bin
    X_sampled = np.random.choice(bins[:-1], size=50, replace=True, p=prob)

    # pod_cpu_limits = random.sample(scaled_numbers, 50)
    pod_cpu_limits = [int(x) for x in X_sampled]
    print(pod_cpu_limits)

    return pod_cpu_limits

# def get_cpu_distribution():
#     # Original CPU core and percentage data
#     cpu_cores = np.array(
#         [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 32, 36, 44, 48, 56, 64, 128])
#     percentages = np.array(
#         [0.15, 9.95, 0.39, 29.51, 0.02, 32.08, 0.02, 19.55, 0.01, 1.91, 0.0, 3.42, 0.0, 1.71, 0.0, 1.06, 0.02, 0.01,
#          0.0, 0.18, 0.0, 0.0, 0.01, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
#
#     # Scale down the distribution
#     scale_factor = 5.947 / 1.4
#     scaled_cpu_cores = cpu_cores / scale_factor
#     scaled_percentages = percentages / np.sum(percentages)
#
#     # Generate 50 samples from the scaled distribution
#     num_pods = 50
#     pod_cpu_limits = np.round(scaled_cpu_cores * 1000).astype(int)
#     pod_cpu_limits = np.random.choice(pod_cpu_limits, num_pods, p=scaled_percentages)
#
#     print(pod_cpu_limits)
#     return pod_cpu_limits


def create_containers():
    container_list = []
    cpu = 0
    memory = 0

    for i in range(jobs_num):
        container = client.V1Container(
            name=f"fedops-client-{i}",
            image=client_image,
            image_pull_policy=client_image_pull_policy,
            # command=["sh", "-c", "mkdir -p /home/ccl/fedops-mjh/mnt"],
            resources=client.V1ResourceRequirements(
                requests={"cpu": str(cpu_limits_list[i]) + "m", "memory": "1Gi"},
                limits={"cpu": str(cpu_limits_list[i]) + "m", "memory": "1Gi"}
            ),
            env=[
                client.V1EnvVar(name="CLIENT_NUMBER", value=str(i))
            ]
            # volume_mounts=[client.V1VolumeMount(name="airflow-nfs", mount_path="/mnt/fedops-mjh")]
            # volume_mounts=[
            #     client.V1VolumeMount(
            #         name="microk8s-hostpath",
            #         mount_path="/home/ccl/fedops-mjh/mnt"
            #     )
            # ]
        )
        container_list.append(container)

    return container_list


def create_job_object(job_num: int):
    node_name = ''
    if job_num % 3 == 0:
        node_name = "ccl-d-server"
    elif job_num % 3 == 1:
        node_name = "ccl-e-server"
    elif job_num % 3 == 2:
        node_name = "ccl-x-server"

    node_name = "ccl-d-server"

    # Configureate Pod template container
    container_list = create_containers()

    # Create and configure a spec section
    template = client.V1PodTemplateSpec(
        # metadata=client.V1ObjectMeta(labels={"app": "fedops-client-mjh" + str(job_num)}),
        metadata=client.V1ObjectMeta(labels={"app": "fedops-client-mjh"}),
        spec=client.V1PodSpec(
            restart_policy="Never",
            containers=[container_list[job_num]],
            # volumes=[client.V1Volume(
            #     name="airflow-nfs",
            #     persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
            #         claim_name="fedops-pvc-mjh"
            #         )
            # )],
            # volumes=[
            #     client.V1Volume(
            #         name="microk8s-hostpath",
            #         host_path=client.V1HostPathVolumeSource(path="home/ccl/fedops-mjh/mnt")
            #     )
            # ]
            # node_name=node_name,
            # preemption_policy="Never"
        ))

    # Create the specification of deployment
    spec = client.V1JobSpec(
        template=template,
        # parallelism=3,
        # completions=3,
        backoff_limit=4)

    # Instantiate the job object
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=JOB_NAME + str(job_num)),
        spec=spec)

    return job


def create_server_job_object():
    # Configureate Pod template container
    container = client.V1Container(
        name="fedops-server",
        image=server_image,
        image_pull_policy=server_image_pull_policy,
        # command=["perl", "-Mbignum=bpi", "-wle", "print bpi(2000)"]
        resources=client.V1ResourceRequirements(
            requests={"cpu": "1000m", "memory": "1Gi"},
            limits={"cpu": "4000m", "memory": "4Gi"}
        )
    )
    # Create and configure a spec section
    # template = client.V1PodTemplateSpec(
    #     metadata=client.V1ObjectMeta(labels={"app": "fedops-mjh"}),
    #     spec=client.V1PodSpec(restart_policy="Never", containers=[container]))

    # Create Service
    service_spec = client.V1ServiceSpec(
        type="LoadBalancer",
        selector={"app": "fedops-mjh"},
        ports=[client.V1ServicePort(port=80, target_port=80)]
    )
    service_meta = client.V1ObjectMeta(name="fedops-service")
    service = client.V1Service(api_version="v1", kind="Service", metadata=service_meta, spec=service_spec)
    template = client.V1PodTemplateSpec(
        metadata=client.V1ObjectMeta(labels={"app": "fedops-mjh"}),
        spec=client.V1PodSpec(
            restart_policy="Never",
            containers=[container],
            service_account_name=service_account_name
        )
    )

    # Create the specification of deployment
    spec = client.V1JobSpec(
        template=template,
        backoff_limit=4)

    # applied with service
    spec.template.metadata.annotations = {"prometheus.io/scrape": "true", "prometheus.io/path": "/metrics"}
    spec.template.metadata.labels = {"app": "fedops-mjh"}
    spec.template.spec.containers[0].ports = [
        client.V1ContainerPort(container_port=8080)
    ]
    spec.template.spec.restart_policy = "Never"
    spec.template.spec.service_account_name = service_account_name
    spec.template.spec.service_account = service_account_name
    spec.template.spec.service = service

    # Instantiate the job object
    job = client.V1Job(
        api_version="batch/v1",
        kind="Job",
        metadata=client.V1ObjectMeta(name=SERVER_JOB_NAME),
        spec=spec)

    return job


def create_job(api_instance, job):
    api_response = api_instance.create_namespaced_job(
        body=job,
        namespace="fedops")
    print("Job created. status='%s'" % str(api_response.status))
    # get_job_status(api_instance)


def get_job_status(api_instance):
    job_completed = False
    while not job_completed:
        api_response = api_instance.read_namespaced_job_status(
            name=JOB_NAME,
            namespace="fedops")
        if api_response.status.succeeded is not None or \
                api_response.status.failed is not None:
            job_completed = True
        time.sleep(1)
        print("Job status='%s'" % str(api_response.status))


def update_job(api_instance, job):
    # Update container image
    job.spec.template.spec.containers[0].image = "perl"
    api_response = api_instance.patch_namespaced_job(
        name=JOB_NAME,
        namespace="default",
        body=job)
    print("Job updated. status='%s'" % str(api_response.status))


def delete_job(api_instance):
    api_response = api_instance.delete_namespaced_job(
        name=JOB_NAME,
        namespace="fedops",
        body=client.V1DeleteOptions(
            propagation_policy='Foreground',
            grace_period_seconds=5))
    print("Job deleted. status='%s'" % str(api_response.status))


def main():
    # Configs can be set in Configuration class directly or using helper
    # utility. If no argument provided, the config will be loaded from
    # default location.
    config.load_kube_config('config_ssh.txt')
    batch_v1 = client.BatchV1Api()

    global cpu_limits_list
    cpu_limits_list = get_cpu_distribution()

    # Create a job object with client-python API. The job we
    # created is same as the `pi-job.yaml` in the /examples folder.
    # job_list: list[client.V1Job] = []
    job = create_server_job_object()
    create_job(batch_v1, job)
    time.sleep(delay_after_server)
    for i in range(jobs_num):
        job = create_job_object(i)
        # job.metadata = client.V1ObjectMeta(name=JOB_NAME + str(i))
        # job_list.append(job)
        create_job(batch_v1, job)
        time.sleep(delay_per_client)


    # create_job(batch_v1, job)

    # update_job(batch_v1, job)

    # delete_job(batch_v1)


if __name__ == '__main__':
    main()