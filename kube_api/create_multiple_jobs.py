from kubernetes import client, config
import time

JOB_NAME = "fedops-client-mjh"
SERVER_JOB_NAME = "fedops-server-mjh"
service_account_name = "fedops-svc-mjh"

jobs_num = 1


def create_containers():
    container_list = []
    cpu = 0
    memory = 0

    for i in range(jobs_num):
        container = client.V1Container(
            name=f"fedops-client-{i}",
            image="kumass2020/fedops-client:latest",
            # image_pull_policy="Always",
            # command=["perl", "-Mbignum=bpi", "-wle", "print bpi(2000)"]
            resources=client.V1ResourceRequirements(
                requests={"cpu": "4000m", "memory": "2Gi"},
                limits={"cpu": "8000m", "memory": "4Gi"}
            ),
            volume_mounts=[client.V1VolumeMount(name="airflow-nfs", mount_path="/mnt/fedops-mjh")]
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
            volumes=[client.V1Volume(
                name="airflow-nfs",
                persistent_volume_claim=client.V1PersistentVolumeClaimVolumeSource(
                    claim_name="fedops-pvc-mjh",
                    )
            )],
            node_name=node_name
        ))

    # Create the specification of deployment
    spec = client.V1JobSpec(
        template=template,
        parallelism=3,
        completions=3,
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
        image="kumass2020/fedops-server:latest",
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

    # Create a job object with client-python API. The job we
    # created is same as the `pi-job.yaml` in the /examples folder.
    # job_list: list[client.V1Job] = []
    job = create_server_job_object()
    create_job(batch_v1, job)
    time.sleep(3)
    for i in range(jobs_num):
        job = create_job_object(i)
        # job.metadata = client.V1ObjectMeta(name=JOB_NAME + str(i))
        # job_list.append(job)
        create_job(batch_v1, job)
        time.sleep(0.5)


    # create_job(batch_v1, job)

    # update_job(batch_v1, job)

    # delete_job(batch_v1)


if __name__ == '__main__':
    main()